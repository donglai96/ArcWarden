#!/usr/bin/env python3
"""CPU-only regression tests for scripts/plot_probe_spectra2d.py."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "plot_probe_spectra2d.py"
SPEC = importlib.util.spec_from_file_location("plot_probe_spectra2d", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
plotter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plotter
SPEC.loader.exec_module(plotter)


class Spectrum2DTests(unittest.TestCase):
    def test_helicity_frequency_and_absolute_power(self):
        dt = 0.6
        nperseg = 1024
        hop = nperseg
        nsample = 8 * nperseg
        omega = 2.0 * np.pi * 7 / (nperseg * dt)
        time = (np.arange(nsample) + 1.0) * dt
        amp_r, amp_l = 1.0e-4, 1.0e-5
        b1 = (amp_r + amp_l) * np.cos(omega * time) + 3.0e-3
        by = (amp_r - amp_l) * np.sin(omega * time) - 7.0e-4
        right, left = plotter.helicity(b1, by)
        sr = plotter.stft_psd(right, dt, nperseg, hop, "constant")
        sl = plotter.stft_psd(left, dt, nperseg, hop, "constant")
        domega = sr.rayleigh_omega
        power_r = float(np.sum(sr.psd, axis=0).mean() * domega)
        power_l = float(np.sum(sl.psd, axis=0).mean() * domega)
        peak = float(sr.omega[np.argmax(sr.psd.mean(axis=1))])
        self.assertLess(abs(peak - omega), 1.0e-12)
        self.assertAlmostEqual(power_r / amp_r**2, 1.0, delta=2.0e-6)
        self.assertAlmostEqual(power_l / amp_l**2, 1.0, delta=2.0e-5)
        self.assertAlmostEqual(10.0 * math.log10(power_r / power_l), 20.0, delta=0.01)

    def test_amplitude_step_survives_without_column_normalization(self):
        dt = 0.6
        nperseg = 1024
        omega = 2.0 * np.pi * 9 / (nperseg * dt)
        time = (np.arange(8 * nperseg) + 1.0) * dt
        amplitude = np.where(np.arange(time.size) < 4 * nperseg, 1.0e-4, 1.0e-5)
        right = amplitude * np.exp(1j * omega * time)
        spectrum = plotter.stft_psd(right, dt, nperseg, nperseg, "constant")
        peak_row = int(np.argmax(spectrum.psd.mean(axis=1)))
        early = float(np.median(spectrum.psd[peak_row, :4]))
        late = float(np.median(spectrum.psd[peak_row, 4:]))
        self.assertAlmostEqual(10.0 * math.log10(early / late), 20.0, delta=0.05)

    def test_loader_ignores_interrupted_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probes.bin"
            complete = np.arange(2 * 6 * 3, dtype="<f4")
            trailing = np.arange(5, dtype="<f4")
            with path.open("wb") as stream:
                complete.tofile(stream)
                trailing.tofile(stream)
            records = plotter.load_probe_records(path, nprobe=3)
            self.assertEqual(records.values.shape, (2, 6, 3))
            self.assertEqual(records.trailing_floats, 5)
            np.testing.assert_array_equal(records.values.ravel(), complete)

    def test_band_mean_interpolates_exact_edges(self):
        q = np.linspace(0.0, 1.0, 18)
        density = np.full_like(q, 7.25)
        for bounds in plotter.DEFAULT_BANDS.values():
            self.assertAlmostEqual(
                plotter._band_mean(q, density, bounds), 7.25, places=13
            )

    def test_zero_power_ratios_are_explicitly_undefined(self):
        self.assertTrue(math.isnan(plotter._safe_linear_ratio(0.0, 0.0)))
        self.assertTrue(math.isnan(plotter._safe_ratio_db(0.0, 0.0)))

    def test_real_component_uses_doubled_one_sided_psd(self):
        dt = 0.6
        nperseg = 1024
        amplitude = 2.3e-4
        omega = 2.0 * np.pi * 11 / (nperseg * dt)
        time = (np.arange(6 * nperseg) + 1.0) * dt
        signal = amplitude * np.cos(omega * time) + 2.0e-3
        doubled = plotter.stft_psd(
            signal,
            dt,
            nperseg,
            nperseg,
            "constant",
            real_one_sided=True,
        )
        undoubled = plotter.stft_psd(
            signal,
            dt,
            nperseg,
            nperseg,
            "constant",
            real_one_sided=False,
        )
        domega = doubled.rayleigh_omega
        power_doubled = float(np.sum(doubled.psd, axis=0).mean() * domega)
        power_undoubled = float(np.sum(undoubled.psd, axis=0).mean() * domega)
        self.assertAlmostEqual(power_doubled / (0.5 * amplitude**2), 1.0, delta=2e-6)
        self.assertAlmostEqual(power_doubled / power_undoubled, 2.0, delta=1e-12)

    def test_real_component_rejects_complex_input(self):
        signal = np.ones(64) + 1j * np.ones(64)
        with self.assertRaisesRegex(ValueError, "real-valued"):
            plotter.stft_psd(signal, 0.6, 32, 16, real_one_sided=True)

    def test_component_selector_uses_writer_channel_contract(self):
        values = np.zeros((8, 6, 2), dtype=np.float32)
        for channel in range(6):
            values[:, channel, 1] = 10.0 + channel
        bpar, auxiliary, is_real = plotter.probe_component_signals(values, 1, "bpar")
        self.assertTrue(is_real)
        self.assertIsNone(auxiliary)
        np.testing.assert_array_equal(bpar, 13.0)
        b1, _, _ = plotter.probe_component_signals(values, 1, "b1")
        by, _, _ = plotter.probe_component_signals(values, 1, "by")
        np.testing.assert_array_equal(b1, 14.0)
        np.testing.assert_array_equal(by, 15.0)
        right, left, is_real = plotter.probe_component_signals(values, 1, "r")
        self.assertFalse(is_real)
        np.testing.assert_array_equal(right, 14.0 + 15.0j)
        np.testing.assert_array_equal(left, 14.0 - 15.0j)

    def test_circular_power_equals_sum_of_two_real_components(self):
        dt = 0.6
        nperseg = 1024
        amplitude = 1.7e-4
        omega = 2.0 * np.pi * 13 / (nperseg * dt)
        time = (np.arange(4 * nperseg) + 1.0) * dt
        b1 = amplitude * np.cos(omega * time)
        by = amplitude * np.sin(omega * time)
        right, _ = plotter.helicity(b1, by)
        sr = plotter.stft_psd(right, dt, nperseg, nperseg, "constant")
        s1 = plotter.stft_psd(
            b1, dt, nperseg, nperseg, "constant", real_one_sided=True
        )
        sy = plotter.stft_psd(
            by, dt, nperseg, nperseg, "constant", real_one_sided=True
        )
        domega = sr.rayleigh_omega
        pr = float(sr.psd.sum(axis=0).mean() * domega)
        p1 = float(s1.psd.sum(axis=0).mean() * domega)
        py = float(sy.psd.sum(axis=0).mean() * domega)
        self.assertAlmostEqual(pr / (p1 + py), 1.0, delta=2e-12)
        self.assertAlmostEqual(pr / p1, 2.0, delta=2e-12)

    def test_amplitude_normalization_does_not_change_axes(self):
        psd = np.array([1.2e-8, 4.7e-7])
        bref, bloc = 0.2, 0.25
        d_ref = plotter._db_density(psd, bref)
        d_local = plotter._db_density(psd, bloc)
        expected = 20.0 * math.log10(bloc / bref)
        np.testing.assert_allclose(d_ref - d_local, expected, rtol=0.0, atol=1e-12)
        q_ref = plotter.positive_frequency_q(1024, 0.6, bref)
        q_again = plotter.positive_frequency_q(1024, 0.6, bref)
        np.testing.assert_array_equal(q_ref, q_again)

    def test_default_output_directories_isolate_components(self):
        base = Path("/tmp/example")
        parser = plotter.build_parser()
        r_args = parser.parse_args([str(base)])
        by_args = parser.parse_args([str(base), "--component", "by"])
        b1_args = parser.parse_args([str(base), "--component", "b1"])
        self.assertEqual(plotter._default_destination(base, r_args).name, "probe_spectra")
        self.assertEqual(plotter._default_destination(base, by_args).name, "probe_spectra_by")
        self.assertEqual(plotter._default_destination(base, b1_args).name, "probe_spectra_b1")

    def test_explicit_output_refuses_cross_component_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "probe_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "analysis_complete": True,
                        "signal": {"component": "by"},
                        "analysis": {"amplitude_normalization": "reference"},
                    }
                ),
                encoding="utf-8",
            )
            plotter._guard_generation_identity(manifest, "by", "reference")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                plotter._guard_generation_identity(manifest, "b1", "reference")


if __name__ == "__main__":
    unittest.main()
