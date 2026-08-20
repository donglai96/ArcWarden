#!/usr/bin/env python3
"""Reproducible magnetic spectra for every warden2d probe.

The probe writer stores one float32 record as ``[6 fields][N probes]`` with
fields ``Epar,E1,Ey,Bpar,B1,By``.  This script freezes the complete-record
prefix of each file, ignores (and reports) an interrupted trailing record,
By default it forms the fixed helicities

    B_R = B1 + i By,   B_L = B1 - i By,

and computes an absolute two-sided STFT PSD before selecting positive
frequencies.  It can instead plot the real field-aligned, in-plane transverse,
or out-of-plane transverse component.  Real components use a correctly doubled
one-sided PSD.  It deliberately does not whiten rows, normalize columns, or
select a fixed number of peaks: all panels use one caller-controlled dB scale.

Default R-helicity products are written to ``OUTDIR/probe_spectra``.  Scalar
components use isolated directories such as ``probe_spectra_by``:

* one 7-probe SHORT/LONG spectrogram stack per field line;
* one two-resolution spectrogram for every individual probe;
* one time-averaged PSD per field line (plus R/L audit for R helicity);
* ``probe_manifest.json`` and ``probe_spectrum_summary.csv``.

Examples
--------
  python scripts/plot_probe_spectra2d.py build/p1_lurepro2d
  python scripts/plot_probe_spectra2d.py build/p1_lurepro2d --component by
  python scripts/plot_probe_spectra2d.py build/p1_wide --db-min -90 --db-max -30
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHANNELS = ("Epar", "E1", "Ey", "Bpar", "B1", "By")
BPAR_INDEX = 3
B1_INDEX = 4
BY_INDEX = 5
DEFAULT_BANDS = {
    "LB": (0.20, 0.45),
    "gap": (0.45, 0.55),
    "UB": (0.55, 0.75),
}


@dataclass(frozen=True)
class Probe:
    latitude_deg: float
    wce_local: float


@dataclass(frozen=True)
class ProbeLine:
    tag: str
    L: float
    filename: str
    probes: tuple[Probe, ...]


@dataclass(frozen=True)
class RunMeta:
    path: Path
    deck: str
    dt: float
    probe_every: int
    b0eq: float
    l0: float
    fields: tuple[str, ...]
    lines: tuple[ProbeLine, ...]

    @property
    def sample_dt(self) -> float:
        return self.dt * self.probe_every


@dataclass(frozen=True)
class ProbeRecords:
    path: Path
    values: np.ndarray
    trailing_floats: int
    trailing_bytes: int
    frozen_bytes: int
    frozen_mtime_ns: int
    prefix_sha256: str

    @property
    def nrecords(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class Spectrum:
    omega: np.ndarray
    time: np.ndarray
    psd: np.ndarray
    nperseg: int
    hop: int
    sample_dt: float

    @property
    def rayleigh_omega(self) -> float:
        return 2.0 * np.pi / (self.nperseg * self.sample_dt)


def _require_float(pattern: str, text: str, name: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing {name!r} in meta file")
    return float(match.group(1))


def _line_tag(L: float) -> str:
    if abs(L - round(L)) < 1.0e-6:
        return f"L{int(round(L))}"
    return "L" + f"{L:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _latitude_tag(latitude_deg: float) -> str:
    if abs(latitude_deg) < 0.05:
        return "eq"
    sign = "p" if latitude_deg > 0 else "m"
    value = abs(latitude_deg)
    if abs(value - round(value)) < 0.05:
        return f"{sign}{int(round(value)):02d}"
    return sign + f"{value:.1f}".replace(".", "p")


def _latitude_label(latitude_deg: float) -> str:
    if abs(latitude_deg) < 0.05:
        return "eq"
    return f"{latitude_deg:+.1f}°"


def parse_meta(path: str | Path, text: str | None = None) -> RunMeta:
    """Parse the mixed-text warden2d ``meta.txt`` by named sections."""
    path = Path(path)
    if text is None:
        text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    deck_match = re.search(r"^deck\s+(.+)$", text, re.MULTILINE)
    deck = deck_match.group(1).strip() if deck_match else "unknown"
    l0 = _require_float(r"^L0\s+([-+\d.eE]+)", text, "L0")
    b0eq = _require_float(r"\bB0eq\s+([-+\d.eE]+)", text, "B0eq")
    cadence = re.search(
        r"^probe_every\s+(\d+)\s+dt\s+([-+\d.eE]+)", text, re.MULTILINE
    )
    if not cadence:
        raise ValueError("missing probe_every/dt in meta file")
    probe_every, dt = int(cadence.group(1)), float(cadence.group(2))

    field_match = re.search(r"^line\s+ns\s+\d+\s+fields\s+(.+)$", text, re.MULTILINE)
    if not field_match:
        raise ValueError("missing line field layout in meta file")
    fields = tuple(field.strip() for field in field_match.group(1).split(","))
    if fields != CHANNELS:
        raise ValueError(f"unsupported field layout {fields}; expected {CHANNELS}")

    primary: list[Probe] = []
    secondary: list[Probe] = []
    section: str | None = None
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    primary_row = re.compile(rf"^\s*({number})\s+({number})\s*$")
    secondary_row = re.compile(rf"^\s*L2\s+({number})\s+({number})\s*$")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("probes (lam_deg"):
            section = "primary"
            continue
        if stripped.startswith("probes2 (lam_deg"):
            section = "secondary"
            continue
        if section == "primary":
            match = primary_row.match(line)
            if match:
                primary.append(Probe(float(match.group(1)), float(match.group(2))))
                continue
            if primary:
                section = None
        elif section == "secondary":
            match = secondary_row.match(line)
            if match:
                secondary.append(Probe(float(match.group(1)), float(match.group(2))))
                continue
            if secondary:
                section = None

    if not primary:
        raise ValueError("meta file has no primary probes")
    if dt <= 0.0 or probe_every <= 0 or b0eq <= 0.0:
        raise ValueError("dt, probe_every, and B0eq must all be positive")
    if any(probe.wce_local <= 0.0 for probe in primary + secondary):
        raise ValueError("all local cyclotron frequencies must be positive")
    result_lines = [ProbeLine(_line_tag(l0), l0, "probes.bin", tuple(primary))]
    if secondary:
        l2 = _require_float(r"^probe2_L\s+([-+\d.eE]+)", text, "probe2_L")
        result_lines.append(
            ProbeLine(_line_tag(l2), l2, "probes2.bin", tuple(secondary))
        )
    return RunMeta(path, deck, dt, probe_every, b0eq, l0, fields, tuple(result_lines))


def load_probe_records(
    path: str | Path, nprobe: int, nchannel: int = len(CHANNELS)
) -> ProbeRecords:
    """Freeze and load only complete records from a possibly live file."""
    path = Path(path)
    stride = nchannel * nprobe
    if stride <= 0:
        raise ValueError("record stride must be positive")
    with path.open("rb") as stream:
        status = os.fstat(stream.fileno())
        frozen_bytes = status.st_size
        complete_float_bytes = frozen_bytes - frozen_bytes % 4
        nfloats = complete_float_bytes // 4
        nrecords, trailing_floats = divmod(nfloats, stride)
        values = np.fromfile(stream, dtype="<f4", count=nrecords * stride)
    if values.size != nrecords * stride:
        raise OSError(f"{path} changed while its frozen prefix was being read")
    if nrecords == 0:
        raise ValueError(f"{path} has no complete probe records")
    values = values.reshape(nrecords, nchannel, nprobe)
    bad = int(values.size - np.count_nonzero(np.isfinite(values)))
    if bad:
        raise ValueError(f"{path} contains {bad} non-finite values")
    return ProbeRecords(
        path=path,
        values=values,
        trailing_floats=int(trailing_floats),
        trailing_bytes=int(frozen_bytes % 4),
        frozen_bytes=int(frozen_bytes),
        frozen_mtime_ns=int(status.st_mtime_ns),
        prefix_sha256=hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    )


def helicity(b1: np.ndarray, by: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed positive-frequency R and L helicity signals."""
    return b1 + 1j * by, b1 - 1j * by


def component_label(component: str) -> str:
    if component == "bpar":
        return r"field-aligned wave component $\delta B_\parallel$"
    if component == "b1":
        return r"in-plane transverse wave component $\delta B_1$"
    if component == "by":
        return r"out-of-plane transverse wave component $\delta B_y$"
    if component == "r":
        return r"R helicity $\delta B_1+i\delta B_y$"
    raise ValueError(f"unknown component {component!r}")


def component_expression(component: str) -> str:
    return {
        "bpar": "Bpar (writer channel 3)",
        "b1": "B1 (writer channel 4)",
        "by": "By (writer channel 5; out of x-z plane)",
        "r": "B1 + 1j*By (writer channels 4,5)",
    }[component]


def probe_component_signals(
    values: np.ndarray, probe_index: int, component: str
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Return primary signal, optional opposite helicity, and real-PSD flag."""
    if component == "r":
        b1 = values[:, B1_INDEX, probe_index].astype(np.float64)
        by = values[:, BY_INDEX, probe_index].astype(np.float64)
        right, left = helicity(b1, by)
        return right, left, False
    channel = {"bpar": BPAR_INDEX, "b1": B1_INDEX, "by": BY_INDEX}.get(component)
    if channel is None:
        raise ValueError(f"unknown component {component!r}")
    signal = values[:, channel, probe_index].astype(np.float64)
    return signal, None, True


def color_normalization_field(
    meta: RunMeta, probe: Probe, normalization: str
) -> float:
    if normalization == "reference":
        return meta.b0eq
    if normalization == "local":
        return probe.wce_local
    raise ValueError(f"unknown color normalization {normalization!r}")


def color_normalization_label(normalization: str) -> str:
    if normalization == "reference":
        return r"B_{0,ref}"
    if normalization == "local":
        return r"B_{0,local}"
    raise ValueError(f"unknown color normalization {normalization!r}")


def periodic_hann(n: int) -> np.ndarray:
    index = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * index / n)


def positive_frequency_q(n: int, sample_dt: float, reference_wce: float) -> np.ndarray:
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=sample_dt)
    return omega[omega > 0.0] / reference_wce


def detrend_window(z: np.ndarray, window: np.ndarray, mode: str) -> np.ndarray:
    """Remove a weighted constant or line from one complex-valued window."""
    if mode == "none":
        return z
    weights = window * window
    total = float(weights.sum())
    mean = np.sum(weights * z) / total
    result = z - mean
    if mode == "constant":
        return result
    if mode != "linear":
        raise ValueError(f"unknown detrend mode {mode!r}")
    coordinate = np.arange(z.size, dtype=np.float64) - 0.5 * (z.size - 1)
    slope = np.sum(weights * coordinate * result) / np.sum(weights * coordinate**2)
    return result - slope * coordinate


def stft_psd(
    signal: np.ndarray,
    sample_dt: float,
    nperseg: int,
    hop: int,
    detrend: str = "linear",
    real_one_sided: bool = False,
) -> Spectrum:
    """Absolute positive-frequency PSD of a complex signal.

    The returned density is per angular-frequency unit:

        |FFT(w * detrend(z))|^2 / (2*pi*fs*sum(w^2)).

    Complex-helicity input is not doubled because its negative-frequency side
    carries the opposite rotation.  Set ``real_one_sided=True`` for a real
    component such as B_parallel; its positive-frequency PSD is then doubled
    so the integral recovers the real-signal variance (DC/Nyquist are omitted).
    """
    original_signal = np.asarray(signal)
    if real_one_sided and np.any(np.imag(original_signal) != 0.0):
        raise ValueError("real_one_sided=True requires a real-valued signal")
    signal = np.asarray(original_signal, dtype=np.complex128)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if nperseg < 16 or hop < 1:
        raise ValueError("nperseg must be >=16 and hop must be positive")
    if signal.size < nperseg:
        raise ValueError(
            f"only {signal.size} samples, fewer than requested N={nperseg}"
        )
    starts = np.arange(0, signal.size - nperseg + 1, hop, dtype=np.int64)
    window = periodic_hann(nperseg)
    window_energy = float(np.sum(window * window))
    fs = 1.0 / sample_dt
    omega_all = 2.0 * np.pi * np.fft.fftfreq(nperseg, d=sample_dt)
    positive = omega_all > 0.0
    omega = omega_all[positive]
    psd = np.empty((omega.size, starts.size), dtype=np.float64)
    scale = 2.0 * np.pi * fs * window_energy
    for column, start in enumerate(starts):
        segment = detrend_window(signal[start : start + nperseg], window, detrend)
        transform = np.fft.fft(window * segment)
        factor = 2.0 if real_one_sided else 1.0
        psd[:, column] = factor * np.abs(transform[positive]) ** 2 / scale
    # Sample zero is written after the first probe cadence, at t=sample_dt.
    centers = (starts + 1.0 + 0.5 * (nperseg - 1)) * sample_dt
    return Spectrum(omega, centers, psd, nperseg, hop, sample_dt)


def _db_density(psd: np.ndarray, b0eq: float) -> np.ndarray:
    # omega_pe=1 in the normalized code.  Units are S_omega*omega_pe/B0eq^2.
    return 10.0 * np.log10(np.maximum(psd / (b0eq * b0eq), 1.0e-300))


def _band_mean(q: np.ndarray, density: np.ndarray, bounds: tuple[float, float]) -> float:
    integral = _band_integral(q, density, bounds)
    return integral / (bounds[1] - bounds[0])


def _band_integral(
    q: np.ndarray, density: np.ndarray, bounds: tuple[float, float]
) -> float:
    """Trapezoid integral with values interpolated at exact band edges."""
    low, high = bounds
    if q.ndim != 1 or density.ndim != 1 or q.size != density.size:
        raise ValueError("band integration requires equal one-dimensional arrays")
    if q.size < 2 or low < q[0] or high > q[-1] or low >= high:
        return math.nan
    interior = (q > low) & (q < high)
    band_q = np.concatenate(([low], q[interior], [high]))
    band_density = np.concatenate(
        ([np.interp(low, q, density)], density[interior], [np.interp(high, q, density)])
    )
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy < 2.0
        trapezoid = np.trapz
    return float(trapezoid(band_density, band_q))


def _safe_ratio_db(numerator: float, denominator: float) -> float:
    ratio = _safe_linear_ratio(numerator, denominator)
    return 10.0 * math.log10(ratio) if ratio > 0.0 else math.nan


def _safe_linear_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if numerator < 0.0 or denominator <= 0.0:
        return math.nan
    return numerator / denominator


def _read_last_energy_time(path: Path) -> float | None:
    if not path.exists():
        return None
    last: float | None = None
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    last = float(row["t"])
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return None
    return last


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _plot_spectrum(
    ax: plt.Axes,
    spectrum: Spectrum,
    reference_wce: float,
    wce_local: float,
    amplitude_field: float,
    fmin: float,
    fmax: float,
    db_min: float,
    db_max: float,
    cmap: str,
):
    q = spectrum.omega / reference_wce
    selected = (q >= fmin) & (q <= fmax)
    time_eq = spectrum.time * reference_wce
    image = ax.pcolormesh(
        time_eq,
        q[selected],
        _db_density(spectrum.psd[selected], amplitude_field),
        shading="auto",
        cmap=cmap,
        vmin=db_min,
        vmax=db_max,
        rasterized=True,
    )
    ax.axhline(0.5, color="#36d9ff", linestyle=":", linewidth=0.8)
    ax.axhline(
        0.5 * wce_local / reference_wce,
        color="white",
        linestyle="--",
        linewidth=0.8,
        alpha=0.9,
    )
    ax.set_ylim(fmin, fmax)
    return image


def _resolution_title(name: str, spectrum: Spectrum, b0eq: float) -> str:
    return (
        f"{name}: N={spectrum.nperseg}, hop={spectrum.hop}, "
        f"TΩref={spectrum.nperseg * spectrum.sample_dt * b0eq:.1f}, "
        f"Δω/Ωref={spectrum.rayleigh_omega / b0eq:.4f}"
    )


def plot_line_stack(
    destination: Path,
    meta: RunMeta,
    line: ProbeLine,
    results: list[dict[str, Spectrum]],
    args: argparse.Namespace,
) -> None:
    nprobe = len(line.probes)
    fig, axes = plt.subplots(
        nprobe,
        2,
        figsize=(14.5, 1.75 * nprobe + 2.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    image = None
    for row, (probe, spectra) in enumerate(zip(line.probes, results)):
        amplitude_field = color_normalization_field(
            meta, probe, args.amplitude_normalization
        )
        for column, key in enumerate(("short_main", "long_main")):
            ax = axes[row, column]
            image = _plot_spectrum(
                ax,
                spectra[key],
                meta.b0eq,
                probe.wce_local,
                amplitude_field,
                args.fmin,
                args.fmax,
                args.db_min,
                args.db_max,
                args.cmap,
            )
            if row == 0:
                ax.set_title(
                    _resolution_title("SHORT" if column == 0 else "LONG", spectra[key], meta.b0eq),
                    fontsize=9,
                )
            ax.text(
                0.008,
                0.92,
                f"{_latitude_label(probe.latitude_deg)}, Ωloc={probe.wce_local:.5f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"},
            )
        axes[row, 0].set_ylabel(r"$\omega/\Omega_{e,ref}$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$t\,\Omega_{e,ref}$")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, pad=0.012, aspect=45)
    norm_label = color_normalization_label(args.amplitude_normalization)
    colorbar.set_label(
        rf"$10\log_{{10}}[S_\omega\,\omega_{{pe}}/{norm_label}^2]$ [dB]"
    )
    fig.suptitle(
        f"{Path(meta.deck).name} — {line.tag}, {component_label(args.component)}, "
        f"{args.detrend} detrend, shared absolute scale\n"
        f"amplitude norm={args.amplitude_normalization}; Ωref=Ωe,eq(L0); "
        "cyan dotted: 0.5 Ωref; white dashed: 0.5 Ωlocal",
        fontsize=11,
    )
    fig.savefig(destination, dpi=args.dpi)
    plt.close(fig)


def plot_individual_probe(
    destination: Path,
    meta: RunMeta,
    line: ProbeLine,
    probe: Probe,
    spectra: dict[str, Spectrum],
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True, sharey=True,
                             constrained_layout=True)
    image = None
    amplitude_field = color_normalization_field(
        meta, probe, args.amplitude_normalization
    )
    for ax, key, name in zip(axes, ("short_main", "long_main"), ("SHORT", "LONG")):
        image = _plot_spectrum(
            ax,
            spectra[key],
            meta.b0eq,
            probe.wce_local,
            amplitude_field,
            args.fmin,
            args.fmax,
            args.db_min,
            args.db_max,
            args.cmap,
        )
        ax.set_title(_resolution_title(name, spectra[key], meta.b0eq), fontsize=9)
        ax.set_ylabel(r"$\omega/\Omega_{e,ref}$")
        factor = meta.b0eq / probe.wce_local
        secondary = ax.secondary_yaxis(
            "right", functions=(lambda q, f=factor: q * f, lambda q, f=factor: q / f)
        )
        secondary.set_ylabel(r"$\omega/\Omega_{e,local}$")
    axes[-1].set_xlabel(r"$t\,\Omega_{e,ref}$")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, pad=0.08, aspect=35)
    norm_label = color_normalization_label(args.amplitude_normalization)
    colorbar.set_label(
        rf"$10\log_{{10}}[S_\omega\,\omega_{{pe}}/{norm_label}^2]$ [dB]"
    )
    fig.suptitle(
        f"{Path(meta.deck).name} — {line.tag}, {_latitude_label(probe.latitude_deg)}, "
        f"Ωlocal={probe.wce_local:.5f}\n"
        f"{component_label(args.component)}; {args.detrend} detrend; "
        f"{args.amplitude_normalization} amplitude norm; "
        f"fixed [{args.db_min:g},{args.db_max:g}] dB\n"
        "cyan dotted: 0.5 Ωref; white dashed: 0.5 Ωlocal",
        fontsize=11,
    )
    fig.savefig(destination, dpi=args.dpi)
    plt.close(fig)


def plot_line_psd(
    destination: Path,
    meta: RunMeta,
    line: ProbeLine,
    results: list[dict[str, Spectrum]],
    args: argparse.Namespace,
) -> None:
    has_aux = args.component == "r"
    if has_aux:
        fig, (power_ax, aux_ax) = plt.subplots(
            2, 1, figsize=(11.5, 8.0), sharex=True, constrained_layout=True
        )
    else:
        fig, power_ax = plt.subplots(1, 1, figsize=(11.5, 5.2), constrained_layout=True)
        aux_ax = None
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(line.probes)))
    for color, probe, spectra in zip(colors, line.probes, results):
        q = spectra["long_main"].omega / meta.b0eq
        selected = (q >= args.fmin) & (q <= args.fmax)
        main_mean = spectra["long_main"].psd.mean(axis=1)
        amplitude_field = color_normalization_field(
            meta, probe, args.amplitude_normalization
        )
        label = _latitude_label(probe.latitude_deg)
        power_ax.plot(
            q[selected],
            _db_density(main_mean[selected], amplitude_field),
            color=color,
            linewidth=1.25,
            label=label,
        )
        if has_aux:
            assert aux_ax is not None
            aux_mean = spectra["long_aux"].psd.mean(axis=1)
            ratio_db = 10.0 * np.log10(
                np.maximum(main_mean, 1e-300) / np.maximum(aux_mean, 1e-300)
            )
            total_db = _db_density(main_mean + aux_mean, amplitude_field)
            ratio_db[total_db < args.db_min + 12.0] = np.nan
            aux_ax.plot(
                q[selected], ratio_db[selected], color=color, linewidth=1.15, label=label
            )
    axes = (power_ax, aux_ax) if aux_ax is not None else (power_ax,)
    for ax in axes:
        for bound, style in ((0.20, ":"), (0.45, ":"), (0.55, "--"), (0.75, ":")):
            ax.axvline(bound, color="0.45", linestyle=style, linewidth=0.8)
        ax.grid(alpha=0.22)
        ax.set_xlim(args.fmin, args.fmax)
    power_ax.set_ylim(args.db_min, args.db_max)
    norm_label = color_normalization_label(args.amplitude_normalization)
    power_ax.set_ylabel(
        rf"$10\log_{{10}}[\langle S_\omega\rangle\,\omega_{{pe}}/{norm_label}^2]$ [dB]"
    )
    power_ax.legend(ncol=4, fontsize=8, loc="best")
    power_ax.set_title(
        f"{line.tag} {component_label(args.component)}: time-averaged LONG PSD\n"
        "vertical lines: LB/gap/UB edges on the common Ωref axis"
    )
    if aux_ax is not None:
        aux_ax.axhline(10.0, color="black", linestyle="--", linewidth=0.8)
        aux_ax.set_ylabel(
            f"R/L positive-frequency power [dB]\n"
            f"(masked below {args.db_min + 12:g} dB total)"
        )
        aux_ax.set_xlabel(r"common frequency $\omega/\Omega_{e,ref}$")
    else:
        power_ax.set_xlabel(r"common frequency $\omega/\Omega_{e,ref}$")
    fig.suptitle(
        f"{Path(meta.deck).name} — {line.tag}; "
        f"amplitude norm={args.amplitude_normalization}"
    )
    fig.savefig(destination, dpi=args.dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path, help="warden2d output directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="product directory (default: a component-specific directory)")
    parser.add_argument(
        "--component",
        choices=("r", "by", "b1", "bpar"),
        default="r",
        help=("magnetic signal: R helicity, out-of-plane By, in-plane transverse "
              "B1, or field-aligned Bpar (default: r)"),
    )
    parser.add_argument(
        "--amplitude-normalization",
        choices=("reference", "local"),
        default="reference",
        help=("color/PSD field normalization only; frequency and time axes always "
              "use Omega_e,ref (default: reference)"),
    )
    parser.add_argument(
        "--cmap",
        default="cividis",
        help="sequential Matplotlib color map (default: cividis)",
    )
    parser.add_argument("--short-n", type=int, default=1024)
    parser.add_argument("--short-hop", type=int, default=128)
    parser.add_argument("--long-n", type=int, default=4096)
    parser.add_argument("--long-hop", type=int, default=512)
    parser.add_argument("--detrend", choices=("none", "constant", "linear"),
                        default="linear")
    parser.add_argument("--fmin", type=float, default=0.05,
                        help="minimum common omega/Omega_e,eq")
    parser.add_argument("--fmax", type=float, default=0.95,
                        help="maximum common omega/Omega_e,eq")
    parser.add_argument("--db-min", type=float, default=-90.0)
    parser.add_argument("--db-max", type=float, default=-30.0)
    parser.add_argument("--dpi", type=int, default=145)
    parser.add_argument("--no-individual", action="store_true",
                        help="skip the per-probe two-panel PNG files")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("short_n", "short_hop", "long_n", "long_hop"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.short_n < 16 or args.long_n < 16:
        raise ValueError("STFT window lengths must be at least 16")
    if not 0.0 <= args.fmin < args.fmax:
        raise ValueError("require 0 <= --fmin < --fmax")
    if args.db_min >= args.db_max:
        raise ValueError("require --db-min < --db-max")
    if args.cmap not in matplotlib.colormaps:
        raise ValueError(f"unknown Matplotlib color map {args.cmap!r}")


def _default_destination(outdir: Path, args: argparse.Namespace) -> Path:
    if args.component == "r" and args.amplitude_normalization == "reference":
        return outdir / "probe_spectra"
    suffix = f"_{args.component}"
    if args.amplitude_normalization == "local":
        suffix += "_local"
    return outdir / f"probe_spectra{suffix}"


def _guard_generation_identity(
    manifest_path: Path, component: str, amplitude_normalization: str
) -> None:
    """Refuse to overwrite a product directory owned by another signal mode."""
    if not manifest_path.exists():
        return
    try:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError(
            f"existing {manifest_path} is unreadable; choose another --output directory"
        )
    signal = old.get("signal", {})
    old_component = signal.get("component", old.get("component"))
    if old_component is None and "helicity" in old:
        old_component = "r"
    analysis = old.get("analysis", {})
    old_normalization = analysis.get(
        "amplitude_normalization", old.get("amplitude_normalization", "reference")
    )
    if old_component is not None and (
        old_component != component or old_normalization != amplitude_normalization
    ):
        raise ValueError(
            f"{manifest_path.parent} already contains {old_component}/{old_normalization} "
            f"products; refusing to overwrite them with {component}/{amplitude_normalization}"
        )


def run(args: argparse.Namespace) -> list[Path]:
    _validate_args(args)
    outdir = args.outdir.resolve()
    destination = (args.output or _default_destination(outdir, args)).resolve()
    meta_path = outdir / "meta.txt"
    meta_bytes = meta_path.read_bytes()
    script_path = Path(__file__).resolve()
    script_bytes = script_path.read_bytes()
    meta = parse_meta(meta_path, meta_bytes.decode("utf-8"))

    grids = {
        "SHORT": positive_frequency_q(args.short_n, meta.sample_dt, meta.b0eq),
        "LONG": positive_frequency_q(args.long_n, meta.sample_dt, meta.b0eq),
    }
    long_q_min, long_q_max = grids["LONG"][0], grids["LONG"][-1]
    if args.fmin < long_q_min or args.fmax > long_q_max:
        raise ValueError(
            f"requested frequency range [{args.fmin:g},{args.fmax:g}] exceeds "
            f"the LONG positive-frequency support [{long_q_min:g},{long_q_max:g}]"
        )
    for name, grid in grids.items():
        count = int(np.count_nonzero((grid >= args.fmin) & (grid <= args.fmax)))
        if count < 2:
            raise ValueError(
                f"requested frequency range [{args.fmin:g},{args.fmax:g}] contains "
                f"only {count} {name} FFT bins; at least 2 are required"
            )

    manifest_path = destination / "probe_manifest.json"
    _guard_generation_identity(
        manifest_path, args.component, args.amplitude_normalization
    )
    destination.mkdir(parents=True, exist_ok=True)
    in_progress = {
        "format_version": 2,
        "analysis_complete": False,
        "simulation_complete": None,
        "status": "analysis generation in progress or interrupted",
        "analysis_script": str(script_path),
        "analysis_script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "source_directory": str(outdir),
        "component": args.component,
        "component_expression": component_expression(args.component),
        "amplitude_normalization": args.amplitude_normalization,
    }
    _atomic_write_text(manifest_path, json.dumps(in_progress, indent=2) + "\n")
    energy_end = _read_last_energy_time(outdir / "energy.csv")

    results_by_tag: dict[str, list[dict[str, Spectrum]]] = {}
    products: list[Path] = []
    manifest_lines: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for line in meta.lines:
        records = load_probe_records(outdir / line.filename, len(line.probes))
        assumed_end = records.nrecords * meta.sample_dt
        if records.trailing_floats or records.trailing_bytes:
            warnings.warn(
                f"{line.filename}: ignored {records.trailing_floats} trailing floats "
                f"and {records.trailing_bytes} trailing bytes"
            )
        if energy_end is not None and abs(energy_end - assumed_end) > meta.sample_dt * 1.01:
            warnings.warn(
                f"{line.filename}: complete probes end at t={assumed_end:g}, "
                f"energy.csv ends at t={energy_end:g}. This can be ordinary diagnostic "
                "cadence/stdio buffering after an interrupted run; because probe records "
                "contain no step IDs, it cannot be distinguished from a resume gap/duplicate"
            )

        line_results: list[dict[str, Spectrum]] = []
        for index, probe in enumerate(line.probes):
            main_signal, auxiliary_signal, real_one_sided = probe_component_signals(
                records.values, index, args.component
            )
            spectra = {
                "short_main": stft_psd(
                    main_signal,
                    meta.sample_dt,
                    args.short_n,
                    args.short_hop,
                    args.detrend,
                    real_one_sided=real_one_sided,
                ),
                "long_main": stft_psd(
                    main_signal,
                    meta.sample_dt,
                    args.long_n,
                    args.long_hop,
                    args.detrend,
                    real_one_sided=real_one_sided,
                ),
            }
            if auxiliary_signal is not None:
                spectra["short_aux"] = stft_psd(
                    auxiliary_signal,
                    meta.sample_dt,
                    args.short_n,
                    args.short_hop,
                    args.detrend,
                )
                spectra["long_aux"] = stft_psd(
                    auxiliary_signal,
                    meta.sample_dt,
                    args.long_n,
                    args.long_hop,
                    args.detrend,
                )
            line_results.append(spectra)

            q = spectra["long_main"].omega / meta.b0eq
            main_mean = spectra["long_main"].psd.mean(axis=1)
            science = (q >= args.fmin) & (q <= args.fmax)
            science_density = main_mean[science]
            peak_q = (
                float(q[science][np.argmax(science_density)])
                if science_density.size and np.max(science_density) > 0.0
                else math.nan
            )
            q_local = spectra["long_main"].omega / probe.wce_local
            means_eq = {
                name: _band_mean(q, main_mean, bounds)
                for name, bounds in DEFAULT_BANDS.items()
            }
            means_local = {
                name: _band_mean(q_local, main_mean, bounds)
                for name, bounds in DEFAULT_BANDS.items()
            }
            summary_row: dict[str, object] = {
                "component": args.component,
                "component_expression": component_expression(args.component),
                "psd_sidedness": (
                    "real positive-frequency one-sided x2"
                    if real_one_sided
                    else "complex positive-frequency x1"
                ),
                "line": line.tag,
                "L": f"{line.L:.6g}",
                "latitude_deg": f"{probe.latitude_deg:.6g}",
                "wce_local": f"{probe.wce_local:.8g}",
                "color_normalization": args.amplitude_normalization,
                "color_normalization_B0": f"{color_normalization_field(meta, probe, args.amplitude_normalization):.8g}",
                "nrecords": records.nrecords,
                "trailing_floats": records.trailing_floats,
                "peak_omega_over_wce_ref": f"{peak_q:.8g}",
                "peak_omega_over_wce_local": f"{peak_q * meta.b0eq / probe.wce_local:.8g}",
                "ref_LB_mean_psd": f"{means_eq['LB']:.9e}",
                "ref_gap_mean_psd": f"{means_eq['gap']:.9e}",
                "ref_UB_mean_psd": f"{means_eq['UB']:.9e}",
                "ref_gap_over_LB_mean": f"{_safe_linear_ratio(means_eq['gap'], means_eq['LB']):.9g}",
                "ref_UB_over_LB_mean": f"{_safe_linear_ratio(means_eq['UB'], means_eq['LB']):.9g}",
                "local_LB_mean_psd": f"{means_local['LB']:.9e}",
                "local_gap_mean_psd": f"{means_local['gap']:.9e}",
                "local_UB_mean_psd": f"{means_local['UB']:.9e}",
                "local_gap_over_LB_mean": f"{_safe_linear_ratio(means_local['gap'], means_local['LB']):.9g}",
                "local_UB_over_LB_mean": f"{_safe_linear_ratio(means_local['UB'], means_local['LB']):.9g}",
            }
            if auxiliary_signal is not None:
                auxiliary_mean = spectra["long_aux"].psd.mean(axis=1)
                main_total = _band_integral(q, main_mean, (args.fmin, args.fmax))
                auxiliary_total = _band_integral(
                    q, auxiliary_mean, (args.fmin, args.fmax)
                )
                summary_row["R_over_L_db"] = (
                    f"{_safe_ratio_db(main_total, auxiliary_total):.8g}"
                )
            summary_rows.append(summary_row)
        results_by_tag[line.tag] = line_results
        manifest_lines.append({
            "tag": line.tag,
            "L": line.L,
            "source": line.filename,
            "frozen_file_bytes": records.frozen_bytes,
            "frozen_file_mtime_ns": records.frozen_mtime_ns,
            "complete_prefix_sha256": records.prefix_sha256,
            "nrecords": records.nrecords,
            "trailing_floats_ignored": records.trailing_floats,
            "trailing_bytes_ignored": records.trailing_bytes,
            "assumed_first_time_wpe": meta.sample_dt,
            "assumed_last_time_wpe": assumed_end,
            "probes": [
                {
                    "index": index,
                    "latitude_deg": probe.latitude_deg,
                    "wce_local": probe.wce_local,
                    "label": f"{line.tag}_lat_{_latitude_tag(probe.latitude_deg)}",
                }
                for index, probe in enumerate(line.probes)
            ],
        })

    for line in meta.lines:
        results = results_by_tag[line.tag]
        stack_path = destination / f"probe_stft_{line.tag}.png"
        plot_line_stack(stack_path, meta, line, results, args)
        products.append(stack_path)
        psd_path = destination / f"probe_psd_{line.tag}.png"
        plot_line_psd(psd_path, meta, line, results, args)
        products.append(psd_path)
        if not args.no_individual:
            for probe, spectra in zip(line.probes, results):
                path = destination / (
                    f"probe_stft_{line.tag}_lat_{_latitude_tag(probe.latitude_deg)}.png"
                )
                plot_individual_probe(path, meta, line, probe, spectra, args)
                products.append(path)

    summary_path = destination / "probe_spectrum_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    products.append(summary_path)

    expected_products = {path.name for path in products} | {"probe_manifest.json"}
    stale_products = sorted(
        path.name
        for path in destination.glob("probe_*")
        if path.is_file() and path.name not in expected_products
    )
    if stale_products:
        warnings.warn(
            "output directory contains stale files excluded from this generation's "
            f"manifest: {', '.join(stale_products)}"
        )
    real_component = args.component != "r"
    source_channels: int | list[int]
    source_channels = (
        [B1_INDEX, BY_INDEX]
        if args.component == "r"
        else [{"bpar": BPAR_INDEX, "b1": B1_INDEX, "by": BY_INDEX}[args.component]]
    )
    manifest = {
        "format_version": 2,
        "analysis_complete": True,
        "simulation_complete": None,
        "simulation_status_note": "not inferable from warden2d meta/probe files",
        "analysis_script": str(script_path),
        "analysis_script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "source_directory": str(outdir),
        "meta_file": str(meta.path),
        "meta_sha256": hashlib.sha256(meta_bytes).hexdigest(),
        "deck": meta.deck,
        "channel_layout": list(meta.fields),
        "signal": {
            "component": args.component,
            "label": component_label(args.component),
            "expression": component_expression(args.component),
            "source_channel_indices": source_channels,
            "real_input": real_component,
            "fft_side": "positive frequencies only",
            "positive_frequency_psd_factor": 2 if real_component else 1,
            "dc_and_nyquist": "excluded",
            "auxiliary_signal": (
                "B1 - 1j*By (L helicity)" if args.component == "r" else None
            ),
        },
        "dt": meta.dt,
        "probe_every": meta.probe_every,
        "sample_dt_wpe": meta.sample_dt,
        "B0eq_reference": meta.b0eq,
        "reference_L": meta.l0,
        "energy_csv_last_time_wpe": energy_end,
        "analysis": {
            "detrend": args.detrend,
            "window": "periodic Hann",
            "short": {"N": args.short_n, "hop": args.short_hop,
                      "rayleigh_omega_over_wce_ref": 2 * math.pi /
                      (args.short_n * meta.sample_dt * meta.b0eq)},
            "long": {"N": args.long_n, "hop": args.long_hop,
                     "rayleigh_omega_over_wce_ref": 2 * math.pi /
                     (args.long_n * meta.sample_dt * meta.b0eq)},
            "frequency_axis": "omega / Omega_e,ref; Omega_e,ref = B0eq at reference_L",
            "time_axis": "time * Omega_e,ref",
            "amplitude_normalization": args.amplitude_normalization,
            "color_normalization": (
                "10log10(S_omega*omega_pe/B0eq_reference^2)"
                if args.amplitude_normalization == "reference"
                else "10log10(S_omega*omega_pe/B0_local(probe)^2)"
            ),
            "cmap": args.cmap,
            "db_range": [args.db_min, args.db_max],
            "frequency_range": [args.fmin, args.fmax],
            "normalization_prohibited": ["row whitening", "column normalization", "top-k peaks"],
            "bands": {
                "bounds": DEFAULT_BANDS,
                "reported_normalizations": [
                    "omega/B0eq_reference",
                    "omega/Omega_e_local",
                ],
                "statistic": "bandwidth-normalized mean PSD",
            },
        },
        "lines": manifest_lines,
        "products": sorted(expected_products),
        "stale_files_not_in_products": stale_products,
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    products.append(manifest_path)

    print(f"rendered {len(products) - 2} figures in {destination}")
    for line in manifest_lines:
        print(
            f"  {line['tag']}: {line['nrecords']} complete records, "
            f"{line['trailing_floats_ignored']} trailing floats ignored"
        )
    print(f"  manifest: {manifest_path}")
    print(f"  summary:  {summary_path}")
    return products


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
