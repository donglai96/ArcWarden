# Learning Note — Step 1: The CMake Build System (line by line)

> Goal of this note: understand **what CMake is**, **what each line of our
> `CMakeLists.txt` does**, and **why it's in that order**. By the end you should
> be able to read and edit the build file yourself.

---

## 0. What is CMake, in one paragraph

CMake is a **build-system generator**. You don't compile with CMake directly.
You write a `CMakeLists.txt` describing *what* you want built (executables,
libraries, which files, which compiler, which flags). Then:

```text
CMakeLists.txt  --(cmake configure)-->  build/ (Makefiles or Ninja files)
build/          --(cmake --build)----->  compiler runs --> ./arcwarden
```

Two distinct phases:

1. **Configure** (`cmake -B build`): CMake reads `CMakeLists.txt`, detects your
   compilers/GPU/libraries, and writes a `build/` directory full of generated
   build scripts plus a `CMakeCache.txt`.
2. **Build** (`cmake --build build`): the generated scripts invoke the actual
   compiler (`nvcc`, `g++`) to produce binaries.

Key mental model: **`CMakeLists.txt` is a script that runs top-to-bottom during
configure.** Order matters. Lines above can affect lines below.

---

## 1. The two commands you actually run

```bash
cmake -B build          # configure: read CMakeLists.txt, generate build/
cmake --build build     # build: compile the sources into ./build/arcwarden
ctest --test-dir build  # run the registered tests
```

- `-B build` = "put the generated files in a directory called `build/`". This is
  an **out-of-source build**: generated junk stays out of your source tree, and
  `rm -rf build` is a clean reset (we did exactly this to clear the bad cache).
- `build/CMakeCache.txt` remembers every decision from configure. **This is why
  a stale value can persist** — if you change a default in `CMakeLists.txt` but
  the value is already in the cache, the cache wins. Deleting `build/` forces a
  fresh decision. (This is the trap behind the `sm_75` bug — more in §7.)

---

## 2. Our `CMakeLists.txt`, section by section

### Line 1 — minimum version

```cmake
cmake_minimum_required(VERSION 3.24)
```

Declares the oldest CMake we support. It also turns on "policy" behavior
matching 3.24 (CMake's backward-compat rules). We need ≥3.24 because modern
CUDA + `CMAKE_CUDA_ARCHITECTURES native` support is recent. If someone runs an
older CMake, they get a clear error instead of mysterious failures.

---

### Lines 6–13 — pick the right `nvcc` *before* the project starts

```cmake
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  foreach(_cuda_root /usr/local/cuda-13.3 /usr/local/cuda-13.1 /usr/local/cuda)
    if(EXISTS ${_cuda_root}/bin/nvcc)
      set(CMAKE_CUDA_COMPILER ${_cuda_root}/bin/nvcc)
      break()
    endif()
  endforeach()
endif()
```

Problem this solves: the **system default `nvcc` is CUDA 12.0**, too old to
generate code for the RTX 5090 (`sm_120`). We must point CMake at a newer
toolkit (13.3).

Reading it as code:

- `CMAKE_CUDA_COMPILER` is a built-in CMake variable: "which CUDA compiler to
  use". If the user already set it (e.g. `-DCMAKE_CUDA_COMPILER=...` on the
  command line), `NOT DEFINED` is false and we leave their choice alone.
- `foreach(_cuda_root ...)` loops over candidate install dirs **in priority
  order**: try 13.3 first, then 13.1, then a generic `/usr/local/cuda` symlink.
- `if(EXISTS .../bin/nvcc)` checks the file is really there.
- `set(CMAKE_CUDA_COMPILER ...)` records the choice; `break()` stops at the
  first hit. `${_cuda_root}` is variable expansion (like `$var` in bash).

> ⚠️ This *must* come **before** `project(... CUDA)`. `project()` is where CMake
> actually goes looking for and locks in the CUDA compiler. Set it after, and
> it's too late.

---

### Lines 22–24 — choose the GPU architecture *before* the project starts

```cmake
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
```

`CMAKE_CUDA_ARCHITECTURES` controls *which GPU instruction set* `nvcc` emits.
`native` means "detect the GPU in this machine and target exactly that" — here
it resolves to `120` (sm_120, the RTX 5090).

Why the `if(NOT DEFINED)` guard? So a user can override:
`cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120`. If they did, we don't clobber
it.

> ⚠️ The ordering here was the actual bug in Step 2. Explained fully in §7.

---

### Line 26 — declare the project and enable languages

```cmake
project(ArcWarden LANGUAGES CXX CUDA)
```

This is the heart of configure. `project()`:

- Names the project `ArcWarden`.
- `LANGUAGES CXX CUDA` tells CMake to **enable the C++ and CUDA compilers** —
  i.e. find them, test that they work, and detect their capabilities.
- This is precisely *why* the two blocks above run **before** it: by the time
  `project()` enables CUDA, it reads `CMAKE_CUDA_COMPILER` and
  `CMAKE_CUDA_ARCHITECTURES`. After this line, CUDA is "locked in".

Think of everything above line 26 as "set the table"; `project()` is "serve the
meal".

---

### Lines 29–32 — language standards

```cmake
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
```

- `CMAKE_CXX_STANDARD 20` → compile host C++ as C++20 (we use `std::exchange`,
  concepts later, etc.).
- `CMAKE_CUDA_STANDARD 20` → same for the CUDA (`.cu`) side.
- `..._REQUIRED ON` → if the compiler *can't* do C++20, **fail loudly** instead
  of silently falling back to an older standard.

These are global defaults applied to every target declared afterwards.

---

### Lines 34–36 — default build type

```cmake
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
```

`CMAKE_BUILD_TYPE` selects the optimization preset:

- `Release` → `-O3`, no debug info: fast (what we want for a PIC code).
- `Debug` → `-O0 -g`: slow but debuggable.
- `RelWithDebInfo` → optimized **and** with debug symbols.

We default to `Release` if the user didn't pick one. Override with
`cmake -B build -DCMAKE_BUILD_TYPE=Debug`. (Note: this only applies to single-
config generators like Make/Ninja, which is what we use.)

---

### Line 39 — find the CUDA libraries

```cmake
find_package(CUDAToolkit REQUIRED)
```

`find_package` locates an external dependency and makes it usable. `CUDAToolkit`
is CMake's built-in module that finds the CUDA runtime, cuFFT, cuBLAS, etc.

- `REQUIRED` → abort configure if it can't be found.
- On success it defines **imported targets** like `CUDA::cudart` and
  `CUDA::cufft`. An imported target bundles the include paths *and* the library
  to link, so you don't hand-write `-I/usr/local/cuda/include
  -L.../lib64 -lcufft`. You just "link the target" and CMake fills in the rest.

---

### Lines 42–44 — define the executable

```cmake
add_executable(arcwarden src/main.cpp)
target_include_directories(arcwarden PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(arcwarden PRIVATE CUDA::cudart CUDA::cufft)
```

- `add_executable(arcwarden src/main.cpp)` → build a program named `arcwarden`
  from `src/main.cpp`. As we add steps, more `.cu`/`.cpp` files go in this list.
- `target_include_directories(arcwarden PRIVATE .../include)` → add our
  `include/` folder to the header search path, so `#include "pic/cuda_utils.hpp"`
  resolves. `${CMAKE_SOURCE_DIR}` is the dir containing the top `CMakeLists.txt`.
- `target_link_libraries(arcwarden PRIVATE CUDA::cudart CUDA::cufft)` → link the
  CUDA runtime and cuFFT. Because these are imported targets, this *also* pulls
  in their header paths automatically.

**What is `PRIVATE`?** It controls *propagation* to other targets that depend on
this one:

- `PRIVATE` → used only to build `arcwarden`; not exposed to dependents.
- `PUBLIC` → used to build `arcwarden` **and** inherited by anything linking it.
- `INTERFACE` → not used to build this target, only passed to dependents.

For a final executable nobody links *against*, `PRIVATE` is the right default.
(It matters once we add libraries that other targets consume.)

---

### Lines 47–48 — register a test

```cmake
enable_testing()
add_test(NAME smoke_device COMMAND arcwarden)
```

- `enable_testing()` turns on CTest, CMake's test runner.
- `add_test(NAME smoke_device COMMAND arcwarden)` registers a test that just
  runs the `arcwarden` binary. A test **passes if the command exits 0**. Our
  `main` returns 0 on success, 1 if no GPU — so this is a real (if minimal)
  smoke test: "the program builds and the GPU is reachable". Run with
  `ctest --test-dir build`. Later steps add real unit tests (FFT round-trip,
  Poisson, charge conservation) the same way.

---

## 3. The mental model of "targets"

Modern CMake is **target-centric**. A *target* (`arcwarden`) is a node that
carries its own properties: sources, include dirs, link libs, compile flags.
You attach properties with `target_*` commands, and `PRIVATE/PUBLIC/INTERFACE`
decides whether a property stays local or flows to dependents.

Old CMake used global commands (`include_directories()`, `link_libraries()`)
that affected *everything*. Avoid those. `target_*` is the modern, scoped way —
it's how the same `include/` only attaches to the targets that need it.

---

## 4. Why ordering matters (the recurring theme)

`CMakeLists.txt` executes top to bottom. The non-obvious rule:

> Anything that influences **how a language is enabled** must be set **before**
> `project(... LANGUAGES ...)`.

That's why both `CMAKE_CUDA_COMPILER` and `CMAKE_CUDA_ARCHITECTURES` are set
*above* `project()`. Everything that just configures *targets* (standards,
`add_executable`, links, tests) comes *after*, because targets don't exist until
the project is declared.

---

## 5. Variables and cache — the part that bites beginners

There are two kinds of variables:

- **Normal variables**: `set(FOO bar)`. Live only during this configure run.
- **Cache variables**: `set(FOO bar CACHE STRING "doc")` or anything passed as
  `-DFOO=bar`. **Persist in `build/CMakeCache.txt` across runs.**

Gotcha: once a value is in the cache, re-running `cmake -B build` will **reuse
the cached value**, not the new default in your edited `CMakeLists.txt`. To
force a fresh decision you either pass `-DFOO=...` again or delete `build/`.

This is exactly the reset we did: `rm -rf build && cmake -B build`.

---

## 6. How to read what configure decided

After configuring, you can inspect the locked-in choices:

```bash
grep -i architectures build/CMakeFiles/*/CMakeCUDACompiler.cmake
# -> CMAKE_CUDA_ARCHITECTURES_NATIVE "120-real"

grep -i "CMAKE_CUDA_COMPILER" build/CMakeCache.txt
# which nvcc got chosen

# what GPU arch the .cu actually compiled to:
cuobjdump -sass ./build/<binary> | grep -oE 'sm_[0-9]+' | sort -u
```

`120-real` means "generate real SASS for sm_120" (vs. `120-virtual` = keep PTX
for forward-compat JIT). `native` expanded to `120-real` = exactly our 5090.

---

## 7. Case study: the `sm_75` bug (why Step 2 touched CMake)

Symptom: the cache showed `CMAKE_CUDA_ARCHITECTURES=75`, even though the GPU is
sm_120 and CMake had *detected* `..._NATIVE = 120-real`.

Root cause — **ordering**. In the *old* file, the architecture block sat
**after** `project(... CUDA)`:

```cmake
project(ArcWarden LANGUAGES CXX CUDA)   # <- enabling CUDA here sets
                                        #    CMAKE_CUDA_ARCHITECTURES to a
                                        #    compiler default (75)
...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES) # <- now it IS defined (=75),
  set(CMAKE_CUDA_ARCHITECTURES native)   #    so this never runs!
endif()
```

When `project()` enables the CUDA language, CMake **initializes**
`CMAKE_CUDA_ARCHITECTURES` to a fallback default (75 on this setup). By the time
our `NOT DEFINED` guard ran, the variable was already defined, so the `native`
branch was skipped — we silently built for the wrong GPU.

Fix — move the block **before** `project()` (lines 22–24 now). Then the variable
is set *before* CUDA is enabled, CMake resolves `native` during compiler
detection, and we get `120-real`. After editing we had to
`rm -rf build` to clear the cached `75` (see §5).

Lesson, restated: **compiler and architecture choices belong above
`project()`; everything cached survives until you delete `build/`.**

---

## 8. Quick reference — commands you'll use

```bash
# fresh configure (after editing CMakeLists or to clear stale cache)
rm -rf build && cmake -B build

# incremental build
cmake --build build

# build a specific target
cmake --build build --target arcwarden

# run tests
ctest --test-dir build --output-on-failure

# override choices at configure time
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=120
```

---

## 9. One-sentence summary

`CMakeLists.txt` is a top-to-bottom script with two phases (configure → build);
set compiler/architecture **before** `project()`, configure **targets** after,
attach everything with scoped `target_*` commands, and remember that
`build/CMakeCache.txt` persists decisions until you delete it.
