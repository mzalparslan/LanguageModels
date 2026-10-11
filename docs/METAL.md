# Metal support (Apple GPU)

The Mini Transformer can train on an Apple GPU through Metal, the same way it can on an NVIDIA
GPU through CUDA. It is optional: on any other platform, or on a Mac with no GPU to give (a
virtual machine, for example), everything else builds and runs on the CPU as before.

You select it the same way for both GPUs:

```cpp
Pipeline<float, MiniTransformer<float>, TranslationParameters<float>>
    pipeline(parameters, ExecutionStrategy::Device);
```

`Device` means "the GPU this machine has". CUDA is used when an NVIDIA GPU is present,
otherwise Metal, and `DeviceError` is thrown when there is neither.

## Status

This code was written on a Windows machine, which cannot compile or run Metal, so it has not
been run on the author's own hardware. What was checked there is everything around it: the
backend selection, the stand-ins used on other platforms, and the CUDA path through the same
interface.

The Metal code itself is built and tested by the macOS job of the CI workflow. On its first
run the job compiled the Objective-C++ and the kernels without changes, the runner reported a
Metal device ("Apple Paravirtual device", unified memory), and the Metal tests passed against
the CPU reference. That is a virtual GPU, so it shows the code is correct more than it shows
speed; timings from the same runner are in [Speed](#speed). The workflow reports on each run
whether the runner had a device; where there is none the Metal tests skip themselves.

## What is the same as CUDA

The design is that of [CUDA.md](CUDA.md), and the code follows it closely:

- the vocabulary-sized part of a training step runs on the GPU (the two embedding tables and
  the output projection, with their gradients and Adam state) and the small layers stay on
  the CPU;
- the eight kernels are the same, in Metal Shading Language: gather rows, scatter-add,
  project, softmax with cross-entropy, projection gradient, input gradient, SGD and Adam;
- the same details hold: a stable softmax, a single threadgroup for the scatter-add so a word
  that occurs twice is added in order, the CPU's clipping and Adam arithmetic, and a flag
  that reports NaN and infinite values;
- both implement the same interface, so `gpu::VocabularyHead` can forward to either.

The kernels are in
[`LanguageModels.Metal/src/VocabularyKernelsSource.h`](../LanguageModels.Metal/src/VocabularyKernelsSource.h)
and the host code in
[`LanguageModels.Metal/src/MetalVocabularyHead.mm`](../LanguageModels.Metal/src/MetalVocabularyHead.mm).

## What is different

| | CUDA | Metal |
|---|---|---|
| Precision | `float` and `double` | `float` only: Metal has no 64-bit floating point |
| Memory | separate GPU memory, copied | shared with the CPU on Apple Silicon: one buffer both can read |
| Ordering | the default stream orders everything | every GPU pass is committed and waited for before the CPU touches a buffer again |
| Kernels | compiled by `nvcc` at build time | compiled from a source string when the first head is created |
| Reductions | warp shuffles | simdgroup operations (`simd_max`, `simd_sum`) |

Three consequences:

- **No `double`.** `MiniTransformer<double>` cannot use a Metal GPU. `gpu::isAvailable<double>()`
  is `false` on a Mac, and `Device` throws `DeviceError` saying so. Use a `float` model.
- **Waiting each step.** Because the CPU and the GPU share buffers, the host waits for each
  pass to finish. A step makes a handful of passes. This is the simple and safe choice; fewer
  passes and no waiting are possible optimizations once there are numbers to guide them.
- **Kernels as a string.** Compiling shaders ahead of time needs Apple's offline Metal compiler,
  which recent Xcode versions ship as an optional download. The compiler that is part of the
  operating system needs nothing extra, so the build needs only a C++ compiler and the Metal
  framework. The cost is a short compile when the first head is created. Math is compiled in
  its safe mode, not the fast mode, so NaN checks and `exp` and `log` behave as expected.

## Building

The Makefile builds `LanguageModels.Metal/` on macOS and nowhere else, as Objective-C++ with
ARC, and links the Metal and Foundation frameworks:

```bash
brew install googletest
export CPATH="$(brew --prefix)/include" LIBRARY_PATH="$(brew --prefix)/lib"
make test BUILD=release
```

Everywhere else the interface headers (`include/gpu/MetalRuntime.h` and
`MetalVocabularyHead.h`) contain inline stand-ins: `metal::isAvailable()` is `false` and
creating a head throws `MetalError`. There is nothing to build or link, which is why the
Visual Studio solution has no Metal project.

## Tests

- `unit/metal/MetalRuntimeTests.cpp` and `MetalVocabularyHeadTests.cpp` check the Metal head
  against the CPU code it replaces, computed in double precision, with tolerances of about
  1e-4 because the head works in float.
- `unit/gpu/GpuVocabularyHeadTests.cpp` checks the choice of backend and that the wrapper
  forwards faithfully. It runs against whichever GPU is present.
- `MiniTransformerFloatTests.cpp` runs the whole model and pipeline in float on the GPU, so it
  covers Metal on a Mac and CUDA on an NVIDIA machine.

Tests that need a GPU print that they were skipped where there is none.

## Speed

Measured by the Benchmark workflow (`.github/workflows/benchmark.yml`, which runs
`LanguageModels.Benchmarks/Benchmark.cpp`) on GitHub's `macos-latest` runner: an Apple M1
(Virtual) with 3 cores and 7 GB, and a paravirtualized GPU shared with other jobs. One
training step in `float` with Adam, Release build, mean over many steps:

| Vocabulary (source / target) | `Sequential` | `Parallel` (3 threads) | `Device` (Metal) |
|---|---|---|---|
| 3,000 / 5,000 | 3.92 ms | 2.27 ms (1.7x) | 3.50 ms (1.1x) |
| 16,097 / 30,152 (the full `fra.txt`) | 18.5 ms | 13.6 ms (1.4x) | 8.52 ms (2.2x) |

One epoch of the full data (191,351 pairs) is then about 59 minutes sequentially, 43 minutes on
three threads and 27 minutes on the GPU.

How to read this:

- **It is a virtual machine.** The CPU has three cores that other jobs share, and the GPU is
  virtual, so the numbers are noisy and say little about a physical Mac. They cannot be compared
  with the NVIDIA table in [CUDA.md](CUDA.md), which is a different machine.
- **The gain grows with the vocabulary**, as it does with CUDA: the GPU does the work that grows
  with the vocabulary, and at 3,000 words there is too little of it to pay for the GPU.
- **The gain is much smaller than CUDA's** (2.2x here, 42x there at the full vocabulary). Some of
  the reason is probably that this single-threaded CPU is fast and the step is not completely
  dominated by the vocabulary here, and some that every step waits for several GPU passes, which
  a virtual GPU may make expensive. Neither explanation has been measured. Profiling on a physical
  Mac with Xcode's Metal tools is the way to find out, and fewer waits per step is the obvious
  first thing to try.

## Limits

- Only `MiniTransformer` has a GPU path, and only in `float` on Metal.
- Batch size is 1, as everywhere in the library.
- Dense updates: Adam updates all three tables at every step, as the CPU does.
- Timings exist only from a virtual machine, and there is no profiling with Xcode's Metal tools.
