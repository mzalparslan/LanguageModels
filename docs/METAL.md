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
the CPU reference. That is a virtual GPU, so it shows the code is correct, not that it is
fast: no timings exist yet, and this document makes no speed claims. The workflow reports on
each run whether the runner had a device; where there is none the Metal tests skip themselves.

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

## Limits

- Only `MiniTransformer` has a GPU path, and only in `float` on Metal.
- Batch size is 1, as everywhere in the library.
- Dense updates: Adam updates all three tables at every step, as the CPU does.
- No timings on a physical Mac yet, and no profiling with Xcode's Metal tools.
