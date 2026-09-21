# CUDA support

The Mini Transformer can train on an NVIDIA GPU. It is optional: without a GPU, or
without the CUDA Toolkit, everything else builds and runs on the CPU as before.

One epoch of the full `fra.txt` (191,351 sentence pairs, 16,097 / 30,152 words) takes about
3 hours on one CPU thread, 36 minutes on 16 threads, and under 5 minutes with the GPU
(`float`, RTX 4060 Ti).

## Contents

- [Why the GPU helps here](#why-the-gpu-helps-here)
- [What runs where](#what-runs-where)
- [The kernels](#the-kernels)
- [Using it](#using-it)
- [Speed](#speed)
- [Correctness](#correctness)
- [Building](#building)
- [Limits](#limits)

## Why the GPU helps here

The model is tiny: width 32, a few tokens per sentence, batch size 1. A GPU is not built for
that, and the encoder and decoder layers are too small to be worth sending over. What is
large is the **vocabulary**. With `fra.txt` the model has about 2.5 million numbers, and
almost all of them sit in three tables:

| table | size | used by a step |
|---|---|---|
| source embeddings | 16,097 x 32 | a few dozen rows |
| target embeddings | 30,152 x 32 | a few dozen rows |
| output projection (+ bias) | 32 x 30,152 | every column: one score per possible word |

Choosing the output word means scoring 30,152 words at every position, and Adam then
updates all 2.5 million numbers, three arrays each (value, first and second moment). That
work grows with the vocabulary, not with the model, and it is the same operation on
millions of independent numbers, which is exactly what a GPU is good at. Measured on the
CPU it is almost the whole step.

## What runs where

```
  CPU                                              GPU (cuda::VocabularyHead)
  ---                                              --------------------------
  token ids  ---------------------------------->   gather embedding rows
  encoder + decoder layers (width 32)  <---------  embedding rows (a few KB)
  decoder output  ----------------------------->   project to 30,152 scores
                                                   softmax + cross-entropy loss
                                                   gradients of projection, bias
  backward through decoder and encoder  <-------   gradient of decoder output
  embedding-row gradients  -------------------->   scatter-add into table gradients
                                                   Adam / SGD on all three tables
```

The GPU keeps the two embedding tables, the projection, their gradients and Adam's moment
estimates for as long as training runs. Per step only a few kilobytes cross to the GPU and
back: token ids, some embedding rows, the decoder output and its gradient. The weights are
copied back to the CPU model once, when training ends, so translating and evaluating work
as they always did.

This split is deliberate. Moving the whole model would need batching to pay off. Moving
only the vocabulary-sized part gives most of the speedup with a small amount of code, and
the layers stay readable on the CPU.

## The kernels

All are in [`LanguageModels.Cuda/src/detail/VocabularyKernels.cuh`](../LanguageModels.Cuda/src/detail/VocabularyKernels.cuh).
The rest of the GPU code is split by job:

| file | contents |
|---|---|
| `src/CudaVocabularyHead.cu` | buffers, kernel launches and the public `VocabularyHead` class |
| `src/CudaRuntime.cu` | device query and the `saxpy` smoke test |
| `src/detail/VocabularyKernels.cuh` | the kernels below |
| `src/detail/DeviceMath.cuh` | block reductions, gradient clipping, the not-finite flag |
| `src/detail/DeviceArray.cuh` | GPU memory that frees itself |
| `src/detail/CudaHost.cuh` | error checking and launch sizes |

The kernels are templates, so they are compiled in the same file that launches them
(`CudaVocabularyHead.cu` includes the headers); putting them in a separate `.cu` would need
relocatable device code, which is slower and needs an extra device-link step.

| kernel | what it does | layout |
|---|---|---|
| `gatherRowsKernel` | embedding lookup: `out[r] = table[ids[r]]` | one thread per number |
| `projectKernel` | `logits = x * W + bias` | one thread per word, one block row per position |
| `softmaxCrossEntropyKernel` | softmax over the vocabulary, the loss, and the gradient `p - onehot` in place | one block per position, block reductions for max and sum |
| `projectionGradientKernel` | gradients of `W` and the bias | one thread per word, one block row per input feature |
| `inputGradientKernel` | gradient sent back to the decoder | one block per output number, block-sum over the vocabulary |
| `scatterAddRowsKernel` | adds embedding-row gradients into the table gradient | one block, see below |
| `sgdKernel`, `adamKernel` | the optimizer step, with the CPU's arithmetic | one thread per number |

A few decisions are worth knowing about:

- **Softmax is computed stably.** The maximum is subtracted before `exp`, using a
  block-wide max and sum reduction in shared memory.
- **Repeated words are added in order.** A word can occur twice in a sentence, and both
  rows must be added to the same gradient row. Doing that with atomics would make the
  result depend on which thread is first. Instead one block owns each column and adds the
  rows in position order, so the result is deterministic. The work is tiny (a few dozen
  rows), so a single block costs nothing.
- **The optimizer copies the CPU's rules.** Both kernels clip each gradient to [-1, 1] and use
  the same betas, epsilon and bias correction as `Parameter::update()`.
- **Bad numbers are reported, not hidden.** The update kernels set a flag if a gradient or
  an updated weight is NaN or infinite. The host reads the flag and throws `NaNError` or
  `NonFiniteError`, as the CPU code does.
- **Adam's moments are allocated on first use and cleared to zero.** SGD training never
  pays for them.

## Using it

From a pipeline, one line:

```cpp
Pipeline<float, MiniTransformer<float>, TranslationParameters<float>>
    pipeline(parameters, ExecutionStrategy::Cuda);

pipeline.train(pairs);      // uploads, trains on the GPU, downloads the weights
pipeline.evaluate(pairs);   // on the CPU, as before
```

`ExecutionStrategy::Cuda` throws `CudaError` if there is no usable GPU. Check first with
`cuda::isAvailable()` and fall back to `ExecutionStrategy::Parallel` if you prefer.

Directly on the model, when you write your own loop:

```cpp
MiniTransformer<float> model(sourceVocab, targetVocab);
auto head = model.createCudaHead();          // uploads the weights

for (std::size_t step = 1; step <= steps; step++) {
    float loss = model.trainStepCuda(head, source, decoderInput, expected,
                                     0.001f, UpdateRule::adam(step));
}
model.downloadFromCudaHead(head);            // weights back to the CPU model
```

`cuda::VocabularyHead<T>` itself (`include/cuda/CudaVocabularyHead.h`) is plain C++ with no
CUDA includes. It is defined for `float` and `double`; the `double` head matches the CPU
model to within rounding, the `float` head is faster.

The interface headers never include CUDA headers, so any C++ compiler can use them. The
GPU code sits behind a pointer-to-implementation and lives only in `LanguageModels.Cuda`.

## Speed

One training step (Adam), Release build, RTX 4060 Ti and a 16-thread CPU:

| vocabulary (source / target) | | one thread | 16 threads | GPU |
|---|---|---|---|---|
| 3,000 / 5,000 | `double` | 10.6 ms | 3.0 ms | 1.3 ms |
| | `float` | 8.1 ms | 1.8 ms | 1.0 ms |
| 16,097 / 30,152 (`fra.txt`) | `double` | 58.6 ms | 17.8 ms | 2.7 ms |
| | `float` | 50.3 ms | 11.4 ms | 1.5 ms |

So the GPU is 22x faster than one CPU thread for `double` and 34x for `float` at the full
vocabulary, and 8x at a small one. The gain grows with the vocabulary, because the CPU
work grows with it and the GPU has spare capacity. Measured on the GPU part alone (the
head, without the CPU layers), `float` takes 0.6 ms and `double` 1.7 ms.

What is left in a 1.5 ms GPU step is mostly the CPU layers and the transfers. Making it
faster would mean batching several sentences per step, which changes the training method,
so it was left out.

The CPU multithreading measured here (`trainStepMultipleThread`) gives bit-identical
results to `trainStep`. See the main README.

## Correctness

Speed only counts if the result is right. What is checked:

- **Against the CPU.** The tests run the same step on the CPU model and the GPU head and
  compare loss, gradients, and weights after several steps. With `double` they agree to
  about 1e-16 at first, with `float` to about 1e-4.
- **Chaos, not a bug.** Training is chaotic: two runs that differ by one rounding error
  drift apart over a few hundred steps (1e-16 became 1e-9 after about 250). So long runs are
  compared on quality (does it reach the same loss, does it translate its sentence), not
  on exact values. One test that compared exact losses after 300 steps failed for this
  reason, and the cause was traced before the test was changed.
- **Errors.** Wrong sizes, ids outside the vocabulary, a zero learning rate, and NaN or
  infinite gradients throw the same exceptions as the CPU code.
- **NVIDIA's compute-sanitizer.** `memcheck`, `racecheck` and `initcheck` report no errors
  for the GPU tests. `initcheck` found a real bug that the tests had missed: Adam's moments
  were read before they were written, which happened to work because fresh GPU memory
  often holds zeros. It is fixed.

  ```
  compute-sanitizer --tool initcheck bin\Debug\x64\LanguageModels.Tests.exe --gtest_filter=CudaVocabularyHead*
  ```
- **Every machine passes.** The CUDA tests print the GPU they ran on and report themselves
  skipped where there is none. The suite passes with MSVC and GCC, with and without the
  toolkit.

## Building

**Visual Studio.** With the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
installed (`CUDA_PATH` set) and an x64 configuration, `nvcc` compiles the `.cu` files
through [`LanguageModels.Cuda/nvcc-build.cmd`](../LanguageModels.Cuda/nvcc-build.cmd), for
compute capability 8.9 (RTX 40 series) by default. For another GPU:

```
msbuild LanguageModels.sln /p:Configuration=Release /p:Platform=x64 /p:CudaComputeCapability=86
```

The Visual Studio CUDA integration is not needed. Without the toolkit, or for Win32, a
stand-in (`CudaUnavailable.cpp`, `CudaVocabularyHeadUnavailable.cpp`) is built instead, so
the solution still builds; `cuda::isAvailable()` returns `false`.

The project uses toolset `v143` on purpose. `nvcc` from CUDA 13.0 does not accept the newer
`v145` compiler as its host compiler.

**Linux / WSL.**

```
make CUDA=1            # also: make CUDA=1 tests, make CUDA=1 run
```

Set `CUDA_PATH` and `CUDA_ARCH` if they differ from `/usr/local/cuda` and `89`. Without
`CUDA=1` the stand-in is used.

## Limits

- **Only `MiniTransformer`.** Vanilla RNN, GPT and BERT have no GPU path; their pipelines
  accept an `ExecutionStrategy` but only the Mini Transformer acts on it.
- **Batch size 1.** One sentence per step. This is a property of the whole library, not just
  the GPU code.
- **Dense updates.** Adam updates every number of the three tables at every step, even the
  rows a sentence did not touch. It is what the CPU does too, so results match, but a sparse
  update would be faster still.
- **One GPU, compute capability 8.9 by default.** Other cards need the option above.
- **Memory.** At the `fra.txt` vocabulary the tables, gradients and Adam state take about 80 MB
  in `double` (half that in `float`), far below the 8 GB of the test card.
