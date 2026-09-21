# LanguageModels

A from-scratch **C++20, header-only** implementation of the core language-model
architectures from Stanford's
[CS224N: Natural Language Processing with Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy),
built in the order the course presents them: RNN, attention, transformer,
tokenizers, BERT, GPT, RoPE and Mixture of Experts.

There are no dependencies and no framework. Tensors, layers, backpropagation
and optimizers are all written by hand, and every gradient is checked against
finite differences in the test suite.

> This is a learning project, not a production LLM library. It trains small models, 
> on CPU (parallel or sequential) and, on NVIDIA GPU for Mini Transformer model. 
> Goal is to show each idea working end to end and to keep the code readable, 
> not to compete with an established framework on scale.

Mini Transformer can train in three ways, with the same results. One training step in
`float` at the vocabulary of the full `fra.txt` (16,097 source and 30,152 target words),
Release build, RTX 4060 Ti and a 16-thread CPU:

| Execution | per step | speedup | one epoch (191,351 pairs) |
|---|---|---|---|
| `Sequential`: `trainStep`, one thread | 50 ms | 1x | about 2.7 hours |
| `Parallel`: `trainStepMultipleThread`, 16 threads | 11 ms | 4.4x | about 36 minutes |
| `Cuda`: `trainStepCuda`, on the GPU | 1.5 ms | 34x | about 4.7 minutes |

See [CUDA](#cuda-optional) and [docs/CUDA.md](docs/CUDA.md) for how it works and how it was
measured.

It is a sibling of the MachineLearningModels repository, which covers classical
(non-deep-learning) ML algorithms and uses the same project layout.

## Contents

- [What is implemented](#what-is-implemented)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Using the library](#using-the-library)
- [Pipelines](#pipelines)
- [Measuring model quality](#measuring-model-quality)
- [Design notes](#design-notes)
- [Tests](#tests)
- [Data files](#data-files)
- [Known limitations](#known-limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## What is implemented

Paths are relative to `LanguageModels/include/` for headers and to
`LanguageModels.Examples/` for the runnable drivers.

| Course stage | Library headers | Driver |
|---|---|---|
| Vanilla RNN (character-level model, BPTT) | `models/VanillaRNN.h`<br>`metrics/Metrics.h` | `VanillaRNN.cpp` |
| Attention | `layers/AttentionHead.h`<br>`layers/MultiHeadAttention.h` | |
| Embeddings (learned, sinusoidal, rotary/RoPE) | `embeddings/Embedding.h`<br>`embeddings/SinusoidalEmbedding.h`<br>`embeddings/RotaryEmbedding.h` | |
| Transformer building blocks | `layers/LinearLayer.h`<br>`layers/FeedForward.h`<br>`normalizations/RMSNorm.h` | |
| Encoder-decoder transformer (translation) | `models/MiniTransformer.h` | `TestSimpleTransformer.cpp`, `TestMiniTransformer.cpp` |
| Tokenizers (WordPiece, Unigram) | `tokenizers/WordPieceTokenizer.h`<br>`tokenizers/UnigramTokenizer.h` | `TestWordPieceTokenizer.cpp` |
| BERT (encoder-only, masked LM + next-sentence) | `models/BERT.h`<br>`models/BertLayer.h` | `TestBert.cpp` |
| GPT (decoder-only) | `models/DecoderOnlyModel.h`<br>`models/BasicGPT.h`<br>`layers/BasicDecoderBlock.h` | `TestBasicGPT.cpp` |
| Mixture of Experts | `models/MoELayer.h`<br>`layers/DecoderWithMoe.h`<br>`models/BasicGPTWithMoE.h` | `TestBasicGPTWithMoE.cpp` |
| Unigram tokenizer + GPT | `models/BasicGPTWithMoE.h` (`GPTWithUnigram`) | `TestGPTWithUnigram.cpp` |
| Evaluation (perplexity, BLEU, ROUGE, F1, baselines) | `metrics/` | `TestTinyShakespeare.cpp` |
| Pipelines (load, split, train and evaluate any model) | `pipelines/`<br>`data/`<br>`utilities/Logger.h` | `TestPipelines.cpp` |

`MiniTransformer` also decodes translations itself: `generate()` (greedy) and `beamSearch()`.

Shared infrastructure:

- `common/Tensor.h`, `common/TensorOps.h`: a flat tensor plus matmul, transpose and row softmax.
- `common/Parameter.h`: weights, gradients and optimizer state, with SGD and Adam updates.
- `common/Exceptions.h`, `common/Validation.h`: the exception types and argument checks (see [Design notes](#design-notes)).
- `common/BenchmarkTimer.h`: a simple millisecond timer.
- `config/`: per-architecture compile-time settings such as head width and context length.
- `layers/DecoderBlockConcept.h`: the `DecoderBlock` C++20 concept that decoder blocks must satisfy.
- `layers/DecoderMultiHead.h`: a multi-head decoder block. It satisfies the concept and is covered by tests, but no named model uses it yet.

## Repository layout

```
LanguageModels/               the header-only library (organized as a VS project)
  include/
    common/  config/  data/  embeddings/  layers/  metrics/
    models/  normalizations/  pipelines/  tokenizers/  utilities/
LanguageModels.Examples/      one driver per course stage, linked into one executable
LanguageModels.Tests/         Google Test unit tests, mirroring include/
LanguageModels.Cuda/          optional GPU code (CUDA), built as a static library
resources/                    data files used by the examples
Makefile                      Linux / WSL build (examples and tests)
LanguageModels.sln            Visual Studio solution
```

## Getting started

### Requirements

- A C++20 compiler. Verified with MSVC (Visual Studio) and GCC 13.
- **Visual Studio:** the projects use the `v143` (Examples) and `v145` (library
  and tests) toolsets. The `v145` toolset ships with Visual Studio 2026; with an
  older Visual Studio, retarget the projects to the toolset you have installed.
- **Linux / WSL:** `g++` and `make`. The tests also need Google Test
  (`sudo apt install libgtest-dev`).

### Visual Studio

Open `LanguageModels.sln`, set `LanguageModels.Examples` as the startup project
and build. Output goes to `bin/<Configuration>/<Platform>/`. A post-build step
copies `resources/*.txt` next to the executable so the examples can load them by
file name.

From a Developer Command Prompt:

```
msbuild LanguageModels.sln /p:Configuration=Release /p:Platform=x64
bin\Release\x64\LanguageModels.Examples.exe --quick
```

### Linux or WSL

```bash
make                     # build the examples (debug)
make BUILD=release       # optimized build
make run ARGS=--quick    # build and run the examples
make test                # build and run the unit tests
make clean
```

### CUDA (optional)

`LanguageModels.Cuda` holds the code that runs on an NVIDIA GPU. It is optional and
does not change how anything else builds. The design, kernels, measurements and checks
are described in [docs/CUDA.md](docs/CUDA.md).

- **Interface.** `include/cuda/CudaRuntime.h` and `CudaVocabularyHead.h` are plain C++
  with no CUDA includes. `cuda::isAvailable()` tells you whether a GPU can be used; the
  rest of the library falls back to the CPU when it cannot.
- **What runs on the GPU.** Almost all of a Mini Transformer training step goes into work
  that grows with the vocabulary: choosing the output word among tens of thousands, and
  updating three tables of millions of numbers when a sentence touches a few dozen of them.
  `cuda::VocabularyHead` keeps those parts on the GPU: the two embedding tables and the
  output projection, with their gradients and Adam state. The small encoder and decoder
  layers stay on the CPU, so a step moves only a few kilobytes between the two.
  `MiniTransformer::trainStepCuda()` runs a step this way, and `ExecutionStrategy::Cuda`
  selects it in a pipeline (the weights are copied back to the CPU model when training
  ends, so translating and evaluating work as before).
- **Speed.** One training step at the vocabulary of the full `fra.txt` (16,097 source and
  30,152 target words), Release build, RTX 4060 Ti and a 16-thread CPU:

  | | `double`, per step | `float`, per step | `float`, one epoch (191,351 pairs) |
  |---|---|---|---|
  | `trainStep`, one thread | 59 ms | 50 ms | about 2.7 hours |
  | `trainStepMultipleThread`, 16 threads | 18 ms | 11 ms | about 36 minutes |
  | `trainStepCuda`, on the GPU | 2.7 ms | 1.5 ms | about 4.7 minutes |

  The GPU code follows the model's type: `MiniTransformer<float>` runs a `float` head,
  which moves half as much memory and is where a GPU is fastest. The `float` model gives
  up precision (about 7 digits instead of 16), so `trainStepCuda` and `trainStep` agree
  to about 1e-4 instead of 1e-16, and `trainStepMultipleThread` is still bit-identical to
  `trainStep`. Both models learn the same way in the tests.
- **Results.** `trainStepCuda` matches `trainStep` to within rounding (about 1e-16 at first).
  Training is chaotic, so two runs that are not bit-identical drift apart over a few hundred
  steps, as a different compiler or thread order would also make them; both learn equally
  well. `trainStepMultipleThread` is bit-identical to `trainStep`.
- **Visual Studio.** With the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  installed (`CUDA_PATH` set) and an x64 configuration, the `.cu` files are compiled by
  `nvcc` through `LanguageModels.Cuda/nvcc-build.cmd`, for compute capability 8.9 by
  default (an RTX 40 series card). Build for another GPU with
  `/p:CudaComputeCapability=86`. No Visual Studio CUDA integration is needed. Without
  the toolkit, or for Win32, a stand-in is built instead and the solution still builds.
- **Toolset.** The project uses `v143` on purpose: `nvcc` of CUDA 13.0 does not accept
  the newer `v145` compiler as its host compiler.
- **Linux / WSL.** `make CUDA=1` (also for `tests` and `run`) compiles the GPU code with
  `nvcc` (set `CUDA_PATH` and `CUDA_ARCH` if they differ from `/usr/local/cuda` and `89`).
  Without `CUDA=1` the stand-in is used.
- **Tests.** The CUDA tests print the GPU they ran on and report themselves skipped
  where there is none, so the suite passes on every machine.
- **Checking the kernels.** NVIDIA's `compute-sanitizer` (in the toolkit) finds what the
  tests cannot, such as memory that is read before it is written, which usually happens
  to hold zeros. It reports no errors for the GPU tests with `--tool memcheck`,
  `--tool racecheck` and `--tool initcheck`:

  ```
  compute-sanitizer --tool initcheck bin\Debug\x64\LanguageModels.Tests.exe --gtest_filter=CudaVocabularyHead*
  ```

### What the example program does

`LanguageModels.Examples` runs every course stage in order: Vanilla RNN,
Simple Transformer, Mini Transformer, BERT, Basic GPT, Basic GPT with Mixture of
Experts, GPT with Unigram tokenizer, WordPiece tokenizer, the Tiny Shakespeare
benchmark (see [Measuring model quality](#measuring-model-quality)), and four
[pipeline](#pipelines) stages (translation, character RNN, character GPT, BERT).
Each stage prints its loss as it trains and a small inference demo at the end. A
stage that throws is reported and the next one still runs.

The **Mini Transformer** stage trains an English to French model on up to 5000
sentence pairs from `fra.txt`, which takes hours (about 20 s per epoch for 500
epochs in a Release build on a desktop PC). Pass `--quick` to train on the
100-pair `fra_debug.txt` instead; the whole program then finishes in about a
minute in a Release build:

```
LanguageModels.Examples.exe --quick
```

`fra.txt` is not stored in the repository; see [Data files](#data-files). In
Visual Studio, set `--quick` under Project Properties > Debugging > Command
Arguments.

Pass `--parallel` to run the pipeline stages with `ExecutionStrategy::Parallel`,
which trains the Mini Transformer on several threads (the other models train the
same way either way).

## Using the library

Everything is header-only: add the `include/` subfolders to your include path
and include the model you want. This example trains a small GPT on the pattern
`0 1 2 0 1 2 ...`, generates from a prompt, and shows the error handling:

```cpp
#include <iostream>
#include <vector>

#include "BasicGPT.h"
#include "BERT.h"

int main() {
    // Vocabulary of 10 tokens, model width 16, 2 layers, context of 20 tokens.
    // Weights come from a fixed default seed.
    BasicGPT gpt(10, 16, 2, 20);

    // Predict each next token of the repeating pattern.
    std::vector<std::size_t> input = { 0, 1, 2, 0, 1 };
    std::vector<std::size_t> target = { 1, 2, 0, 1, 2 };
    for (int step = 0; step < 200; step++) {
        double loss = gpt.trainStep(input, target, 0.05);
        if (step % 50 == 0) {
            std::cout << "step " << step << " loss " << loss << "\n";
        }
    }

    // Greedy generation from a prompt: prints 0 1 2 0 1 2 0 1 2 0
    for (std::size_t token : gpt.generate({ 0, 1 }, 8)) {
        std::cout << token << ' ';
    }
    std::cout << "\n";

    // Bad input throws a specific exception type.
    try {
        gpt.trainStep(input, target, -1.0);
    }
    catch (const InvalidParameterError& error) {
        std::cout << "rejected: " << error.what() << "\n";
    }

    // Encoder models take an explicit optimizer; Adam needs a 1-based step count.
    BertModel<double> bert(20, 16, 2, 12);
    double loss = bert.trainStep({ 2, 7, 4, 3, 9, 3 }, { 0, 0, 0, 0, 1, 1 },
        { 0, 0, 5, 0, 0, 8 }, 1, 0.001, UpdateRule::adam(1));
    std::cout << "bert loss " << loss << "\n";
}
```

The `LanguageModels.Examples/` drivers show complete training loops for every
model, including tokenization and evaluation.

## Pipelines

A `Pipeline` runs the same five steps for every model: load the data, split it,
train, evaluate. Only the model, its settings and the kind of data change:

```cpp
#include "DataLoader.h"
#include "DataSplitter.h"
#include "MiniTransformerPipeline.h"

void runTranslation(Logger& logger) {
    // Tab-separated sentence pairs, normalized; then a shuffled 80/20 split.
    auto samples = DataLoader::loadSentencePairs("fra.txt", 5000);
    auto dataSet = DataSplitter::trainTestSplit(std::move(samples));

    TranslationParameters<double> parameters;
    parameters.epochs = 30;

    Pipeline<double, MiniTransformer<double>, TranslationParameters<double>>
        pipeline{ parameters, ExecutionStrategy::Parallel, logger };

    pipeline.train(dataSet.trainingData);
    TranslationMetrics metrics = pipeline.evaluate(dataSet.testData);

    logger.info() << "BLEU-4: " << metrics.bleu
        << ", word error rate: " << metrics.wordErrorRate;
    std::cout << pipeline.modelAdapter().translate("hello") << "\n";
}
```

Models that have a pipeline (header in `pipelines/`):

- **`MiniTransformer<T>`** (`MiniTransformerPipeline.h`)
  - Sample: `SentencePair`. Settings: `TranslationParameters`.
  - `evaluate()` returns `TranslationMetrics`: BLEU-4, word error rate, exact
    matches and teacher-forced perplexity.
- **`VanillaRNN<T>`** (`VanillaRnnPipeline.h`)
  - Sample: `char`, so the data is a text. Settings: `RnnParameters`.
  - `evaluate()` returns `LanguageModelMetrics`: perplexity and accuracy next to
    unigram and bigram baselines.
- **`DecoderOnlyModel<T, ...>`, for example `BasicGPT`** (`GptPipeline.h`)
  - Sample: `char`, so the data is a text. Settings: `GptParameters`.
  - `evaluate()` returns `LanguageModelMetrics`.
- **`BertModel<T>`** (`BertPipeline.h`)
  - Sample: `std::string`, one sentence. Settings: `BertParameters`.
  - `evaluate()` returns `BertMetrics`: hidden-word perplexity and accuracy, and a
    next-sentence confusion matrix.

How it fits together:

- **Data.** `DataLoader` reads sentence pairs, a text as characters, or a text as
  normalized lines. `DataSplitter::trainTestSplit()` shuffles (for independent examples
  such as sentence pairs); `DataSplitter::sequentialSplit()` cuts once and keeps the
  order (for text read as a stream, and for BERT's "next sentence").
- **Vocabulary comes from the training data.** A model cannot even be built before it
  has seen the data, so each adapter builds its `WordTokenizer` or `CharTokenizer` from
  the training part only. Words and characters unseen in training become `<UNK>`, so
  held-out text is scored fairly and nothing leaks from the test set.
- **`ExecutionStrategy`** is `Sequential`, `Parallel` or `Cuda`. Only `MiniTransformer` acts
  on it: `Sequential` calls `trainStep()`, `Parallel` calls `trainStepMultipleThread()`
  (bit-identical results, several threads) and `Cuda` calls `trainStepCuda()` (results
  equal to within rounding, the vocabulary-sized work on the GPU; it throws `CudaError`
  when there is no CUDA device, see [CUDA](#cuda-optional)). The other models accept the
  strategy and train the same way. More strategies can be added to the enum later.
- **Logging.** Progress goes through `Logger` (`logger.info() << ...`), the same
  interface as the sibling MachineLearningModels project. The default logger is shared
  and writes to `std::clog`; pass your own to choose the level or the stream.
- **Adding a model.** Specialize `ModelAdapter<T, Model, Parameters>` so that it
  satisfies the `PipelineAdapter` concept (`Sample`, `Parameters`, `Result`, `train()`,
  `evaluate()`), as `MiniTransformerPipeline.h` does.

Errors are typed: `train()` and `evaluate()` throw `InvalidSizeError` for empty or
unusable data, and `evaluate()` before `train()` throws `PipelineStateError`; a missing
file throws `DataLoadError`.

## Measuring model quality

`include/metrics/` scores models independently of how they are built, so any two
models can be compared on the same footing:

- **`Metrics.h`**: loss, perplexity, next-token accuracy and bits per token. The result
  type of `VanillaRNN::evaluate()` and `DecoderOnlyModel::evaluate()`.
- **`LogitMetrics.h`**: the same, straight from a `[rows, vocab]` logits tensor
  (`scoreLogits`), `MetricsAccumulator` to pool many windows of text, and
  `topKAccuracy`. For GPT, Mini Transformer and BERT's masked-word head.
- **`ConfusionMatrix.h`**: accuracy, and per-class and macro precision, recall and F1.
  For classification heads such as BERT's next-sentence prediction.
- **`TextMetrics.h`**: BLEU (sentence and corpus), ROUGE-1/2/L, edit distance, and word
  and character error rate. For translation and generation quality.
- **`NGramBaseline.h`**: unigram and bigram language models, a yardstick a trained
  model has to beat.

```cpp
// Perplexity of a decoder-only model on one window of text.
Metrics metrics = gpt.evaluate(input, target);
std::cout << "perplexity " << metrics.perplexity << "\n";

// Translation quality of generated sentences against references.
auto score = evaluation::corpusBleu(candidates, references);   // BLEU-4
```

**Checking that the numbers are right.** Perplexity means little alone, so the
tests pin the metrics to values that can be worked out independently:

- BLEU is checked on the worked examples from the original paper (Papineni et al.,
  2002), for instance modified precisions of 17/18 and 10/17 for its good candidate.
- Edit distance, F1 and ROUGE use textbook examples (`kitten` to `sitting` is 3 edits).
- **Tiny Shakespeare**, the standard small character-level benchmark, has a published
  size (1,115,394 characters, 65 distinct). Its baselines follow from counting: a
  uniform guess has perplexity 65, letter frequencies give 27.5 (4.78 bits per
  character), and one character of context gives 11.6. The tests compare
  `NGramBaseline` with values from an independent Python implementation, and check
  that a character RNN trained for a few hundred steps scores well below 65 on text
  it has not seen. The `Tiny Shakespeare Benchmark` example stage trains the RNN
  longer and prints all of these side by side (`--quick` shortens it).

## Design notes

**One decoder-only model, several blocks.** `DecoderOnlyModel<T, BlockT, Config>`
is the single GPT implementation. `BasicGPT`, `BasicGPTWithMoE` and
`GPTWithUnigram` are aliases that plug in different decoder blocks. `BlockT` must
satisfy the `DecoderBlock` concept, so a wrong block type is a readable compile
error.

**Compile-time RoPE tables.** The rotary embedding's cosine and sine tables are
sized at compile time from a config type (`Config::d_head`, `Config::maxSeqLen`),
so each model can use its own head width and context length instead of one global
constant. See `config/`.

**Deterministic by default.** There is no global random engine. Each model builds
its own from a seed (default 42, the same in Debug and Release) and passes it to
every layer while the weights are drawn, so a given seed always produces the same
model.

**Optimizers.** `Parameter::update(lr, UpdateRule)` applies either plain SGD or
Adam (`UpdateRule::sgd()` / `UpdateRule::adam(step)`). Adam's moment buffers are
allocated on first use, and gradients are clipped element-wise to [-1, 1].

**Multi-threaded training step.** `MiniTransformer::trainStepMultipleThread()` is
`trainStep()` with the vocabulary-sized work (output projection, softmax, and clearing
and updating the embedding tables) split across `std::thread`s from a small shared
`ThreadPool`. The tiny encoder and decoder layers stay on the calling thread. Every
element is computed with the same operations in the same order, so the loss and the
weights are bit-identical to `trainStep()` at any thread count. With a 3,000 / 5,000
word vocabulary on a 16-thread desktop it ran about 4.5 times faster (Release build).

**Errors are typed, not silent.** Invalid input throws instead of producing
`NaN`s or crashing:

| Exception | Meaning |
|---|---|
| `NaNError`, `NonFiniteError` | a value became `NaN` or infinite (both derive from `std::domain_error`) |
| `DivisionByZeroError` | a denominator was exactly zero |
| `InvalidParameterError` | an argument value is out of range (learning rate, token id, label) |
| `InvalidSizeError` | data has the wrong shape or length |
| `InvalidParameterSizeError` | a configuration size is zero or inconsistent (a subtype of `InvalidSizeError`) |

**Hand-written backpropagation.** Each layer implements `forward()` and
`backward()` explicitly and accumulates gradients, which are cleared with
`zeroGrad()` before each step. Nothing is hidden behind an autograd engine.

## Tests

`LanguageModels.Tests` is a [Google Test](https://github.com/google/googletest)
project with about 630 tests. There is one `*Tests.cpp` per library header, under
`unit/<folder>/`. They cover:

- exact behavior: known matrix products, softmax values, RoPE angles, metrics formulas;
- **gradient correctness:** a finite-difference checker (`TestSupport.h`) verifies
  every hand-written `backward()`, including attention with RoPE and full decoder blocks;
- properties: causal masking (a decoder never sees later tokens), determinism for a
  given seed, and optimizer convergence;
- error handling: every exception type and every invalid-argument path;
- learning: the models actually reduce loss and learn small patterns.

Run them:

- **Visual Studio:** build the solution and run `LanguageModels.Tests`, or use Test Explorer
  with the Google Test Adapter. Google Test comes from the
  `Microsoft.googletest.v140.windesktop.msvcstl.static.rt-dyn` NuGet package listed in
  `LanguageModels.Tests/packages.config`; Visual Studio restores it on build.
- **Command line:** `bin\Debug\x64\LanguageModels.Tests.exe`, optionally with
  `--gtest_filter=RMSNorm*`.
- **Linux / WSL:** `make test`.

## Data files

`resources/` holds the data used by the examples:

| File | Tracked in git | Used by |
|---|---|---|
| `vocab_test.txt` | yes | the WordPiece tokenizer stage |
| `fra_debug.txt` | yes (100 sentence pairs) | the Mini Transformer stage with `--quick` |
| `fra.txt` | **no** (about 36 MB) | the Mini Transformer stage by default |
| `tinyshakespeare.txt` | yes (1.1 MB) | the Tiny Shakespeare benchmark stage and its tests |

`fra.txt` is a tab-separated English to French sentence-pair file (English, French
and an optional attribution column per line), the format of the Tatoeba-derived
files published for example at [manythings.org/anki](https://www.manythings.org/anki/).
To run the full Mini Transformer training, download a file in that format, name it
`fra.txt` and place it in `resources/`. If it is missing, that stage reports that it
could not load the data and the remaining stages still run; `--quick` needs nothing
extra.

The sentence pairs in `fra_debug.txt` come from the [Tatoeba](https://tatoeba.org)
project, licensed [CC BY 2.0 (France)](https://creativecommons.org/licenses/by/2.0/fr/).
Each line keeps its attribution column (`Attribution: tatoeba.org #<sentence ids>`), so
the credit travels with the data. The code in this repository is under the MIT license
below; the Tatoeba data keeps its own license.

`tinyshakespeare.txt` is the Tiny Shakespeare text distributed with Andrej Karpathy's
[char-rnn](https://github.com/karpathy/char-rnn) project (`data/tinyshakespeare/input.txt`),
a concatenation of Shakespeare's plays, whose text is in the public domain. The char-rnn
repository does not state a license for the compiled file.

## Known limitations

- **Small scale.** Single-threaded CPU code with batch size 1; the models use widths
  of 16 to 32. Training a real-sized model is not a goal.
- **Mixture of Experts.** The router is not trained (it receives no gradient) and the
  load-balancing auxiliary loss is not computed yet. The experts themselves do train.
- **Mini Transformer.** Its loss on the sentence-pair data drops quickly and then
  plateaus, and held-out translations are poor. The split is by position in an
  alphabetically sorted file, so most held-out sentences never occur in training.
  The stage demonstrates the training loop and the decoders (greedy and beam search),
  not translation quality.
- **`VanillaRNN::sample()`** draws from the C library's `rand()` rather than the
  model's seeded engine, so call `std::srand` first if you need repeatable samples.
- **Not included:** RLHF and preference optimization (reward models, PPO, DPO). That is
  planned as its own project.

## Acknowledgements

Built while following Stanford's
[CS224N](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy)
lectures.

## License

The code is released under the [MIT License](LICENSE).
