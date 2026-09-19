# LanguageModels

A from-scratch **C++20, header-only** implementation of the core language-model
architectures from Stanford's
[CS224N: Natural Language Processing with Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy),
built in the order the course presents them: RNN, attention, transformer,
tokenizers, BERT, GPT, RoPE and Mixture of Experts.

There are no dependencies and no framework. Tensors, layers, backpropagation
and optimizers are all written by hand, and every gradient is checked against
finite differences in the test suite.

> This is a learning project, not a production LLM library. It runs on the
> CPU with batch size 1 and small models. The goal is to show each idea working
> end to end and to keep the code readable, not to compete with an established
> framework on speed or scale.

It is a sibling of the MachineLearningModels repository, which covers classical
(non-deep-learning) ML algorithms and uses the same project layout.

## Contents

- [What is implemented](#what-is-implemented)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Using the library](#using-the-library)
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
| Vanilla RNN (character-level model, BPTT) | `models/VanillaRNN.h`, `metrics/Metrics.h` | `VanillaRNN.cpp` |
| Attention | `layers/AttentionHead.h`, `layers/MultiHeadAttention.h` | |
| Embeddings (learned, sinusoidal, rotary/RoPE) | `embeddings/Embedding.h`, `embeddings/SinusoidalEmbedding.h`, `embeddings/RotaryEmbedding.h` | |
| Transformer building blocks | `layers/LinearLayer.h`, `layers/FeedForward.h`, `normalizations/RMSNorm.h` | |
| Encoder-decoder transformer (translation) | `models/MiniTransformer.h` | `TestSimpleTransformer.cpp`, `TestMiniTransformer.cpp` |
| Tokenizers (WordPiece, Unigram) | `tokenizers/WordPieceTokenizer.h`, `tokenizers/UnigramTokenizer.h` | `TestWordPieceTokenizer.cpp` |
| BERT (encoder-only, masked LM + next-sentence) | `models/BERT.h`, `models/BertLayer.h` | `TestBert.cpp` |
| GPT (decoder-only) | `models/DecoderOnlyModel.h`, `models/BasicGPT.h`, `layers/BasicDecoderBlock.h` | `TestBasicGPT.cpp` |
| Mixture of Experts | `models/MoELayer.h`, `layers/DecoderWithMoe.h`, `models/BasicGPTWithMoE.h` | `TestBasicGPTWithMoE.cpp` |
| Unigram tokenizer + GPT | `models/BasicGPTWithMoE.h` (`GPTWithUnigram`) | `TestGPTWithUnigram.cpp` |

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
    common/  config/  embeddings/  layers/
    metrics/  models/  normalizations/  tokenizers/
LanguageModels.Examples/      one driver per course stage, linked into one executable
LanguageModels.Tests/         Google Test unit tests, mirroring include/
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

### What the example program does

`LanguageModels.Examples` runs every course stage in order: Vanilla RNN,
Simple Transformer, Mini Transformer, BERT, Basic GPT, Basic GPT with Mixture of
Experts, GPT with Unigram tokenizer, and WordPiece tokenizer. Each stage prints
its loss as it trains and a small inference demo at the end. A stage that throws
is reported and the next one still runs.

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
project with about 390 tests. There is one `*Tests.cpp` per library header, under
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
