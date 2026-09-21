#include <iostream>
#include <exception>
#include <string>

#include "ExecutionStrategy.h"

// Each course stage lives in its own translation unit and exposes a single
// entry point (testXxx). This file is one real main() for whole
// project, running every stage in order course introduces them.
int testVanilla();
int testSimpleSet();
int testFraEngTranslation();
int testBert();
int testBasicGPT();
int testBasicGPTWithMoE();
int testGPTWithUnigram();
int testWordPieceTokenizer();
int testTinyShakespeare();
int testTranslationPipeline();
int testRnnPipeline();
int testGptPipeline();
int testBertPipeline();

// When true (--quick), Mini Transformer stage trains on 100-pair
// fra_debug.txt instead of full fra.txt (a ~1.5-2hr run), and Tiny
// Shakespeare benchmark trains for fewer steps and scores less text.
bool useQuickDataset = false;

// How pipeline stages train (--parallel): Sequential, or Parallel, which
// runs Mini Transformer's training steps on several threads. other
// models train same way either way.
ExecutionStrategy executionStrategy = ExecutionStrategy::Sequential;

namespace {
// A stage throwing (or hitting an access violation) shouldn't take out
// every later stage's demo output with it.
void runStage(const char* banner, int (*stage)()) {
    std::cout << "\n===== " << banner << " =====" << std::endl;
    try {
        stage();
    } catch (const std::exception& e) {
        std::cout << "[" << banner << "] threw std::exception: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "[" << banner << "] threw an unrecognized exception" << std::endl;
    }
}
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string argument = argv[i];
        if (argument == "--quick") {
            useQuickDataset = true;
        }
        else if (argument == "--parallel") {
            executionStrategy = ExecutionStrategy::Parallel;
        }
        else {
            std::cout << "Usage: LanguageModels.Examples [--quick] [--parallel]\n"
                << "  --quick     train Mini Transformer on 100-pair debug dataset\n"
                << "              (fra_debug.txt) instead of full fra.txt, and shorten the\n"
                << "              Tiny Shakespeare benchmark and pipeline stages\n"
                << "  --parallel  run pipeline stages with ExecutionStrategy::Parallel\n"
                << "              (the Mini Transformer trains on several threads)\n";
            return argument == "--help" ? 0 : 1;
        }
    }
    
    runStage("Vanilla RNN", testVanilla);
    runStage("Simple Transformer", testSimpleSet);
    runStage("Mini Transformer", testFraEngTranslation);
    runStage("BERT", testBert);
    runStage("Basic GPT", testBasicGPT);
    runStage("Basic GPT with Mixture of Experts", testBasicGPTWithMoE);
    runStage("GPT with Unigram Tokenizer", testGPTWithUnigram);
    runStage("WordPiece Tokenizer", testWordPieceTokenizer);
    runStage("Tiny Shakespeare Benchmark", testTinyShakespeare);
    runStage("Translation Pipeline (Mini Transformer)", testTranslationPipeline);
    runStage("Character RNN Pipeline", testRnnPipeline);
    runStage("Character GPT Pipeline", testGptPipeline);
    runStage("BERT Pipeline", testBertPipeline);

    return 0;
}
