#include <iostream>
#include <exception>
#include <string>

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

// When true (--quick), the Mini Transformer stage trains on the 100-pair
// fra_debug.txt instead of the full fra.txt (a ~1.5-2hr run), and the Tiny
// Shakespeare benchmark trains for fewer steps and scores less text.
bool useQuickDataset = true;

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
        else {
            std::cout << "Usage: LanguageModels.Examples [--quick]\n"
                << "  --quick  train the Mini Transformer on the 100-pair debug dataset\n"
                << "           (fra_debug.txt) instead of the full fra.txt, and shorten the\n"
                << "           Tiny Shakespeare benchmark\n";
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

    return 0;
}
