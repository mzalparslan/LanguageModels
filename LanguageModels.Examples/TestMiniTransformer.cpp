
#include "MiniTransformer.h"
#include "BenchmarkTimer.h"
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <cctype>
#include <algorithm>

/**
 * @brief Minimal word-level tokenizer: every distinct whitespace-separated
 * word gets next free id, in order of first appearance.
 */
class SimpleTokenizer {
public:
    // Word -> id.
    std::map<std::string, std::size_t> wordToId;
    // Id -> word (the inverse of wordToId).
    std::map<std::size_t, std::string> idToWord;
    // Number of distinct words seen so far.
    std::size_t vocabSize = 0;

    /**
     * @brief Adds any words of sentence that are not in vocabulary yet.
     */
    void add(const std::string& sentence) {
        std::stringstream stream(sentence);
        std::string word;
        while (stream >> word) {
            if (wordToId.find(word) == wordToId.end()) {
                wordToId[word] = vocabSize;
                idToWord[vocabSize] = word;
                vocabSize++;
            }
        }
    }

    /**
     * @brief Converts a sentence to word ids; unknown words map to id 0.
     */
    std::vector<std::size_t> encode(const std::string& sentence) {
        std::stringstream stream(sentence);
        std::string word;
        std::vector<std::size_t> ids;
        while (stream >> word) {
            if (wordToId.find(word) != wordToId.end()) {
                ids.push_back(wordToId[word]);
            }
            else {
                ids.push_back(0); // UNK
            }
        }
        return ids;
    }

    /**
     * @brief Converts ids back to words joined by spaces; unknown ids are skipped.
     */
    std::string decode(const std::vector<std::size_t>& ids) {
        std::string text = "";
        for (std::size_t id : ids) {
            if (idToWord.count(id)) {
                text += idToWord[id] + " ";
            }
        }
        return text;
    }
};

/**
 * @brief Lowercases text and keeps only ASCII letters, spaces, apostrophes
 * and hyphens. Accented letters (multi-byte UTF-8) are dropped.
 */
std::string normalize(const std::string& text) {
    std::string normalized;
    for (char character : text) {
        // <cctype> functions are undefined for negative values, which UTF-8
        // continuation bytes (accented characters) are when stored in a char.
        unsigned char byteValue = static_cast<unsigned char>(character);
        if (std::isalpha(byteValue) || byteValue == ' ') {
            normalized += static_cast<char>(std::tolower(byteValue));
        }
        else if (byteValue == '\'' || byteValue == '-') {
            // Keep apostrophes and hyphens for now
            normalized += character;
        }
    }
    return normalized;
}

/**
 * @brief Loads tab-separated English/French sentence pairs (the Tatoeba
 * format: English, French, attribution).
 *
 * @param path File to read.
 * @param limit Maximum number of pairs to load.
 * @return (english, "<SOS> french <EOS>") pairs, normalized; empty if the
 * file cannot be opened. Lines that normalize to an empty side are skipped.
 */
std::vector<std::pair<std::string, std::string>> loadDataset(const std::string& path, std::size_t limit = 2000) {
    std::vector<std::pair<std::string, std::string>> data;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << path << ". Using synthetic data.\n";
        return {};
    }

    std::string line;
    while (std::getline(file, line) && data.size() < limit) {
        std::size_t firstTab = line.find('\t');
        if (firstTab == std::string::npos) {
            continue;
        }

        std::string english = normalize(line.substr(0, firstTab));
        std::string remainder = line.substr(firstTab + 1);
        std::size_t secondTab = remainder.find('\t'); // Remove attribution if present
        std::string french = normalize(remainder.substr(0, secondTab));

        if (english.empty() || french.empty()) {
            continue;
        }

        // Add SOS/EOS to target
        french = "<SOS> " + french + " <EOS>";
        data.push_back({ english, french });
    }
    std::cout << "Loaded " << data.size() << " pairs from " << path << "\n";
    return data;
}

// Set by main() when --quick is passed: use the small debug dataset.
extern bool useQuickDataset;

/**
 * @brief Demo stage: trains MiniTransformer on English -> French sentence
 * pairs, then translates held-out pairs with greedy decoding and with
 * beam search.
 *
 * held-out split is by position (the last 10%). data file is sorted
 * alphabetically, so most held-out source sentences never occur in training
 * and are translated poorly; this stage demonstrates training loop and
 * decoders, not generalization.
 *
 * Uses the full fra.txt unless the executable was started with --quick
 * (see useQuickDataset).
 *
 * @return 0 on completion, -1 if dataset could not be loaded.
 */
int testFraEngTranslation() {
    // 1. Dataset
    std::vector<std::pair<std::string, std::string>> data;

    // Try Data Load
    // By default the full fra.txt is used: up to 5000 pairs, which is a
    // ~1.5-2hr training run. Running the executable with --quick selects
    // fra_debug.txt (100 pairs) for fast iteration instead.
    const char* datasetPath = useQuickDataset ? "fra_debug.txt" : "fra.txt";
    data = loadDataset(datasetPath, 5000); // Load up to 5000 pairs

    // Fallback if empty
    if (data.empty()) {
        std::cerr << "ERR: Unable to load data!\n";
        return -1;
    }

    // 2. Tokenizers
    SimpleTokenizer encoderTokenizer, decoderTokenizer;
    for (auto& sentencePair : data) {
        encoderTokenizer.add(sentencePair.first);
        decoderTokenizer.add(sentencePair.second);
    }

    std::cout << "Src Vocab: " << encoderTokenizer.vocabSize << "\n";
    std::cout << "Tgt Vocab: " << decoderTokenizer.vocabSize << "\n";

    // 3. Model: vocabulary sizes set the embedding tables and the output width.
    // RoPE is always used for positions.
    MiniTransformer<double> model(encoderTokenizer.vocabSize, decoderTokenizer.vocabSize);
    double learningRate = 0.001; // Adam typically uses lower LR than SGD

    // 4. Split Data: first 90% for training, last 10% held out.
    std::size_t splitIdx = static_cast<std::size_t>(static_cast<double>(data.size()) * double(0.9));
    std::vector<std::pair<std::string, std::string>> trainData(data.begin(), data.begin() + splitIdx);
    std::vector<std::pair<std::string, std::string>> testData(data.begin() + splitIdx, data.end());

    std::cout << "Training on " << trainData.size() << " samples.\n";
    std::cout << "Testing on " << testData.size() << " samples.\n";

    // 5. Training Loop: one optimizer step per sentence pair (batch size 1).
    std::cout << "Training...\n";
    std::size_t adamStep = 1; // Adam timestep: 1-based, increased after every example

    // Small datasets need many more passes to be memorized than large ones.
    std::size_t maxEpochs = (data.size() < 100) ? 2000 : 500;

    BenchmarkTimer epochTimer;
    epochTimer.start();

    for (std::size_t epoch = 0; epoch < maxEpochs; epoch++) {
        double totalLoss = 0;
        for (auto& sentencePair : trainData) {
            auto src = encoderTokenizer.encode(sentencePair.first);
            auto tgtFull = decoderTokenizer.encode(sentencePair.second);

            if (tgtFull.size() < 2) {
                continue;
            }

            // Teacher forcing: decoder is fed target minus its last
            // token and must predict target minus its first token, i.e.
            // at every position it predicts next word.
            std::vector<std::size_t> decoderInput(tgtFull.begin(), tgtFull.end() - 1);
            std::vector<std::size_t> expectedIds(tgtFull.begin() + 1, tgtFull.end());

            totalLoss += model.trainStep(src, decoderInput, expectedIds, learningRate, UpdateRule::adam(adamStep));
            adamStep++;
        }

        // Log the average per-sentence loss every 100 epochs.
        if (epoch % 100 == 0) {
            std::cout << "100 x Epoch Elapsed time: " << epochTimer.stop() << " ms" << std::endl;
            epochTimer.start();

            std::cout << "Epoch " << epoch << " Loss: " << totalLoss / trainData.size() << "\n";
        }
    }
    // Ignore last batch's benchmark.
    (void)epochTimer.stop();
    
    // Greedy decoding: start from <SOS> and repeatedly append single
    // most likely next word until <EOS> or 10 words.
    std::cout << "\n--- Greedy Inference ---\n";
    for (auto& sentencePair : testData) {
        std::cout << "Src: " << sentencePair.first << " -> ";
        std::vector<std::size_t> src = encoderTokenizer.encode(sentencePair.first);
        std::vector<std::size_t> tgt = { (std::size_t)decoderTokenizer.wordToId["<SOS>"] };

        for (std::size_t decodeStep = 0; decodeStep < 10; ++decodeStep) {
            Tensor<double> logits;
            model.forward(src, tgt, logits);

            std::size_t seqLen = logits.shape[0];
            std::size_t outputVocabSize = logits.shape[1];
            std::size_t bestId = 0;
            // Arg-max over the last position: its scores rank every possible next word.
            double bestScore = -1e9;
            for (std::size_t wordId = 0; wordId < outputVocabSize; wordId++) {
                double score = logits.data[(seqLen - 1) * outputVocabSize + wordId];
                if (score > bestScore) {
                    bestScore = score;
                    bestId = wordId;
                }
            }

            if (decoderTokenizer.idToWord[bestId] == "<EOS>") {
                break;
            }
            tgt.push_back(bestId);
        }
        std::cout << decoderTokenizer.decode(tgt) << "\n";
    }

    // 6. Beam Search Inference: instead of committing to single best word
    // at each step, keep K best partial translations (ranked by summed log
    // probability) and extend all of them, which can find better full sentences.
    std::cout << "\n--- Beam Search (K=3) ---\n";
    class BeamCandidate {
    public:
        std::vector<std::size_t> sequence = {};
        double score = 0;
        bool finished = false;
    };

    // Number of hypotheses kept alive at every step.
    std::size_t beamWidth = 3;

    for (auto& sentencePair : testData) {
        std::cout << "Src: " << sentencePair.first << " -> ";
        std::vector<std::size_t> src = encoderTokenizer.encode(sentencePair.first);

        // Init Beam: a single hypothesis containing only <SOS>.
        std::vector<BeamCandidate> beams;
        beams.push_back({ {(std::size_t)decoderTokenizer.wordToId["<SOS>"]}, 0.0, false });

        for (int step = 0; step < 10; ++step) {
            std::vector<BeamCandidate> nextBeams;

            for (auto& beam : beams) {
                if (beam.finished) {
                    nextBeams.push_back(beam);
                    continue;
                }

                Tensor<double> logits;
                model.forward(src, beam.sequence, logits);

                std::size_t seqLen = logits.shape[0];
                std::size_t outputVocabSize = logits.shape[1];

                // Softmax on last token: turn its logits into log probabilities
                // (with max subtracted first for numerical stability).
                double maxLogit = -1e9;
                std::size_t lastRowStart = (seqLen - 1) * outputVocabSize;
                for (std::size_t wordId = 0; wordId < outputVocabSize; wordId++) {
                    maxLogit = std::max(maxLogit, logits.data[lastRowStart + wordId]);
                }

                double sum = 0;
                std::vector<double> logProbs(outputVocabSize);
                for (std::size_t wordId = 0; wordId < outputVocabSize; wordId++) {
                    double expValue = std::exp(logits.data[lastRowStart + wordId] - maxLogit);
                    sum += expValue;
                    logProbs[wordId] = expValue; // holds exp() until normalized below
                }

                for (std::size_t wordId = 0; wordId < outputVocabSize; wordId++) {
                    logProbs[wordId] = std::log(logProbs[wordId] / sum);
                }

                // Expand: extend this hypothesis with every possible next word.
                // (The pruning below keeps only the beamWidth best overall.)
                for (std::size_t wordId = 0; wordId < outputVocabSize; wordId++) {
                    BeamCandidate newBeam = beam;
                    newBeam.sequence.push_back(wordId);
                    newBeam.score += logProbs[wordId];
                    if (decoderTokenizer.idToWord[wordId] == "<EOS>") {
                        newBeam.finished = true;
                    }
                    nextBeams.push_back(newBeam);
                }
            }

            // Prune: keep only the beamWidth highest-scoring hypotheses.
            std::sort(nextBeams.begin(), nextBeams.end(),
                [](const BeamCandidate& left, const BeamCandidate& right) {
                return left.score > right.score; // Descending
                });

            if (nextBeams.size() > beamWidth) {
                nextBeams.resize(beamWidth);
            }
            beams = nextBeams;

            // Stop early once every kept hypothesis has produced <EOS>.
            bool allFinished = true;
            for (auto& beam : beams) {
                if (!beam.finished) {
                    allFinished = false;
                }
            }
            if (allFinished) {
                break;
            }
        }

        if (!beams.empty()) {
            std::cout << decoderTokenizer.decode(beams[0].sequence) << " (score: " << beams[0].score << ")\n";
        }
    }
    
    return 0;
}
