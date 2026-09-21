#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include "Validation.h"

/**
 * @brief Unigram Language Model tokenizer (the algorithm behind SentencePiece).
 *
 * Every token in vocabulary has a probability; a sentence is split into
 * sequence of tokens whose probabilities multiply to highest value
 * (found with Viterbi algorithm). Training starts from a large seed
 * vocabulary and alternates between segmenting text with current
 * probabilities (E-step) and re-estimating probabilities from resulting
 * token counts (M-step), pruning down to a target vocabulary size.
 *
 * This is a simplified version: seed is single characters plus
 * whitespace-separated words rather than all substrings.
 */
class UnigramTokenizer {
public:
    /**
     * @brief Per-token data.
     */
    class TokenInfo {
    public:
        double score; ///< log probability
        int id;
    };

    std::unordered_map<std::string, TokenInfo> vocab; ///< token -> info
    // Id -> token string (the inverse of vocab).
    std::vector<std::string> idToToken;

    /**
     * @brief Config
     */
    // Vocabulary size that training prunes down to.
    std::size_t vocabSizeTarget;

    /**
     * @param targetVocab Desired vocabulary size after training.
     * @throws InvalidParameterSizeError If targetVocab is zero.
     */
    UnigramTokenizer(std::size_t targetVocab = 1000) : vocabSizeTarget(targetVocab) {
        validation::requirePositiveSize(targetVocab, "Unigram target vocabulary size");
    }

    /**
     * @brief Rebuilds vocabulary from token counts: every token's score
     * becomes log(count / total), and ids are assigned in order of decreasing
     * count so they are deterministic. Replaces current vocabulary.
     *
     * @param counts Occurrence count of each token.
     * @throws InvalidSizeError If counts is empty.
     * @throws DivisionByZeroError If counts sum to zero.
     */
    void updateScores(const std::unordered_map<std::string, std::size_t>& counts) {
        std::size_t totalCount = 0;
        for (auto& p : counts) {
            totalCount += p.second;
        }

        // log(0) is -infinity and every score below divides by total.
        validation::requireNonEmpty(counts.size(), "Token counts");
        validation::requireNonZeroDenominator(totalCount, "total token count");

        double logTotal = std::log((double)totalCount);

        vocab.clear();
        idToToken.clear();

        // Sort by count (descending) so IDs are deterministic and most
        // frequent token gets id 0.
        std::vector<std::pair<std::size_t, std::string>> sortedTokens;
        for (auto& p : counts) {
            sortedTokens.push_back({ p.second, p.first });
        }
        std::sort(sortedTokens.rbegin(), sortedTokens.rend());

        int idCounter = 0;
        for (auto& p : sortedTokens) {
            TokenInfo info;
            info.score = std::log((double)p.first) - logTotal;
            info.id = idCounter++;
            vocab[p.second] = info;
            idToToken.push_back(p.second);
        }
    }
    /**
     * @brief Viterbi segmentation: finds split of text into vocabulary
     * tokens with highest total log probability, by dynamic programming
     * over prefixes.
     *
     * @param text String to segment (one word during training).
     * @return chosen tokens in order. Stops early, returning a partial
     * result, if some part of text is not covered by any token.
     */
    std::vector<std::string> viterbiEncode(const std::string& text) {
        std::size_t n = text.size();
        if (n == 0) {
            return {};
        }

        // dp[i] = max log_prob to cover text[0...i-1]
        std::vector<double> dp(n + 1, -1e18); // -infinity
        std::vector<int> startIndex(n + 1, -1); // Backpointer

        dp[0] = 0;

        for (std::size_t i = 1; i <= n; i++) {
            // Try every substring text[j...i-1] that is in vocab and keep the
            // best way to reach position i (all starts j are checked; token
            // length is not capped).
            for (std::size_t j = 0; j < i; j++) {
                // subword text[j...i-1]
                std::string sub = text.substr(j, i - j);
                if (vocab.count(sub)) {
                    double score = vocab[sub].score;
                    if (dp[j] + score > dp[i]) {
                        dp[i] = dp[j] + score;
                        startIndex[i] = (int)j;
                    }
                }
            }

            // If no token reaches position i, dp[i] stays at -infinity. The
            // vocabulary is seeded with every single character, so this only
            // happens for characters unseen in training; nothing recovers it.
            if (dp[i] <= -1e17) {
            }
        }

        // Reconstruct: follow backpointers from end of text to the
        // start, then reverse into reading order.
        std::vector<std::string> result;
        int curr = (int)n;
        while (curr > 0) {
            int start = startIndex[curr];
            if (start == -1) {
                break; // Error
            }
            result.push_back(text.substr(start, curr - start));
            curr = start;
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    /**
     * @brief Trains vocabulary with Expectation-Maximization.
     *
     * @param text Training corpus.
     * @param maxIterations Maximum number of EM rounds; stops earlier once
     * vocabulary has reached vocabSizeTarget (after at least 3 rounds).
     * @throws InvalidSizeError If text is empty.
     * @throws InvalidParameterError If maxIterations <= 0.
     */
    void train(const std::string& text, int maxIterations = 10) {
        validation::requireNonEmpty(text.size(), "Training text");
        if (maxIterations <= 0) {
            throw InvalidParameterError(
                "Unigram maxIterations must be greater than zero!");
        }

        // 1. Initialization (Seed Vocab)
        // Collect all single chars (kept in seed so any text can be encoded)
        std::map<char, std::size_t> charCounts;
        for (char c : text) {
            charCounts[c]++;
        }

        // Collect frequence candidates (words split by space)
        // A real Unigram implementation considers ALL substrings, which is huge.
        // We will seed with Chars + Space-separated words.
        std::map<std::string, std::size_t> seedCounts;

        std::string currentWord;
        for (char c : text) {
            seedCounts[std::string(1, c)]++; // Always add chars
            if (std::isspace(c)) {
                if (false == currentWord.empty()) {
                    seedCounts[currentWord]++;
                }
                currentWord.clear();
            }
            else {
                currentWord += c;
            }
        }

        if (false == currentWord.empty()) {
            seedCounts[currentWord]++;
        }

        // Initial Score Update: turn seed counts into log probabilities.
        std::unordered_map<std::string, std::size_t> currentCounts;
        for (auto& p : seedCounts) {
            currentCounts[p.first] = p.second;
        }
        updateScores(currentCounts);

        std::cout << "Tokenizer Initialized. Vocab size: " << vocab.size() << std::endl;

        // 2. EM Loop
        for (int iter = 0; iter < maxIterations; iter++) {
            std::unordered_map<std::string, std::size_t> expectedCounts;

            // E-Step: Viterbi on text to find optimal segmentation.
            // Assumes `text` is manageable/sampled (Viterbi on massive text
            // is slow), and processes word-by-word (space-delimited) rather
            // than whole line at once, treating whitespace as a literal
            // separator token rather than folding it into words (unlike
            // SentencePiece's `_` convention) so its counts aren't lost.
            std::string wAccum;
            for (std::size_t i = 0; i < text.size(); i++) {
                if (std::isspace(text[i])) {
                    if (false == wAccum.empty()) {
                        std::vector<std::string> tokens = viterbiEncode(wAccum);
                        for (const auto& t : tokens) {
                            expectedCounts[t]++;
                        }
                    }
                    wAccum.clear();
                    std::string spaceToken(1, text[i]);
                    if (vocab.count(spaceToken)) {
                        expectedCounts[spaceToken]++;
                    }
                }
                else {
                    wAccum += text[i];
                }
            }
            if (!wAccum.empty()) {
                std::vector<std::string> tokens = viterbiEncode(wAccum);
                for (const auto& t : tokens) {
                    expectedCounts[t]++;
                }
            }

            // M-Step: re-estimate every token's probability from counts
            // Viterbi segmentation produced.
            updateScores(expectedCounts);

            // Pruning: updateScores() keeps every token that was counted, so
            // if vocabulary grew past vocabSizeTarget, keep only most
            // frequent vocabSizeTarget tokens.
            if (vocab.size() > vocabSizeTarget) {
                // Keep top target. updateScores() sorts idToToken by
                // count descending, so idToToken[0] is most frequent.
                auto copyIdToToken = idToToken;

                std::unordered_map<std::string, std::size_t> prunedCounts;
                for (std::size_t i = 0; i < vocabSizeTarget; i++) {
                    prunedCounts[idToToken[i]] = expectedCounts[idToToken[i]];
                }
                updateScores(prunedCounts); // re-calc log probs
            }

            std::cout << "EM Iteration " << iter + 1 << " finished. Vocab: " << vocab.size() << std::endl;
            if (vocab.size() <= vocabSizeTarget && iter > 2) {
                break; // Converged enough
            }
        }
    }

    /**
     * @brief Converts text to token ids by segmenting whole string with
     * Viterbi. Whitespace is handled as an ordinary character, so it must be
     * in vocabulary (it is, when training text contained spaces).
     *
     * @param text Text to encode.
     * @return Token ids.
     * @throws InvalidParameterError If text contains a character vocabulary does not cover.
     */
    std::vector<int> encode(const std::string& text) {
        std::vector<std::string> tokens = viterbiEncode(text);

        // viterbiEncode() stops early at a character no token covers, which
        // would silently drop rest of text.
        std::size_t coveredLength = 0;
        for (const auto& t : tokens) {
            coveredLength += t.size();
        }
        if (coveredLength != text.size()) {
            throw InvalidParameterError("Text contains a character that is not in vocabulary!");
        }

        std::vector<int> ids;
        for (const auto& t : tokens) {
            ids.push_back(vocab[t].id);
        }
        return ids;
    }

    /**
     * @brief Converts token ids back to text by concatenating tokens.
     *
     * @param ids Token ids.
     * @return decoded string.
     * @throws InvalidParameterError If an id is negative or outside vocabulary.
     */
    std::string decode(const std::vector<int>& ids) {
        std::string res;
        for (int id : ids) {
            if (id < 0) {
                throw InvalidParameterError("Token id must not be negative!");
            }
            validation::requireBelow(static_cast<std::size_t>(id), idToToken.size(), "Token id");
            res += idToToken[id];
        }
        return res;
    }
};