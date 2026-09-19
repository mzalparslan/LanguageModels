#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <utility>
#include <vector>

#include "DecoderBlockConcept.h"
#include "Embedding.h"
#include "RotaryEmbedding.h"
#include "LinearLayer.h"
#include "RMSNorm.h"
#include "ModelConfig.h"

/**
 * @brief Generic decoder-only (GPT-style) language model: embeds tokens,
 * stacks numLayers decoder blocks of type BlockT, then projects back to
 * vocabulary logits through a final norm + linear head.
 *
 * See models/BasicGPT.h and models/BasicGPTWithMoE.h for the thin named
 * aliases built on top of this, each just fixing BlockT and Config.
 *
 * @tparam T Floating-point mode.
 * @tparam BlockT Decoder block type to stack; must satisfy DecoderBlock
 * (see layers/DecoderBlockConcept.h) and be constructible as
 * BlockT(dim, dim * 4, blockArgs...). Blocks that also satisfy HasAuxLoss
 * contribute their MoE auxiliary loss to trainStep().
 * @tparam Config Compile-time config supplying RotaryEmbedding's sizing
 * (Config::d_head, Config::maxSeqLen) and weight-initialization seed
 * (Config::randomSeed) -- sizes must match how this model is
 * actually constructed (see config/BasicGPTConfig.h for why).
 */
template <typename T, typename BlockT, typename Config = ModelConfig>
    requires DecoderBlock<BlockT, T, Config>
class DecoderOnlyModel {
public:
    // Model width.
    std::size_t dModel;
    // Number of token ids model knows.
    std::size_t vocabSize;
    // Number of stacked decoder blocks.
    std::size_t numLayers;
    // Context length model was built for.
    std::size_t maxLen;
    // decoder stack, applied in order.
    std::vector<BlockT> layers;

    /**
     * @brief Aggregated MoE load-balancing auxiliary loss from most
     * recent trainStep(). Stays 0 for block types without getAuxLoss()
     * (e.g. BasicDecoderBlock).
     */
    T lastAuxLoss = T(0);

    /**
     * @brief Builds model; initial weights are drawn from an engine seeded
     * with Config::randomSeed, so same Config always gives same model.
     *
     * @param blockArgs Forwarded to each layer's constructor after
     * (dim, dim * 4, rng) -- e.g. (useMoe, numExperts, topK[, numHeads]),
     * matching whichever BlockT this model was instantiated with.
     * @throws InvalidParameterSizeError If the vocabulary, width, layer count or
     * context length is zero, or the context length exceeds Config::maxSeqLen.
     */
    template <typename... BlockArgs>
        requires std::constructible_from<BlockT, std::size_t, std::size_t, RandomEngine&, BlockArgs&...>
    DecoderOnlyModel(std::size_t vocab, std::size_t dim, std::size_t nLayers, std::size_t contextLen, BlockArgs&&... blockArgs)
        : DecoderOnlyModel(RandomEngine(Config::randomSeed), vocab, dim, nLayers, contextLen,
            std::forward<BlockArgs>(blockArgs)...) {}

    /**
     * @brief Forward pass: embeds tokens, runs decoder stack, and
     * projects to vocabulary logits.
     *
     * @param x Token ids [seq].
     * @param logits Output [seq, vocabSize]; row i scores token that
     * follows position i.
     * @throws InvalidSizeError If x is empty or longer than the context length.
     * @throws InvalidParameterError If a token id is outside the vocabulary.
     */
    void forward(const std::vector<std::size_t>& x, Tensor<T>& logits) {
        validation::requireNonEmpty(x.size(), "Input token sequence");
        validation::requireAtMost(x.size(), maxLen, "Input sequence length");

        // 1. Embed
        tokenEmb.forward(x, embOut);

        // 2. Layers
        Tensor<T> curr = embOut;
        for (auto& layer : layers) {
            Tensor<T> next;
            layer.forward(curr, next, &rope);
            curr = next;
        }

        // 3. Final Norm
        Tensor<T> normCurr;
        finalNorm.forward(curr, normCurr);

        // 4. Logits
        lmHead.forward(normCurr, logits);
    }

    /**
     * @brief One full training step (plain SGD): zero gradients, forward,
     * cross-entropy loss, backward through whole model, update.
     *
     * @param x Input token ids [seq].
     * @param targets Expected next-token id for each position [seq].
     * @param lr Learning rate.
     * @return Mean cross-entropy over sequence, plus MoE auxiliary
     * loss when block type tracks one.
     * @throws InvalidParameterError If lr <= 0, or a target id is outside the vocabulary.
     * @throws InvalidSizeError If targets and x differ in length.
     * @throws NaNError, NonFiniteError If the loss is not finite.
     */
    T trainStep(const std::vector<std::size_t>& x, const std::vector<std::size_t>& targets, T lr) {
        validation::requirePositiveFinite(lr, "Learning rate");
        validation::requireSameSize(targets.size(), x.size(), "Target sequence");

        // Zero Grad: backward passes accumulate, so start clean.
        tokenEmb.zeroGrad();
        for (auto& layer : layers) {
            layer.zeroGrad();
        }
        finalNorm.zeroGrad();
        lmHead.zeroGrad();

        Tensor<T> logits;
        forward(x, logits);

        // Loss: Cross Entropy
        std::size_t seq = logits.shape[0];
        std::size_t vocab = logits.shape[1];
        Tensor<T> dlogits(logits.shape, 0);
        T loss = 0;

        for (std::size_t i = 0; i < seq; i++) {
            // Softmax
            T maxV = T(-1e9);
            for (std::size_t j = 0; j < vocab; j++) {
                maxV = std::max(maxV, logits.data[i * vocab + j]);
            }
            T sum = 0;
            std::vector<T> probs(vocab);
            for (std::size_t j = 0; j < vocab; j++) {
                probs[j] = std::exp(logits.data[i * vocab + j] - maxV);
                sum += probs[j];
            }
            validation::requireFinite(sum, "Softmax normalizer");
            validation::requireNonZeroDenominator(sum, "softmax normalizer");
            for (std::size_t j = 0; j < vocab; j++) {
                probs[j] /= sum;
            }

            // Log Loss
            std::size_t y = targets[i];
            validation::requireBelow(y, vocab, "Target token id");
            loss -= std::log(probs[y]);

            // Grad: for softmax + cross-entropy, dLoss/dlogits = probs - one_hot(y).
            for (std::size_t j = 0; j < vocab; j++) {
                dlogits.data[i * vocab + j] = probs[j];
            }
            dlogits.data[i * vocab + y] -= T(1);
        }

        // A NaN, or an infinite loss (a target with probability 0), would poison
        // every gradient below, so stop before backpropagating.
        validation::requireFinite(loss, "Training loss");

        // Backward: output head, final norm, then blocks from last to first.
        Tensor<T> dNormOut;
        lmHead.backward(dlogits, dNormOut);

        Tensor<T> dFinalLayerOut;
        finalNorm.backward(dNormOut, dFinalLayerOut);

        Tensor<T> currDout = dFinalLayerOut;
        for (std::size_t i = numLayers; i-- > 0; ) {
            Tensor<T> nextDout;
            layers[i].backward(currDout, nextDout);
            currDout = nextDout;
        }

        // Route gradient reaching input into embedding rows used.
        tokenEmb.backward(x, currDout);

        // Update
        tokenEmb.update(lr);
        for (auto& layer : layers) {
            layer.update(lr);
        }
        finalNorm.update(lr);
        lmHead.update(lr);

        // Aggregate MoE aux loss, if this BlockT tracks one.
        T totalAux = 0;
        if constexpr (HasAuxLoss<BlockT, T>) {
            for (auto& layer : layers) {
                totalAux += layer.getAuxLoss();
            }
        }
        lastAuxLoss = totalAux;

        validation::requireNonZeroDenominator(seq, "Sequence length");
        validation::requireFinite(totalAux, "MoE auxiliary loss");
        return (loss / seq) + totalAux;
    }

    /**
     * @brief Greedy autoregressive generation: repeatedly runs model and
     * appends highest-scoring next token.
     *
     * @param startTokens Prompt token ids; must not be empty.
     * @param maxNewTokens Number of tokens to append.
     * @return prompt followed by generated tokens.
     * @throws InvalidSizeError If the prompt is empty, or the prompt plus the
     * generated tokens would exceed the context length.
     */
    std::vector<std::size_t> generate(const std::vector<std::size_t>& startTokens, std::size_t maxNewTokens) {
        validation::requireNonEmpty(startTokens.size(), "Prompt");
        // The last forward pass sees the prompt plus all but the final new token.
        if (maxNewTokens > 0) {
            validation::requireAtMost(startTokens.size() + maxNewTokens - 1, maxLen,
                "Prompt length plus generated tokens");
        }

        std::vector<std::size_t> curr = startTokens;
        for (std::size_t i = 0; i < maxNewTokens; i++) {
            Tensor<T> logits;
            forward(curr, logits);

            // Get last token logits
            std::size_t seq = logits.shape[0];
            std::size_t vocab = logits.shape[1];
            std::size_t lastRowIdx = (seq - 1) * vocab;

            // Greedy: take arg-max token of last position (no sampling).
            std::size_t nextToken = 0;
            T maxVal = T(-1e9);
            for (std::size_t j = 0; j < vocab; j++) {
                if (logits.data[lastRowIdx + j] > maxVal) {
                    maxVal = logits.data[lastRowIdx + j];
                    nextToken = j;
                }
            }

            curr.push_back(nextToken);
        }
        return curr;
    }

private:
    // Token id -> vector lookup.
    Embedding<T> tokenEmb;
    // Rotary position encoding shared by every block's attention.
    RotaryEmbedding<T, Config> rope;
    // Final normalization before output projection.
    RMSNorm<T> finalNorm;
    // Output projection: model width -> vocabulary logits.
    LinearLayer<T> lmHead;

    /**
     * @brief Token embeddings of last forward() call.
     */
    Tensor<T> embOut;

    /**
     * @brief Does actual construction with engine created by the
     * public constructor, so every layer draws from one engine.
     */
    template <typename... BlockArgs>
        requires std::constructible_from<BlockT, std::size_t, std::size_t, RandomEngine&, BlockArgs&...>
    DecoderOnlyModel(RandomEngine rng, std::size_t vocab, std::size_t dim, std::size_t nLayers, std::size_t contextLen, BlockArgs&&... blockArgs)
        : dModel(dim), vocabSize(vocab), numLayers(nLayers), maxLen(contextLen),
        tokenEmb(vocab, dim, rng), finalNorm(dim), lmHead(dim, vocab, rng)
    {
        validation::requirePositiveSize(nLayers, "Layer count");
        validation::requirePositiveSize(contextLen, "Context length");
        // Rotary position tables only cover Config::maxSeqLen positions.
        if (contextLen > Config::maxSeqLen) {
            throw InvalidParameterSizeError("Context length exceeds Config::maxSeqLen!");
        }

        layers.reserve(nLayers);
        for (std::size_t i = 0; i < nLayers; i++) {
            layers.emplace_back(dim, dim * 4, rng, blockArgs...);
        }
    }
};
