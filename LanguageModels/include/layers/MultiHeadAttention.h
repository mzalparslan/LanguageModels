#pragma once

#include "LinearLayer.h"
#include "AttentionHead.h"

/**
 * @brief Multi-head self-attention: runs numHeads independent AttentionHead
 * instances on dHead-wide slices in parallel, then concatenates and
 * projects result back to dModel through Wo.
 */
template <typename T>
class MultiHeadAttention {
public:
    // Model width: input and output vector size.
    std::size_t dModel;
    // Number of parallel heads.
    std::size_t numHeads;
    // Width each head works in: dModel / numHeads.
    std::size_t dHead;

    // One AttentionHead per head; each reads full dModel-wide input and
    // projects it to its own dHead-wide Q/K/V.
    std::vector<AttentionHead<T>> heads;
    // Output projection applied to concatenated head outputs.
    LinearLayer<T> Wo;

    /**
     * @param dm Model width; must be divisible by nHeads.
     * @param nHeads Number of attention heads.
     * @param rng Engine initial weights are drawn from.
     * @throws InvalidParameterSizeError If dm or nHeads is zero, or dm is not
     * divisible by nHeads.
     */
    MultiHeadAttention(std::size_t dm, std::size_t nHeads, RandomEngine& rng) : dModel(dm), numHeads(nHeads), dHead(checkedHeadWidth(dm, nHeads)), Wo(dm, dm, rng) {
        for (std::size_t i = 0; i < nHeads; i++) {
            heads.emplace_back(dm, dHead, rng);
        }
    }

    /**
     * @brief Clears accumulated gradients of every head and of Wo.
     */
    void zeroGrad() {
        for (auto& head : heads) {
            head.zeroGrad();
        }
        Wo.zeroGrad();
    }

    /**
     * @brief Runs every head on same inputs, concatenates their outputs
     * side by side, and mixes them with Wo.
     *
     * @param inputQ Query source [seq, dModel].
     * @param inputK Key source [seq, dModel].
     * @param inputV Value source [seq, dModel].
     * @param out Result [seq, dModel].
     * @param mask If true, applies causal mask in every head.
     * @param rope Optional rotary position embedding applied inside each head.
     *
     * @remark `rope`, if supplied, must be sized for dHead (per-head width),
     * not dModel -- each AttentionHead only ever sees a dHead-wide slice of
     * Q/K. Passing a dModel-sized RotaryEmbedding here will read/write past
     * each head's actual row width.
     */
    template <typename RopeConfig = ModelConfig>
    void forward(const Tensor<T>& inputQ, const Tensor<T>& inputK, const Tensor<T>& inputV, Tensor<T>& out, bool mask = false, RotaryEmbedding<T, RopeConfig>* rope = nullptr) {
        validation::requireColumns(inputQ, dModel, "MultiHeadAttention query input");
        std::size_t seq = inputQ.shape[0];

        concatOut = Tensor<T>({ seq, dModel });

        for (std::size_t h = 0; h < numHeads; ++h) {
            Tensor<T> headOut;
            heads[h].forward(inputQ, inputK, inputV, headOut, rope, mask);

            // Concat: head h's dHead-wide output goes into columns
            // [h * dHead, (h + 1) * dHead) of each row.
            for (std::size_t i = 0; i < seq; i++) {
                for (std::size_t j = 0; j < dHead; j++) {
                    concatOut.data[i * dModel + h * dHead + j] = headOut.data[i * dHead + j];
                }
            }
        }

        // Mix concatenated heads back to dModel.
        Wo.forward(concatOut, out);
    }

    /**
     * @brief Backward pass: splits gradient across heads, runs each
     * head's backward, and sums their input gradients.
     *
     * @param dout Gradient w.r.t. forward()'s output [seq, dModel].
     * @param dQ Gradient w.r.t. inputQ [seq, dModel].
     * @param dK Gradient w.r.t. inputK [seq, dModel].
     * @param dV Gradient w.r.t. inputV [seq, dModel].
     */
    void backward(const Tensor<T>& dout, Tensor<T>& dQ, Tensor<T>& dK, Tensor<T>& dV) {
        // Through output projection: gives gradient of the
        // concatenated head outputs.
        Tensor<T> dConcat;
        Wo.backward(dout, dConcat);

        std::size_t seq = dConcat.shape[0];
        dQ = Tensor<T>({ seq, dModel }, 0);
        dK = Tensor<T>({ seq, dModel }, 0);
        dV = Tensor<T>({ seq, dModel }, 0);

        for (std::size_t h = 0; h < numHeads; ++h) {
            // Slice dConcat for this head (the inverse of forward()'s concat).
            Tensor<T> dHeadOut({ seq, dHead });
            for (std::size_t i = 0; i < seq; i++) {
                for (std::size_t j = 0; j < dHead; j++) {
                    dHeadOut.data[i * dHead + j] = dConcat.data[i * dModel + h * dHead + j];
                }
            }

            Tensor<T> dqH, dkH, dvH;
            heads[h].backward(dHeadOut, dqH, dkH, dvH);

            // All heads read from same Q, K, V (the same input_x), so
            // gradients from every head accumulate onto same dQ/dK/dV.
            for (std::size_t i = 0; i < dqH.size(); i++) {
                dQ[i] += dqH[i];
            }
            for (std::size_t i = 0; i < dkH.size(); i++) {
                dK[i] += dkH[i];
            }
            for (std::size_t i = 0; i < dvH.size(); i++) {
                dV[i] += dvH[i];
            }
        }
    }

    /**
     * @brief Applies accumulated gradients to every head and Wo.
     *
     * @param lr Learning rate.
     * @param rule Optimizer to use; see UpdateRule.
     */
    void update(T lr, UpdateRule rule = UpdateRule::sgd()) {
        for (auto& head : heads) {
            head.update(lr, rule);
        }
        Wo.update(lr, rule);
    }

private:
    /**
     * @brief Concatenated head outputs [seq, dModel] from last
     * forward() call, needed as Wo's input during backward.
     */
    Tensor<T> concatOut;

    /**
     * @brief Validates head configuration and returns per-head width.
     * It runs while members are being built, before anything divides by
     * head count.
     */
    static std::size_t checkedHeadWidth(std::size_t dm, std::size_t nHeads) {
        validation::requirePositiveSize(dm, "MultiHeadAttention model width");
        validation::requirePositiveSize(nHeads, "MultiHeadAttention head count");
        validation::requireDivisible(dm, nHeads, "Model width", "head count");
        return dm / nHeads;
    }
};
