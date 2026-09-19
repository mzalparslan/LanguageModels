#pragma once

#include "LinearLayer.h"
#include "FeedForward.h"

/**
 * @brief Mixture of Experts (MoE) layer: a drop-in replacement for a
 * feed-forward sublayer that holds several FeedForward "experts" and a small
 * router (gate). For every token router scores all experts, top-k are
 * run, and their outputs are blended by (renormalized) router weights.
 * Parameters grow with expert count, but each token only pays for k
 * experts.
 *
 * @remark Stateful: forward() caches routing decisions and input, and
 * backward() consumes them, so call backward() right after forward() it
 * belongs to.
 */
template <typename T>
class MoELayer {
private:
    LinearLayer<T> gate; ///< Router: [dModel, numExperts]

    // Cache for backward
    Tensor<T> gateLogits;
    Tensor<T> inputCache;

public:
    // Number of experts to choose from.
    std::size_t numExperts;
    // Number of experts each token is routed to.
    std::size_t topK;
    // Model width: input and output vector size.
    std::size_t dModel;
    std::vector<FeedForward<T>> experts;
    //< [Seq] -> list of topK expert indices
    std::vector<std::vector<int>> selectedIndices;
    //< [Seq] -> list of topK weights
    std::vector<std::vector<T>> selectedWeights;

    /**
     * @brief Load-balancing auxiliary loss from most recent forward()
     * call (see ModelConfig::moeLoadBalancingAlpha for intended
     * formula). Not yet computed -- forward() leaves this at 0, same as
     * gate.backward() being skipped below; router doesn't currently
     * learn from a real signal, only experts do.
     */
    T currAuxLoss = T(0);

    /**
     * @param dm Model width.
     * @param dFf Inner width of each expert's feed-forward network.
     * @param rng Engine initial weights are drawn from.
     * @param nExp Number of experts.
     * @param k Experts per token (must not exceed nExp).
     * @throws InvalidParameterSizeError If a size is zero or k > nExp.
     */
    MoELayer(std::size_t dm, std::size_t dFf, RandomEngine& rng, std::size_t nExp = 4, std::size_t k = 2)
        : dModel(dm), numExperts(nExp), topK(k), gate(dm, nExp, rng)
    {
        validation::requirePositiveSize(nExp, "MoELayer expert count");
        validation::requirePositiveSize(k, "MoELayer experts per token");
        if (k > nExp) {
            throw InvalidParameterSizeError("MoELayer experts per token must not exceed the expert count!");
        }

        for (std::size_t i = 0; i < nExp; i++) {
            experts.emplace_back(dm, dFf, rng);
        }
    }

    /**
     * @brief Clears accumulated gradients of router and all experts.
     */
    void zeroGrad() {
        gate.zeroGrad();
        for (auto& expert : experts) {
            expert.zeroGrad();
        }
    }

    /**
     * @brief Routes every token to its top-k experts and blends their outputs.
     *
     * @param x Input [seq, dModel].
     * @param out Output [seq, dModel].
     */
    void forward(const Tensor<T>& x, Tensor<T>& out) {
        validation::requireColumns(x, dModel, "MoELayer input");
        inputCache = x;
        std::size_t seq = x.shape[0];
        out = Tensor<T>(x.shape, 0);

        // Router: one score per expert for each token. [seq, numExperts]
        gate.forward(x, gateLogits);

        // Softmax over experts
        Tensor<T> probs = gateLogits; // copy
        softmaxRow(probs);

        selectedIndices.assign(seq, std::vector<int>(topK));
        selectedWeights.assign(seq, std::vector<T>(topK));

        for (std::size_t i = 0; i < seq; i++) {
            // Find Top-K: rank this token's experts by router probability
            // and keep k best.
            std::vector<std::pair<T, int>> candidates;
            for (std::size_t j = 0; j < numExperts; j++) {
                candidates.push_back({ probs.data[i * numExperts + j], (int)j });
            }
            std::sort(candidates.rbegin(), candidates.rend()); // descending

            // Normalize Top-K weights (optional but common) or just use raw softmax probs
            // Standard approach: use softmax prob directly

            T rowTotalWeight = 0;
            for (std::size_t k = 0; k < topK; k++) {
                selectedIndices[i][k] = candidates[k].second;
                selectedWeights[i][k] = candidates[k].first;
                rowTotalWeight += candidates[k].first;
            }
            validation::requireNonZeroDenominator(rowTotalWeight, "MoE selected expert weight total");
            // Renormalize weights to sum to 1 among selected?
            // GPT-4/GShard often re-normalizes. Let's do it.
            for (std::size_t k = 0; k < topK; k++) {
                selectedWeights[i][k] /= rowTotalWeight;
            }

            // Compute output: run each chosen expert on this token and add its
            // result to out[i], scaled by its router weight.
            for (std::size_t k = 0; k < topK; k++) {
                int expIdx = selectedIndices[i][k];
                T weight = selectedWeights[i][k];

                // Ideally we construct a batch for each expert, but for simplicity here we do row-by-row
                // Construct single-row tensor for expert
                Tensor<T> rowIn({ 1, dModel });
                for (std::size_t d = 0; d < dModel; ++d) {
                    rowIn.data[d] = x.data[i * dModel + d];
                }

                Tensor<T> rowOut;
                experts[expIdx].forward(rowIn, rowOut);

                // Accumulate: out[i] += weight * expert_out
                for (std::size_t d = 0; d < dModel; ++d) {
                    out.data[i * dModel + d] += weight * rowOut.data[d];
                }
            }
        }
    }

    /**
     * @brief Backward pass: trains experts that were selected in last
     * forward() call and returns gradient with respect to input.
     * router itself does not receive a real gradient (see below).
     *
     * @param dout Gradient w.r.t. forward()'s output [seq, dModel].
     * @param dx Gradient w.r.t. forward()'s input [seq, dModel].
     */
    void backward(const Tensor<T>& dout, Tensor<T>& dx) {
        if (selectedIndices.empty()) {
            throw InvalidSizeError("MoELayer::backward() called before forward()!");
        }
        validation::requireShape(dout, selectedIndices.size(), dModel, "MoELayer output gradient");

        std::size_t seq = dout.shape[0];
        dx = Tensor<T>(dout.shape, 0);

        // Gradient w.r.t. router logits; stays 0 (see router section
        // at end).
        Tensor<T> dGateLogits(gateLogits.shape, 0);

        // Visit every (token, selected expert) pair from forward().
        for (std::size_t i = 0; i < seq; i++) {
            for (std::size_t k = 0; k < topK; k++) {
                int expIdx = selectedIndices[i][k];
                T weight = selectedWeights[i][k];

                // FeedForward only caches its *last* forward() call's input,
                // so each expert's cache needs re-priming with this row
                // before backward() can use it correctly -- inefficient but
                // correct without changing FeedForward's caching design.
                Tensor<T> rowIn({ 1, dModel });
                for (std::size_t d = 0; d < dModel; ++d) {
                    rowIn.data[d] = inputCache.data[i * dModel + d];
                }

                Tensor<T> rowOutDummy;
                experts[expIdx].forward(rowIn, rowOutDummy); // Re-prime expert's cache

                // d(out) = weight * d(expert)
                // d(expert_in) = weight * d(expert_out) = weight * dout
                // Scaling here (rather than after backward()) makes weight
                // flow into expert's own W/b gradients too, not just dx.
                Tensor<T> rowDout({ 1, dModel });
                for (std::size_t d = 0; d < dModel; ++d) {
                    rowDout.data[d] = weight * dout.data[i * dModel + d];
                }

                Tensor<T> dExpertIn;
                experts[expIdx].backward(rowDout, dExpertIn); // This produces grad w.r.t input AND updates weights grad

                // dx += dExpertIn (already weight-scaled via rowDout above)
                for (std::size_t d = 0; d < dModel; ++d) {
                    dx.data[i * dModel + d] += dExpertIn.data[d];
                }

                // Gradient w.r.t Weight (for Router)
                // out = w * expert_out
                // dw = dout * expert_out
                // expert_out is rowOutDummy
                T dot = 0;
                for (std::size_t d = 0; d < dModel; ++d) {
                    dot += rowDout.data[d] * rowOutDummy.data[d];
                }

                // dot (d(weight) via out = weight * expertOut) isn't
                // propagated further: differentiating through top-k
                // softmax routing choice itself isn't implemented, so the
                // router gets no real gradient signal below -- only the
                // experts learn from a real loss.
            }
        }

        // Router backward (Gate): dGateLogits is left at 0 (see above), so
        // this call doesn't train router -- only experts adapt,
        // to whatever routing (untrained, effectively random) gate
        // produces.
        Tensor<T> dGateInDummy;
        gate.backward(dGateLogits, dGateInDummy);
    }

    /**
     * @brief Applies accumulated gradients to router and all experts.
     *
     * @param lr Learning rate.
     * @param rule Optimizer to use; see UpdateRule.
     */
    void update(T lr, UpdateRule rule = UpdateRule::sgd()) {
        gate.update(lr, rule);
        for (auto& expert : experts) {
            expert.update(lr, rule);
        }
    }
};
