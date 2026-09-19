#pragma once

#include <memory>

#include "FeedForward.h"
#include "MoELayer.h"
#include "RMSNorm.h"
#include "MultiHeadAttention.h"

/**
 * @brief GPT Components (Decoder-Only), multi-head attention variant.
 *
 * Standard Decoder Block: Pre-Norm
 * x = x + Attn(Norm(x))
 * x = x + FF(Norm(x)) OR MoE(Norm(x))
 */
template <typename T>
class DecoderMultiHead {
private:
    // Causal multi-head self-attention sublayer.
    MultiHeadAttention<T> attn;
    bool useMoe = false;

    /**
     * @brief Residual sums cached for backward.
     */
    Tensor<T> res1, res2;
    /**
     * @brief Inputs to norm1/norm2 (the residual stream at each sublayer).
     */
    Tensor<T> n1In, n2In;

public:
    /**
     * @brief Exactly one of these is active, selected by useMoe.
     */
    std::unique_ptr<FeedForward<T>> ff;
    std::unique_ptr<MoELayer<T>> moe;

    // Normalization applied before attention (norm1) and before FF/MoE (norm2).
    RMSNorm<T> norm1, norm2;
    // Model width.
    std::size_t dModel;

    /**
     * @param dim Model width; must be divisible by numHeads.
     * @param dFfDim Inner width of feed-forward sublayer (or of each expert).
     * @param rng Engine initial weights are drawn from.
     * @param moeFlag Use a Mixture-of-Experts layer instead of a plain FF.
     * @param numExperts Number of experts (MoE only).
     * @param topK Experts each token is routed to (MoE only).
     * @param numHeads Number of attention heads.
     */
    DecoderMultiHead(std::size_t dim, std::size_t dFfDim, RandomEngine& rng, bool moeFlag = false, std::size_t numExperts = 4, std::size_t topK = 2, std::size_t numHeads = 4)
        : dModel(dim), attn(dim, numHeads, rng), norm1(dim), norm2(dim), useMoe(moeFlag)
    {
        if (useMoe) {
            moe = std::make_unique<MoELayer<T>>(dim, dFfDim, rng, numExperts, topK);
        }
        else {
            ff = std::make_unique<FeedForward<T>>(dim, dFfDim, rng);
        }
    }

    /**
     * @brief Clears accumulated gradients of every sublayer; only the
     * active FF/MoE branch exists.
     */
    void zeroGrad() {
        attn.zeroGrad();
        norm1.zeroGrad();
        norm2.zeroGrad();
        if (useMoe) {
            moe->zeroGrad();
        }
        else {
            ff->zeroGrad();
        }
    }

    /**
     * @brief Forward pass through both sublayers, each wrapped in a residual
     * connection.
     *
     * @param x Input [seq, dModel].
     * @param out Output [seq, dModel].
     * @param rope Optional rotary position embedding for attention heads
     * (sized for per-head width); nullptr disables it.
     */
    template <typename RopeConfig = ModelConfig>
    void forward(const Tensor<T>& x, Tensor<T>& out, RotaryEmbedding<T, RopeConfig>* rope = nullptr) {
        validation::requireColumns(x, dModel, "DecoderMultiHead input");
        // 1. Attention Sublayer: normalize input (pre-norm), run causal
        // self-attention on it, then add it back to un-normalized input.
        n1In = x;
        Tensor<T> n1Out;
        norm1.forward(n1In, n1Out);

        Tensor<T> attnOut;
        attn.forward(n1Out, n1Out, n1Out, attnOut, true, rope);

        // Residual 1
        res1 = Tensor<T>(x.shape);
        for (std::size_t i = 0; i < res1.size(); i++) {
            res1[i] = n1In[i] + attnOut[i];
        }

        // 2. FeedForward / MoE Sublayer, on output of attention
        // residual; MoE routes each token to its top-k experts.
        n2In = res1;
        Tensor<T> n2Out;
        norm2.forward(n2In, n2Out);

        Tensor<T> ffOut;
        if (useMoe) {
            moe->forward(n2Out, ffOut);
        }
        else {
            ff->forward(n2Out, ffOut);
        }

        // Residual 2
        res2 = Tensor<T>(x.shape);
        for (std::size_t i = 0; i < res2.size(); i++) {
            res2[i] = n2In[i] + ffOut[i];
        }

        out = res2;
    }

    /**
     * @brief Backward pass: walks two sublayers in reverse, adding each
     * residual path's gradient to its sublayer path's gradient.
     *
     * @param dout Gradient w.r.t. forward()'s output [seq, dModel].
     * @param dx Gradient w.r.t. forward()'s input [seq, dModel].
     */
    void backward(const Tensor<T>& dout, Tensor<T>& dx) {
        // --- FF/MoE Sublayer ---
        Tensor<T> dN2Out;
        if (useMoe) {
            moe->backward(dout, dN2Out);
        }
        else {
            ff->backward(dout, dN2Out);
        }

        // Backprop through Norm2
        Tensor<T> dN2InV2;
        norm2.backward(dN2Out, dN2InV2);

        // Total grad at n2In (res1): residual path passes dout through
        // unchanged, sublayer path comes back through norm2.
        Tensor<T> dRes1 = Tensor<T>(dout.shape);
        for (std::size_t i = 0; i < dRes1.size(); i++) {
            dRes1[i] = dout[i] + dN2InV2[i]; // + residual
        }

        // --- Attention Sublayer ---
        // Q, K and V were all same tensor (n1Out), so its gradient is the
        // sum of three input gradients.
        Tensor<T> dN1OutAccum = Tensor<T>(dout.shape, 0);
        Tensor<T> dq, dk, dv;
        attn.backward(dRes1, dq, dk, dv);

        for (std::size_t i = 0; i < dN1OutAccum.size(); i++) {
            dN1OutAccum[i] = dq[i] + dk[i] + dv[i];
        }

        // Backprop through Norm1
        Tensor<T> dN1InV2;
        norm1.backward(dN1OutAccum, dN1InV2);

        // Total grad at x (n1In): residual path plus path through norm1.
        dx = Tensor<T>(dout.shape);
        for (std::size_t i = 0; i < dx.size(); i++) {
            dx[i] = dRes1[i] + dN1InV2[i]; // + residual
        }
    }

    /**
     * @brief Applies accumulated gradients to every sublayer.
     *
     * @param lr Learning rate.
     * @param rule Optimizer to use; see UpdateRule.
     */
    void update(T lr, UpdateRule rule = UpdateRule::sgd()) {
        attn.update(lr, rule);
        norm1.update(lr, rule);
        norm2.update(lr, rule);
        if (useMoe) {
            moe->update(lr, rule);
        }
        else {
            ff->update(lr, rule);
        }
    }

    /**
     * @brief Mixture-of-Experts load-balancing auxiliary loss from last
     * forward() call, or 0 when block uses a plain FF. Satisfies the
     * optional HasAuxLoss concept.
     */
    T getAuxLoss() const {
        if (useMoe && moe) {
            return moe->currAuxLoss;
        }
        return 0;
    }
};
