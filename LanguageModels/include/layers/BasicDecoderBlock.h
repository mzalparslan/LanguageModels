#pragma once

#include "AttentionHead.h"
#include "FeedForward.h"
#include "RMSNorm.h"

/**
 * @brief Standard Decoder Block: Pre-Norm
 * x = x + Attn(Norm(x))
 * x = x + FF(Norm(x))
 *
 * Pre-norm (normalizing input of each sublayer, not its output) keeps
 * residual path an unmodified identity, which stabilizes training of
 * deep stacks.
 *
 * @remark Stateful: forward() caches its intermediate values and backward()
 * consumes them, so call backward() right after forward() it belongs to.
 */
template <typename T>
class BasicDecoderBlock {
private:
    // Causal self-attention sublayer.
    AttentionHead<T> attn;

    /**
     * @brief Cache for residual connections
     */
    Tensor<T> res1, res2;
    /**
     * @brief Inputs to norms (which are residuals from prev steps)
     */
    Tensor<T> n1In, n2In;

public:
    // Position-wise feed-forward sublayer.
    FeedForward<T> ff;
    // Normalization applied before attention (norm1) and before FF (norm2).
    RMSNorm<T> norm1, norm2;
    // Model width.
    std::size_t dModel;

    /**
     * @param dim Model width.
     * @param dFfDim Inner width of feed-forward sublayer.
     * @param rng Engine initial weights are drawn from.
     */
    BasicDecoderBlock(std::size_t dim, std::size_t dFfDim, RandomEngine& rng)
        : dModel(dim), attn(dim, rng), ff(dim, dFfDim, rng), norm1(dim), norm2(dim) {
    }

    /**
     * @brief Clears accumulated gradients of every sublayer.
     */
    void zeroGrad() {
        attn.zeroGrad();
        ff.zeroGrad();
        norm1.zeroGrad();
        norm2.zeroGrad();
    }

    /**
     * @brief Forward pass through both sublayers, each wrapped in a residual
     * connection.
     *
     * @param x Input [seq, dModel].
     * @param out Output [seq, dModel].
     * @param rope Optional rotary position embedding for attention head;
     * nullptr disables it.
     */
    template <typename RopeConfig = ModelConfig>
    void forward(const Tensor<T>& x, Tensor<T>& out, RotaryEmbedding<T, RopeConfig>* rope = nullptr) {
        validation::requireColumns(x, dModel, "BasicDecoderBlock input");
        // 1. Attention Sublayer
        n1In = x; // Pre-norm input
        Tensor<T> n1Out;
        norm1.forward(n1In, n1Out);

        Tensor<T> attnOut;
        // Self-Attention with Causal Mask (mask=true)
        attn.forward(n1Out, n1Out, n1Out, attnOut, rope, true);

        // Residual 1: add sublayer's output back onto its (un-normalized)
        // input.
        res1 = Tensor<T>(x.shape);
        for (std::size_t i = 0; i < res1.size(); i++) {
            res1[i] = n1In[i] + attnOut[i];
        }

        // 2. FeedForward Sublayer, on output of attention residual.
        n2In = res1;
        Tensor<T> n2Out;
        norm2.forward(n2In, n2Out);

        Tensor<T> ffOut;
        ff.forward(n2Out, ffOut);

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
        // dx comes from next layer (or top)
        // dRes2 = dout

        // --- FF Sublayer ---
        // res2 = n2In + ffOut
        // dFfOut = dout
        // dN2InV1 = dout (pass through residual)

        // Backprop through FF
        // FF input was n2Out = Norm2(n2In)
        Tensor<T> dN2Out;
        ff.backward(dout, dN2Out); // dFfOut -> dN2Out (grad at FF input)

        // Backprop through Norm2
        Tensor<T> dN2InV2;
        norm2.backward(dN2Out, dN2InV2);

        // Total grad at n2In (res1)
        Tensor<T> dRes1 = Tensor<T>(dout.shape);
        for (std::size_t i = 0; i < dRes1.size(); i++) {
            // + residual
            dRes1[i] = dout[i] + dN2InV2[i];
        }

        // --- Attention Sublayer ---
        // res1 = n1In + attnOut
        // dAttnOut = dRes1
        // dN1InV1 = dRes1

        // Backprop through Attention
        // Attn input was n1Out = Norm1(n1In)
        Tensor<T> dN1OutAccum = Tensor<T>(dout.shape, 0);
        Tensor<T> dq, dk, dv;
        attn.backward(dRes1, dq, dk, dv);

        // Q, K, V all came from n1Out
        for (std::size_t i = 0; i < dN1OutAccum.size(); i++) {
            dN1OutAccum[i] = dq[i] + dk[i] + dv[i];
        }

        // Backprop through Norm1
        Tensor<T> dN1InV2;
        norm1.backward(dN1OutAccum, dN1InV2);

        // Total grad at x (n1In)
        dx = Tensor<T>(dout.shape);
        for (std::size_t i = 0; i < dx.size(); i++) {
            // + residual
            dx[i] = dRes1[i] + dN1InV2[i];
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
        ff.update(lr, rule);
        norm1.update(lr, rule);
        norm2.update(lr, rule);
    }
};
