#pragma once

#include "FeedForward.h"
#include "AttentionHead.h"
#include "RMSNorm.h"
#include "Embedding.h"
#include "MiniTransformerConfig.h"

/**
 * @brief Small encoder-decoder Transformer for sequence-to-sequence tasks
 * such as translation ("Attention Is All You Need" layout, pre-norm, with
 * RoPE for positions).
 *
 * encoder has one self-attention layer and one feed-forward layer. The
 * decoder has masked self-attention, cross-attention over encoder output,
 * and a feed-forward layer, followed by a projection to target-vocabulary
 * logits. Every sublayer is wrapped in a residual connection.
 *
 * @remark Stateful: forward() caches intermediate activations of every
 * layer, and trainStep() runs manual backward pass against them.
 */
template <typename T>
class MiniTransformer {
private:
    // Token embeddings for source (encoder) and target (decoder) vocabularies.
    Embedding<T> encEmb, decEmb;

    RotaryEmbedding<T, MiniTransformerConfig> ropeEnc; ///< RoPE Module

    AttentionHead<T> encAttn; ///< Encoder self-attention
    AttentionHead<T> decAttn1; ///< Decoder masked self-attention
    AttentionHead<T> decAttn2; ///< Decoder cross-attention (queries from decoder, keys/values from encoder)

    /**
     * @brief RMSNorm Layers
     */
    RMSNorm<T> encNorm1, encNorm2;
    RMSNorm<T> decNorm1, decNorm2, decNorm3;

    /**
     * @brief Cache for Pre-Norm inputs (x before norm)
     */
    Tensor<T> encN1In, encN2In;
    Tensor<T> decN1In, decN2In, decN3In;

    /**
     * @brief Forward Caches
     */
    Tensor<T> encOut, encRes1;
    Tensor<T> encFfOut; ///< Cache for encoder FF output
    Tensor<T> decOut3, ffOut;
    Tensor<T> decRes1, decRes2;

    /**
     * @brief Does actual construction with an engine created from the
     * public constructor seed, so every layer draws from one engine.
     */
    MiniTransformer(RandomEngine rng, std::size_t srcVocab, std::size_t tgtVocab)
        // RotaryEmbedding takes its dimension/max length from
        // MiniTransformerConfig at compile time, not from constructor arguments.
        : encEmb(srcVocab, d_model, rng), decEmb(tgtVocab, d_model, rng),
        encAttn(d_model, rng), decAttn1(d_model, rng), decAttn2(d_model, rng),
        ff(d_model, d_model * 2, rng), encFf(d_model, d_model * 2, rng), proj(d_model, tgtVocab, rng),
        // Init Norms
        encNorm1(d_model), encNorm2(d_model),
        decNorm1(d_model), decNorm2(d_model), decNorm3(d_model)
    {
    }

public:
    // Model width shared by every layer; must match MiniTransformerConfig::d_head.
    static constexpr std::size_t d_model = 32;
    static_assert(MiniTransformerConfig::d_head == d_model,
        "MiniTransformerConfig::d_head must match MiniTransformer::d_model");
    FeedForward<T> ff;      ///< Decoder FF

    /**
     * @param srcVocab Source vocabulary size (encoder embedding rows).
     * @param tgtVocab Target vocabulary size (embedding rows and output logits).
     * @param maxLen Unused: maximum sequence length comes from
     * MiniTransformerConfig::maxSeqLen at compile time.
     * @param randomSeed Seed for initial weights; same seed always
     * gives same model.
     */
    MiniTransformer(std::size_t srcVocab, std::size_t tgtVocab, std::size_t maxLen = 100,
        std::uint32_t randomSeed = 42)
        : MiniTransformer(RandomEngine(randomSeed), srcVocab, tgtVocab) {}

    /**
     * @brief Runs encoder over src, then decoder over tgt attending
     * to encoder output, and returns per-position target-vocabulary logits.
     *
     * @param src Source token ids.
     * @param tgt Target token ids fed to decoder (the sequence shifted
     * right, so position i predicts token i + 1).
     * @param logits Output [tgt.size(), tgtVocab].
     * @throws InvalidSizeError If a sequence is empty or longer than MiniTransformerConfig::maxSeqLen.
     * @throws InvalidParameterError If a token id is outside its vocabulary.
     */
    void forward(const std::vector<std::size_t>& src, const std::vector<std::size_t>& tgt, Tensor<T>& logits) {
        validation::requireNonEmpty(src.size(), "Source token sequence");
        validation::requireNonEmpty(tgt.size(), "Target token sequence");
        validation::requireAtMost(src.size(), MiniTransformerConfig::maxSeqLen, "Source sequence length");
        validation::requireAtMost(tgt.size(), MiniTransformerConfig::maxSeqLen, "Target sequence length");

        // ==========================
        // 1. ENCODER
        // ==========================

        // 1.0 Embed
        Tensor<T> srcEmb;
        encEmb.forward(src, srcEmb);

        // 1.1 Layer 1: Self-Attn
        // Pre-Norm: x = x + Attn(Norm(x))
        encN1In = srcEmb; // Save residual source (x)
        Tensor<T> normSrc;
        encNorm1.forward(encN1In, normSrc); // Norm(x)

        Tensor<T> attnOut;
        encAttn.forward(normSrc, normSrc, normSrc, attnOut, &ropeEnc, false);

        // Residual Add: encRes1 = x + attnOut
        encRes1 = Tensor<T>(attnOut.shape);
        for (std::size_t i = 0; i < encRes1.size(); i++) {
            encRes1[i] = encN1In[i] + attnOut[i];
        }

        // 1.2 Layer 2: FeedForward
        // x = x + FF(Norm(x))
        encN2In = encRes1;
        Tensor<T> normRes1;
        encNorm2.forward(encN2In, normRes1);

        encFf.forward(normRes1, encFfOut);

        // Residual Add: encOut = x + ffOut
        encOut = Tensor<T>(encFfOut.shape);
        for (std::size_t i = 0; i < encOut.size(); i++) {
            encOut[i] = encN2In[i] + encFfOut[i];
        }

        // ==========================
        // 2. DECODER
        // ==========================

        // 2.0 Embed
        Tensor<T> tgtEmb;
        decEmb.forward(tgt, tgtEmb);

        // 2.1 Layer 1: Masked Self-Attn
        decN1In = tgtEmb;
        Tensor<T> dNorm1;
        decNorm1.forward(decN1In, dNorm1);

        Tensor<T> maskAttnOut;
        decAttn1.forward(dNorm1, dNorm1, dNorm1, maskAttnOut, &ropeEnc, true);

        decRes1 = Tensor<T>(maskAttnOut.shape);
        for (std::size_t i = 0; i < decRes1.size(); i++) {
            decRes1[i] = decN1In[i] + maskAttnOut[i];
        }

        // 2.2 Layer 2: Cross Attn
        decN2In = decRes1;
        Tensor<T> dNorm2;
        decNorm2.forward(decN2In, dNorm2);

        // Q from Decoder (normalized), K/V from Encoder (encOut)
        // Note: LLaMA applies simple linear projection to K/V, but standard Pre-Norm takes Enc output as is.
        // We do typically normalize K/V in Encoder output step (which we did: encOut is a residual sum).
        // Some papers suggest applying a final Norm to Encoder Output. We skipped that for "Mini".
        Tensor<T> crossOut;
        decAttn2.forward(dNorm2, encOut, encOut, crossOut, &ropeEnc, false);

        decRes2 = Tensor<T>(crossOut.shape);
        for (std::size_t i = 0; i < decRes2.size(); i++) {
            decRes2[i] = decN2In[i] + crossOut[i];
        }

        // 2.3 Layer 3: FF
        decN3In = decRes2;
        Tensor<T> dNorm3;
        decNorm3.forward(decN3In, dNorm3);

        ff.forward(dNorm3, ffOut);

        decOut3 = Tensor<T>(ffOut.shape);
        for (std::size_t i = 0; i < decOut3.size(); i++) {
            decOut3[i] = decN3In[i] + ffOut[i];
        }

        // Final Projection
        // (Ideally we would have one more Norm here: decOut3 = Norm(decOut3))
        proj.forward(decOut3, logits);
    }

    /**
     * @brief One full training step: zero gradients, forward, cross-entropy
     * loss, backward through every layer, then a parameter update.
     *
     * @param src Source token ids.
     * @param tgt Decoder input ids.
     * @param label Expected next-token ids, one per decoder position.
     * @param lr Learning rate.
     * @param rule Optimizer to apply (e.g. UpdateRule::adam(step)).
     * @return Summed cross-entropy loss over target positions.
     * @throws InvalidParameterError If lr <= 0, or a label id is out of range.
     * @throws InvalidSizeError If label and tgt differ in length.
     * @throws NaNError, NonFiniteError If the loss is not finite.
     */
    T trainStep(const std::vector<std::size_t>& src,
        const std::vector<std::size_t>& tgt,
        const std::vector<std::size_t>& label,
        T lr, UpdateRule rule)
    {
        validation::requirePositiveFinite(lr, "Learning rate");
        validation::requireSameSize(label.size(), tgt.size(), "Label sequence");

        // Zero Gradients: backward passes accumulate, so start clean.
        encEmb.zeroGrad(); 
        decEmb.zeroGrad();
        encAttn.zeroGrad(); 
        decAttn1.zeroGrad(); 
        decAttn2.zeroGrad();
        ff.zeroGrad(); 
        encFf.zeroGrad(); 
        proj.zeroGrad();
        encNorm1.zeroGrad(); 
        encNorm2.zeroGrad();
        decNorm1.zeroGrad(); 
        decNorm2.zeroGrad(); 
        decNorm3.zeroGrad();

        Tensor<T> logits;
        forward(src, tgt, logits);

        // Cross Entropy Loss Grad: for softmax + cross-entropy gradient
        // w.r.t. logits is simply (probabilities - one_hot(label)).
        std::size_t seq = logits.shape[0];
        std::size_t vocab = logits.shape[1];
        Tensor<T> dlogits(logits.shape, 0);
        T loss = T(0);

        for (std::size_t i = 0; i < seq; i++) {
            // Softmax
            T maxValue = -1e9;
            for (std::size_t j = 0; j < vocab; j++) {
                maxValue = std::max(maxValue, logits.data[i * vocab + j]);
            }
            T sum = T(0);
            std::vector<T> probs(vocab);
            for (std::size_t j = 0; j < vocab; j++) {
                probs[j] = std::exp(logits.data[i * vocab + j] - maxValue);
                sum += probs[j];
            }
            validation::requireFinite(sum, "Softmax normalizer");
            validation::requireNonZeroDenominator(sum, "softmax normalizer");
            for (std::size_t j = 0; j < vocab; j++) {
                probs[j] /= sum;
            }

            // Log Loss
            std::size_t y = label[i];
            validation::requireBelow(y, vocab, "Label token id");
            loss -= std::log(probs[y]);

            // Grad: p - y
            for (std::size_t j = 0; j < vocab; j++) {
                dlogits.data[i * vocab + j] = probs[j];
            }
            dlogits.data[i * vocab + y] -= 1.0;
        }

        // A NaN, or an infinite loss (a label with probability 0), would poison
        // every gradient below, so stop before backpropagating.
        validation::requireFinite(loss, "Training loss");

        // =======================
        // BACKWARD PASS
        // =======================

        // --- Decoder Final Projection ---
        Tensor<T> dDecOut3;
        proj.backward(dlogits, dDecOut3);

        // --- Decoder Layer 3: FF ---
        // x = x + FF(Norm(x))
        // d_x = d_out + Norm.back(FF.back(d_out))
        Tensor<T> dNorm3Out; // Grad at output of Norm3 (input to FF)
        ff.backward(dDecOut3, dNorm3Out); // FF.back takes gradient from "FF output side" (dDecOut3)

        Tensor<T> dDecN3In; // Grad at input of Norm3
        decNorm3.backward(dNorm3Out, dDecN3In);

        // Residual Sum: dDecRes2 = dDecOut3 (skip) + dDecN3In (branch)
        Tensor<T> dDecRes2(dDecOut3.shape);
        for (std::size_t i = 0; i < dDecRes2.size(); i++) {
            dDecRes2[i] = dDecOut3[i] + dDecN3In[i];
        }
        // --- Decoder Layer 2: Cross Attn ---
        // x = x + Attn(Norm(x), context)
        Tensor<T> dNorm2Out;
        Tensor<T> dEncOutCross, dEncOutCrossK; // Gradients w.r.t Encoder Outputs (K, V)
        decAttn2.backward(dDecRes2, dNorm2Out, dEncOutCross, dEncOutCrossK);

        Tensor<T> dDecN2In;
        decNorm2.backward(dNorm2Out, dDecN2In);

        // Residual Sum: dDecRes1 = dDecRes2 + dDecN2In
        Tensor<T> dDecRes1(dDecRes2.shape);
        for (std::size_t i = 0; i < dDecRes1.size(); i++) {
            dDecRes1[i] = dDecRes2[i] + dDecN2In[i];
        }

        // --- Decoder Layer 1: Masked Self Attn ---
        Tensor<T> dNorm1Out, dDummyK, dDummyV;
        decAttn1.backward(dDecRes1, dNorm1Out, dDummyK, dDummyV);
        // Self-attention: Q, K, V all come from same source (dNorm1Out).
        // AttentionHead::backward returns dQ_in (into dNorm1Out arg), dK_in, dV_in.
        // We must sum them up.
        for (std::size_t i = 0; i < dNorm1Out.size(); i++) {
            dNorm1Out[i] += dDummyK[i] + dDummyV[i];
        }

        Tensor<T> dDecN1In;
        decNorm1.backward(dNorm1Out, dDecN1In);

        // Residual Sum: dTgtEmb = dDecRes1 + dDecN1In
        Tensor<T> dTgtEmb(dDecRes1.shape);
        for (std::size_t i = 0; i < dTgtEmb.size(); i++) {
            dTgtEmb[i] = dDecRes1[i] + dDecN1In[i];
        }

        decEmb.backward(tgt, dTgtEmb);
        // --- ENCODER BACKWARD ---
        // Gradient from Cross Attn: dEncOut = dEncOutCross + dEncOutCrossK
        Tensor<T> dEncOut(dEncOutCross.shape);
        for (std::size_t i = 0; i < dEncOut.size(); i++) {
            dEncOut[i] = dEncOutCross[i] + dEncOutCrossK[i];
        }

        // --- Encoder Layer 2: FF ---
        Tensor<T> dEnorm2Out;
        encFf.backward(dEncOut, dEnorm2Out);

        Tensor<T> dEncN2In;
        encNorm2.backward(dEnorm2Out, dEncN2In);

        // Residual: dEncRes1 = dEncOut + dEncN2In
        Tensor<T> dEncRes1(dEncOut.shape);
        for (std::size_t i = 0; i < dEncRes1.size(); i++) {
            dEncRes1[i] = dEncOut[i] + dEncN2In[i];
        }

        // --- Encoder Layer 1: Self Attn ---
        Tensor<T> dEnorm1Out, dEsk, dEsv;
        encAttn.backward(dEncRes1, dEnorm1Out, dEsk, dEsv);
        for (std::size_t i = 0; i < dEnorm1Out.size(); i++) {
            dEnorm1Out[i] += dEsk[i] + dEsv[i];
        }

        Tensor<T> dEncN1In;
        encNorm1.backward(dEnorm1Out, dEncN1In);

        // Residual: dSrcEmb = dEncRes1 + dEncN1In
        Tensor<T> dSrcEmb(dEncRes1.shape);
        for (std::size_t i = 0; i < dSrcEmb.size(); i++) {
            dSrcEmb[i] = dEncRes1[i] + dEncN1In[i];
        }

        encEmb.backward(src, dSrcEmb);

        // Update All: every layer applies its accumulated gradients.
        encEmb.update(lr, rule);
        decEmb.update(lr, rule);
        encAttn.update(lr, rule);
        decAttn1.update(lr, rule);
        decAttn2.update(lr, rule);
        ff.update(lr, rule);
        encFf.update(lr, rule);
        proj.update(lr, rule);
        encNorm1.update(lr, rule);
        encNorm2.update(lr, rule);
        decNorm1.update(lr, rule);
        decNorm2.update(lr, rule); 
        decNorm3.update(lr, rule);

        return loss;
    }

private:
    FeedForward<T> encFf;  ///< Encoder FF
    LinearLayer<T> proj; ///< To Vocab
};