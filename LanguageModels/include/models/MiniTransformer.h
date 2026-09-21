#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

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
        forwardImpl(src, tgt, logits, 1);
    }

private:
    /**
     * @brief work of forward(), with final projection to target
     * vocabulary (the largest matrix multiply of a step) split between
     * threads. result does not depend on thread count.
     *
     * @param threads Most threads to use; 1 runs everything on caller.
     */
    void forwardImpl(const std::vector<std::size_t>& src, const std::vector<std::size_t>& tgt,
        Tensor<T>& logits, std::size_t threads) {
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
        if (threads > 1) {
            proj.forwardParallel(decOut3, logits, threads);
        }
        else {
            proj.forward(decOut3, logits);
        }
    }

public:
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
     * @throws NaNError, NonFiniteError If loss is not finite.
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

    /**
     * @brief trainStep() with vocabulary-sized work spread over several
     * threads; same training step, only faster.
     *
     * Almost all of a step's time goes into work that grows with the
     * vocabulary, not with model: output projection (forward and
     * backward), softmax over every position, and clearing and updating
     * two embedding tables and projection weights. Those parts run on
     * several threads. small encoder and decoder layers (width 32, a few
     * tokens) stay on calling thread, since handing them to another thread
     * would cost more than work.
     *
     * Every element is still computed with same operations in same
     * order as trainStep(), so loss and updated weights are
     * bit-identical to trainStep()'s, whatever thread count.
     *
     * @param src Source token ids.
     * @param tgt Decoder input ids.
     * @param label Expected next-token ids, one per decoder position.
     * @param lr Learning rate.
     * @param rule Optimizer to apply (e.g. UpdateRule::adam(step)).
     * @param threadCount Most threads to use; 0 uses hardware thread count.
     * shared ThreadPool caps it at its size.
     * @return Summed cross-entropy loss over target positions.
     * @throws same exceptions as trainStep().
     */
    T trainStepMultipleThread(const std::vector<std::size_t>& src,
        const std::vector<std::size_t>& tgt,
        const std::vector<std::size_t>& label,
        T lr, UpdateRule rule, std::size_t threadCount = 0)
    {
        validation::requirePositiveFinite(lr, "Learning rate");
        validation::requireSameSize(label.size(), tgt.size(), "Label sequence");

        ThreadPool& pool = ThreadPool::shared();
        const std::size_t threads = std::min(
            threadCount == 0 ? ThreadPool::defaultThreadCount() : threadCount, pool.maxThreads());

        // Zero Gradients: backward passes accumulate, so start clean. two
        // embedding tables and projection are big ones.
        encEmb.zeroGradParallel(threads);
        decEmb.zeroGradParallel(threads);
        encAttn.zeroGrad();
        decAttn1.zeroGrad();
        decAttn2.zeroGrad();
        ff.zeroGrad();
        encFf.zeroGrad();
        proj.zeroGradParallel(threads);
        encNorm1.zeroGrad();
        encNorm2.zeroGrad();
        decNorm1.zeroGrad();
        decNorm2.zeroGrad();
        decNorm3.zeroGrad();

        Tensor<T> logits;
        forwardImpl(src, tgt, logits, threads);

        // Cross Entropy Loss Grad: for softmax + cross-entropy gradient
        // w.r.t. logits is simply (probabilities - one_hot(label)).
        // Positions are independent, so threads take a share of them; each writes
        // only its own rows of dlogits and its own entry of logProbOfLabel.
        const std::size_t seq = logits.shape[0];
        const std::size_t vocab = logits.shape[1];
        Tensor<T> dlogits(logits.shape, 0);
        std::vector<T> logProbOfLabel(seq);

        const std::size_t minSoftmaxElementsPerThread = 4096;
        pool.parallelFor(seq, std::max<std::size_t>(minSoftmaxElementsPerThread / vocab, 1), threads,
            [&](std::size_t begin, std::size_t end) {
                std::vector<T> probs(vocab);
                for (std::size_t i = begin; i < end; i++) {
                    // Softmax
                    T maxValue = -1e9;
                    for (std::size_t j = 0; j < vocab; j++) {
                        maxValue = std::max(maxValue, logits.data[i * vocab + j]);
                    }
                    T sum = T(0);
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
                    logProbOfLabel[i] = std::log(probs[y]);

                    // Grad: p - y
                    for (std::size_t j = 0; j < vocab; j++) {
                        dlogits.data[i * vocab + j] = probs[j];
                    }
                    dlogits.data[i * vocab + y] -= T(1);
                }
            });

        // Summed in position order, as trainStep() does, so total is identical.
        T loss = T(0);
        for (std::size_t i = 0; i < seq; i++) {
            loss -= logProbOfLabel[i];
        }

        // A NaN, or an infinite loss (a label with probability 0), would poison
        // every gradient below, so stop before backpropagating.
        validation::requireFinite(loss, "Training loss");

        // =======================
        // BACKWARD PASS
        // =======================

        // --- Decoder Final Projection ---
        Tensor<T> dDecOut3;
        proj.backwardParallel(dlogits, dDecOut3, threads);

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

        // Update All: every layer applies its accumulated gradients. big
        // tables are updated by several threads.
        encEmb.updateParallel(lr, rule, threads);
        decEmb.updateParallel(lr, rule, threads);
        encAttn.update(lr, rule);
        decAttn1.update(lr, rule);
        decAttn2.update(lr, rule);
        ff.update(lr, rule);
        encFf.update(lr, rule);
        proj.updateParallel(lr, rule, threads);
        encNorm1.update(lr, rule);
        encNorm2.update(lr, rule);
        decNorm1.update(lr, rule);
        decNorm2.update(lr, rule);
        decNorm3.update(lr, rule);

        return loss;
    }

    /**
     * @brief One translation, finished or still being extended: its token ids
     * and how likely model thinks it is.
     */
    class Hypothesis {
    public:
        // Token ids: begins with start token, and ends with end token
        // if one was produced.
        std::vector<std::size_t> tokens;
        // Sum of ln p(token | earlier tokens) over every token after start token.
        T logProbability = T(0);
        // True once end token was produced.
        bool finished = false;
    };

    /**
     * @brief Greedy decoding: starts from start token and repeatedly
     * appends single most likely next token, until end token or
     * maxNewTokens tokens.
     *
     * Works on token ids only; mapping words to ids (and finding ids of the
     * start and end tokens) is caller's tokenizer's job.
     *
     * @param src Source token ids.
     * @param startId Target token that begins a sentence (e.g. <SOS>).
     * @param endId Target token that ends a sentence (e.g. <EOS>).
     * @param maxNewTokens Most tokens to generate after start token.
     * @return start token followed by generated tokens, ending with
     * endId if model produced it.
     * @throws InvalidSizeError If src is empty, or maxNewTokens exceeds
     * MiniTransformerConfig::maxSeqLen.
     * @throws InvalidParameterError If startId or endId is outside target vocabulary.
     */
    std::vector<std::size_t> generate(const std::vector<std::size_t>& src, std::size_t startId,
        std::size_t endId, std::size_t maxNewTokens) {
        requireDecodableLength(src, maxNewTokens);

        std::vector<std::size_t> tokens = { startId };
        for (std::size_t step = 0; step < maxNewTokens; step++) {
            Tensor<T> logits;
            forward(src, tokens, logits);

            const std::size_t vocab = logits.shape[1];
            validation::requireBelow(endId, vocab, "End token id");

            // Arg-max over last position: its scores rank every possible next token.
            const T* lastRow = &logits.data[(logits.shape[0] - 1) * vocab];
            std::size_t best = 0;
            for (std::size_t tokenId = 1; tokenId < vocab; tokenId++) {
                if (lastRow[tokenId] > lastRow[best]) {
                    best = tokenId;
                }
            }

            tokens.push_back(best);
            if (best == endId) {
                break;
            }
        }
        return tokens;
    }

    /**
     * @brief Beam search: instead of committing to single best token at
     * each step, keeps beamWidth best partial translations (ranked by
     * summed log probability) and extends all of them, which can find better
     * whole sentences than greedy decoding. A width of 1 is greedy decoding.
     *
     * @param src Source token ids.
     * @param startId Target token that begins a sentence (e.g. <SOS>).
     * @param endId Target token that ends a sentence (e.g. <EOS>).
     * @param maxNewTokens Most tokens to generate after start token.
     * @param beamWidth Number of hypotheses kept alive at every step.
     * @return highest-scoring hypothesis found.
     * @throws InvalidSizeError If src is empty, or maxNewTokens exceeds
     * MiniTransformerConfig::maxSeqLen.
     * @throws InvalidParameterSizeError If beamWidth is zero.
     * @throws InvalidParameterError If startId or endId is outside target vocabulary.
     */
    Hypothesis beamSearch(const std::vector<std::size_t>& src, std::size_t startId,
        std::size_t endId, std::size_t maxNewTokens, std::size_t beamWidth) {
        requireDecodableLength(src, maxNewTokens);
        validation::requirePositiveSize(beamWidth, "Beam width");

        // Init beam: a single hypothesis containing only start token.
        std::vector<Hypothesis> beams(1);
        beams[0].tokens = { startId };

        for (std::size_t step = 0; step < maxNewTokens; step++) {
            std::vector<Hypothesis> nextBeams;

            for (const Hypothesis& beam : beams) {
                if (beam.finished) {
                    nextBeams.push_back(beam);
                    continue;
                }

                Tensor<T> logits;
                forward(src, beam.tokens, logits);

                const std::size_t vocab = logits.shape[1];
                validation::requireBelow(endId, vocab, "End token id");

                // Softmax on last position: turn its logits into log probabilities
                // (with max subtracted first for numerical stability).
                const std::size_t lastRowStart = (logits.shape[0] - 1) * vocab;
                T maxLogit = T(-1e9);
                for (std::size_t tokenId = 0; tokenId < vocab; tokenId++) {
                    maxLogit = std::max(maxLogit, logits.data[lastRowStart + tokenId]);
                }

                T sum = T(0);
                std::vector<T> logProbs(vocab);
                for (std::size_t tokenId = 0; tokenId < vocab; tokenId++) {
                    T expValue = std::exp(logits.data[lastRowStart + tokenId] - maxLogit);
                    sum += expValue;
                    logProbs[tokenId] = expValue; // holds exp() until normalized below
                }
                for (std::size_t tokenId = 0; tokenId < vocab; tokenId++) {
                    logProbs[tokenId] = std::log(logProbs[tokenId] / sum);
                }

                // Expand: extend this hypothesis with every possible next token.
                // (The pruning below keeps only beamWidth best overall.)
                for (std::size_t tokenId = 0; tokenId < vocab; tokenId++) {
                    Hypothesis extended = beam;
                    extended.tokens.push_back(tokenId);
                    extended.logProbability += logProbs[tokenId];
                    if (tokenId == endId) {
                        extended.finished = true;
                    }
                    nextBeams.push_back(extended);
                }
            }

            // Prune: keep only beamWidth highest-scoring hypotheses.
            std::sort(nextBeams.begin(), nextBeams.end(),
                [](const Hypothesis& left, const Hypothesis& right) {
                    return left.logProbability > right.logProbability; // Descending
                });
            if (nextBeams.size() > beamWidth) {
                nextBeams.resize(beamWidth);
            }
            beams = nextBeams;

            // Stop early once every kept hypothesis has produced end token.
            bool allFinished = true;
            for (const Hypothesis& beam : beams) {
                if (!beam.finished) {
                    allFinished = false;
                }
            }
            if (allFinished) {
                break;
            }
        }

        return beams[0];
    }

private:
    /**
     * @brief Shared argument checks of generate() and beamSearch().
     */
    void requireDecodableLength(const std::vector<std::size_t>& src, std::size_t maxNewTokens) const {
        validation::requireNonEmpty(src.size(), "Source token sequence");
        // last decoder pass sees start token plus all but final new token.
        validation::requireAtMost(maxNewTokens, MiniTransformerConfig::maxSeqLen,
            "Number of generated tokens");
    }

    FeedForward<T> encFf;  ///< Encoder FF
    LinearLayer<T> proj; ///< To Vocab
};