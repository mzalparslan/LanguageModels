#pragma once

#include "BertLayer.h"

/**
 * @brief BERT (Bidirectional Encoder Representations from Transformers)
 *
 * A stack of bidirectional (unmasked) encoder layers with two pre-training
 * heads:
 * - MLM (Masked Language Modeling): predict original token at [MASK]ed
 *   positions from context on both sides.
 * - NSP (Next Sentence Prediction): classify from [CLS] vector whether
 *   sentence B follows sentence A.
 * Input embedding = word + learned position + segment (A/B) embeddings.
 *
 * @remark Stateful: trainStep() runs forward and backward back to back
 * against cached activations.
 */
template <typename T>
class BertModel {
private:
    /**
     * @brief Embeddings
     */
    Embedding<T> wordEmb;
    Embedding<T> posEmb; ///< Learnable Position Embeddings
    Embedding<T> typeEmb; ///< Token Type Embeddings (Segment A vs B)
    RMSNorm<T> embNorm;
    RMSNorm<T> mlmNorm;

    /**
     * @brief Caches
     */
    Tensor<T> embOut;
    Tensor<T> mlmDenseOut;
    std::vector<Tensor<T>> layerOuts;

    /**
     * @brief Does actual construction with an engine created from the
     * public constructor seed, so every layer draws from one engine.
     */
    BertModel(RandomEngine rng, std::size_t vocab, std::size_t dim, std::size_t nLayers, std::size_t maxL)
        : vocabSize(vocab), dModel(dim), maxLen(maxL),
        wordEmb(vocab, dim, rng), posEmb(maxL, dim, rng), typeEmb(2, dim, rng),
        embNorm(dim),
        mlmDense(dim, dim, rng), mlmNorm(dim), mlmProj(dim, vocab, rng),
        nspProj(dim, 2, rng)
    {
        validation::requirePositiveSize(nLayers, "BERT layer count");
        validation::requirePositiveSize(maxL, "BERT maximum sequence length");

        for (std::size_t i = 0; i < nLayers; i++) {
            layers.emplace_back(dim, dim * 4, rng);
        }
    }

public:
    // Model (hidden) width.
    std::size_t dModel;
    // Number of token ids model knows.
    std::size_t vocabSize;
    // Longest supported sequence (rows of position embedding).
    std::size_t maxLen;

    /**
     * @brief Encoder Layers
     */
    std::vector<BertLayer<T>> layers;

    /**
     * @param vocab Vocabulary size.
     * @param dim Model width.
     * @param nLayers Number of encoder layers.
     * @param maxL Maximum sequence length.
     * @param randomSeed Seed for initial weights; same seed always
     * gives same model.
     * @throws InvalidParameterSizeError If a size is zero.
     */
    BertModel(std::size_t vocab, std::size_t dim, std::size_t nLayers, std::size_t maxL = 128,
        std::uint32_t randomSeed = 42)
        : BertModel(RandomEngine(randomSeed), vocab, dim, nLayers, maxL) {}

    /**
     * @brief Clears accumulated gradients of every component.
     */
    void zeroGrad() {
        wordEmb.zeroGrad(); posEmb.zeroGrad(); typeEmb.zeroGrad(); embNorm.zeroGrad();
        for (auto& l : layers) {
            l.zeroGrad();
        }
        mlmDense.zeroGrad(); mlmNorm.zeroGrad(); mlmProj.zeroGrad();
        nspProj.zeroGrad();
    }

    /**
     * @brief Encoder forward pass: embeds tokens and runs them through
     * every encoder layer.
     *
     * @param src [Batch*Seq] (Flattened) token ids
     * @param types [Batch*Seq] segment ids (0 = sentence A, 1 = sentence B)
     * @param out returns: encoded sequence [Seq, dModel]
     * @throws InvalidSizeError If src is empty, longer than maxLen, or types has a different length.
     * @throws InvalidParameterError If a token or segment id is out of range.
     */
    void forwardEncoder(const std::vector<std::size_t>& src, const std::vector<std::size_t>& types, Tensor<T>& out) {
        validation::requireNonEmpty(src.size(), "Token sequence");
        validation::requireSameSize(types.size(), src.size(), "Segment id sequence");
        validation::requireAtMost(src.size(), maxLen, "Token sequence length");

        std::size_t seqLen = src.size();

        // 1. Embeddings
        Tensor<T> wEmb, pEmb, tEmb;
        wordEmb.forward(src, wEmb);
        typeEmb.forward(types, tEmb);

        // Position info (0, 1, 2... for each sample in batch? Or simplified to single sequence)
        // Assuming single batch for simplicity, or we need to pass position indices properly.
        // For "Mini", let's assume single sequence or handle flattened correctly.
        // Let's assume src is ONE sequence of length N.
        std::vector<std::size_t> posIds(seqLen);
        for (std::size_t i = 0; i < seqLen; i++) {
            posIds[i] = i;
        }
        posEmb.forward(posIds, pEmb);

        // Sum three embeddings element-wise into one input vector per token.
        embOut = Tensor<T>({ seqLen, dModel });
        for (std::size_t i = 0; i < embOut.size(); i++) {
            embOut[i] = wEmb[i] + pEmb[i] + tEmb[i];
        }

        // Norm
        embNorm.forward(embOut, out);

        // 2. Encoder Layers: each layer's output feeds next one; outputs
        // are kept in layerOuts for backward pass.
        layerOuts.resize(layers.size());
        Tensor<T> curr = out;
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i].forward(curr, layerOuts[i]);
            curr = layerOuts[i];
        }
        out = curr;
    }

    /**
     * @brief Encoder + MLM head forward pass, returning per-position logits over
     * vocabulary. Shared by trainStep() and by inference call sites so
     * dense->ReLU->norm->proj chain isn't hand-duplicated at each one.
     *
     * @param src Token ids (with [MASK] tokens already substituted).
     * @param types Segment ids.
     * @param outLogits Result [Seq, vocabSize]; row i scores every token as
     * possible original at position i.
     */
    void predictMaskedLogits(const std::vector<std::size_t>& src, const std::vector<std::size_t>& types, Tensor<T>& outLogits) {
        Tensor<T> encoded;
        forwardEncoder(src, types, encoded);

        Tensor<T> denseOut;
        mlmDense.forward(encoded, denseOut);

        Tensor<T> act(denseOut.shape);
        for (std::size_t i = 0; i < act.size(); i++) {
            act[i] = std::max(T(0), denseOut[i]);
        }

        Tensor<T> normOut;
        mlmNorm.forward(act, normOut);

        mlmProj.forward(normOut, outLogits);
    }

    /**
     * @brief Full Training Step
     *
     * @param src input tokens (masked)
     * @param types segment ids
     * @param mlmLabels original tokens for [MASK] positions (0 elsewhere)
     * @param nspLabel 0 or 1 (IsNext or NotNext)
     * @param lr Learning rate.
     * @param rule Optimizer to apply (e.g. UpdateRule::adam(step)).
     * @return Combined MLM + NSP loss for this example.
     * @throws InvalidParameterError If lr <= 0, or a label is out of range.
     * @throws InvalidSizeError If the input vectors have inconsistent lengths.
     * @throws NaNError, NonFiniteError If the loss is not finite.
     */
    T trainStep(const std::vector<std::size_t>& src, const std::vector<std::size_t>& types,
        const std::vector<std::size_t>& mlmLabels, std::size_t nspLabel,
        T lr, UpdateRule rule)
    {
        validation::requirePositiveFinite(lr, "Learning rate");
        validation::requireSameSize(mlmLabels.size(), src.size(), "MLM label sequence");
        validation::requireBelow(nspLabel, 2, "NSP label");

        zeroGrad();

        // 1. Encoder Forward
        Tensor<T> encoded;
        forwardEncoder(src, types, encoded);

        // 2. MLM Head
        // Dense -> Act -> Norm -> Proj
        // Act (ReLU)
        mlmDense.forward(encoded, mlmDenseOut);
        Tensor<T> mlmAct(mlmDenseOut.shape);
        for (std::size_t i = 0; i < mlmAct.size(); i++) {
            mlmAct[i] = std::max(T(0), mlmDenseOut[i]);
        }

        Tensor<T> mlmNormOut;
        mlmNorm.forward(mlmAct, mlmNormOut);

        Tensor<T> mlmLogits;
        mlmProj.forward(mlmNormOut, mlmLogits);

        // 3. NSP Head
        // Pooler: Take 1st token (CLS)
        // In BERT, CLS is at index 0.
        // We need shape [1, dModel]
        Tensor<T> clsToken({ 1, dModel });
        for (std::size_t j = 0; j < dModel; j++) {
            clsToken[j] = encoded.data[j]; // encoded[0] is first token vector
        }

        Tensor<T> nspLogits;
        nspProj.forward(clsToken, nspLogits);

        // 4. Loss Calculation: MLM cross-entropy averaged over masked positions,
        // plus NSP cross-entropy.
        T totalLoss = 0;

        // MLM Loss (Cross Entropy on masked positions)
        Tensor<T> dMlmLogits(mlmLogits.shape, 0);
        std::size_t nMasks = 0;

        // mlmLabels[i] == 0 means "not masked" (0 is [PAD], never a real
        // masked-position label), so those positions are skipped below and
        // every other position contributes to MLM loss.

        std::size_t seqLen = src.size();
        for (std::size_t i = 0; i < seqLen; i++) {
            if (mlmLabels[i] == 0) {
                continue; // Not masked
            }

            // Softmax over vocab
            std::size_t rowOffset = i * vocabSize; // Flattened index offset

            T maxV = -1e9;
            for (std::size_t k = 0; k < vocabSize; k++) {
                maxV = std::max(maxV, mlmLogits.data[rowOffset + k]);
            }

            T sum = 0;
            std::vector<T> probs(vocabSize);
            for (std::size_t k = 0; k < vocabSize; k++) {
                probs[k] = std::exp(mlmLogits.data[rowOffset + k] - maxV);
                sum += probs[k];
            }

            std::size_t target = mlmLabels[i];
            validation::requireBelow(target, vocabSize, "MLM label");
            validation::requireFinite(sum, "MLM softmax normalizer");
            validation::requireNonZeroDenominator(sum, "MLM softmax normalizer");
            T prob = probs[target] / sum;
            totalLoss -= std::log(prob + 1e-9);
            nMasks++;

            // Grad
            for (std::size_t k = 0; k < vocabSize; k++) {
                dMlmLogits.data[rowOffset + k] = (probs[k] / sum); // p_i
            }
            dMlmLogits.data[rowOffset + target] -= 1.0;
        }

        if (nMasks > 0) {
            totalLoss /= nMasks; // Mean reduction
        }

        // NSP Loss
        // Softmax over 2 classes
        T nspMax = std::max(nspLogits.data[0], nspLogits.data[1]);
        T nspSum = 0;
        T nspProbs[2];
        for (int k = 0; k < 2; k++) {
            nspProbs[k] = std::exp(nspLogits.data[k] - nspMax);
            nspSum += nspProbs[k];
        }
        validation::requireFinite(nspSum, "NSP softmax normalizer");
        validation::requireNonZeroDenominator(nspSum, "NSP softmax normalizer");
        T nspProb = nspProbs[nspLabel] / nspSum;
        totalLoss -= std::log(nspProb + 1e-9);

        Tensor<T> dNspLogits(nspLogits.shape, 0);
        dNspLogits.data[0] = nspProbs[0] / nspSum;
        dNspLogits.data[1] = nspProbs[1] / nspSum;
        dNspLogits.data[nspLabel] -= 1.0;

        // A non-finite loss would poison every gradient computed from it.
        validation::requireFinite(totalLoss, "Training loss");

        // 5. Backprop: through two heads first, then down encoder
        // stack and into embeddings.

        // NSP
        Tensor<T> dCls;
        nspProj.backward(dNspLogits, dCls);

        // MLM
        Tensor<T> dMlmNormOut;
        mlmProj.backward(dMlmLogits, dMlmNormOut);

        Tensor<T> dMlmAct;
        mlmNorm.backward(dMlmNormOut, dMlmAct);

        // ReLU grad
        Tensor<T> dMlmDenseOut(dMlmAct.shape);
        for (std::size_t i = 0; i < dMlmAct.size(); i++) {
            if (mlmDenseOut[i] > 0) {
                dMlmDenseOut[i] = dMlmAct[i];
            }
            else {
                dMlmDenseOut[i] = 0;
            }
        }

        Tensor<T> dEncoded;
        mlmDense.backward(dMlmDenseOut, dEncoded); // Gradients from MLM head

        // Add NSP gradient to first token (CLS) of dEncoded
        // dEncoded [Seq, Dim]
        for (std::size_t j = 0; j < dModel; j++) {
            dEncoded.data[j] += dCls.data[j];
        }

        // Backprop through Layers
        Tensor<T> currGrad = dEncoded;
        for (std::size_t i = layers.size(); i-- > 0; ) {
            Tensor<T> prevGrad;
            layers[i].backward(currGrad, prevGrad);
            currGrad = prevGrad;
        }

        // Backprop Embeddings
        Tensor<T> dEmbOut;
        embNorm.backward(currGrad, dEmbOut);

        // Distribute to W, P, T
        std::vector<std::size_t> posIds(seqLen);
        for (std::size_t i = 0; i < seqLen; i++) {
            posIds[i] = i;
        }

        wordEmb.backward(src, dEmbOut);
        typeEmb.backward(types, dEmbOut);
        posEmb.backward(posIds, dEmbOut);

        // Update: apply gradients accumulated above to every component.
        wordEmb.update(lr, rule);
        typeEmb.update(lr, rule);
        posEmb.update(lr, rule);
        embNorm.update(lr, rule);

        for (auto& layer : layers) {
            layer.update(lr, rule);
        }

        mlmDense.update(lr, rule);
        mlmNorm.update(lr, rule);
        mlmProj.update(lr, rule);
        nspProj.update(lr, rule);

        return totalLoss;
    }

private:
    /**
     * @brief MLM Head
     */
    LinearLayer<T> mlmDense;
    LinearLayer<T> mlmProj; ///< To Vocab

public:
    /**
     * @brief NSP Head
     */
    LinearLayer<T> nspProj; ///< To 2 classes
};
