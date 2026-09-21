#pragma once

#include <functional>

#include "TensorOps.h"
#include "LinearLayer.h"
#include "RotaryEmbedding.h"
#include "ModelConfig.h"

/**
 * @brief Self-Attention head: projects a dModel-wide input to dHead-wide
 * Q/K/V and output. dHead equals dModel for a standalone single head; a
 * MultiHeadAttention builds several with dHead = dModel / numHeads.
 */
template <typename T>
class AttentionHead {
public:
	std::size_t dModel;
	std::size_t dHead;
	/**
	 * @brief Query layer.
	 */
	LinearLayer<T> Wq;
	/**
	 * @brief Key layer.
	 */
	LinearLayer<T> Wk;
	/**
	 * @brief Value layer.
	 */
	LinearLayer<T> Wv;
	/**
	 * @brief Output layer.
	 */
	LinearLayer<T> Wo;
	Tensor<T> attnWeights;

	/**
	 * @brief Single standalone head: dHead equals dModel.
	 *
	 * @param dm Input (model) width.
	 * @param rng Engine initial weights are drawn from.
	 */
	AttentionHead(std::size_t dm, RandomEngine& rng) : AttentionHead(dm, dm, rng) {}

	/**
	 * @brief Head that reads a dm-wide input and works in a dh-wide space.
	 * Wq/Wk/Wv project dm -> dh; Wo mixes dh-wide result.
	 *
	 * @param dm Input (model) width.
	 * @param dh Per-head width; dm / numHeads when used by MultiHeadAttention.
	 * @param rng Engine initial weights are drawn from.
	 */
	AttentionHead(std::size_t dm, std::size_t dh, RandomEngine& rng) : dModel(dm), dHead(dh),
		Wq(dm, dh, rng), Wk(dm, dh, rng), Wv(dm, dh, rng), Wo(dh, dh, rng) {
	}

	/**
	 * @brief Scaled dot-product attention:
	 * out = Wo( softmax(Q K^T / sqrt(dHead)) V ), with Q/K/V = Wq/Wk/Wv(input).
	 *
	 * Templated on RopeConfig so callers can pass a RotaryEmbedding sized for
	 * their own d_model (see MiniTransformerConfig.h) instead of being
	 * locked to ModelConfig::d_head.
	 *
	 * @param inputQ Query source [seqQ, dModel].
	 * @param inputK Key source [seqK, dModel].
	 * @param inputV Value source [seqK, dModel] (same length as inputK).
	 * @param out Result [seqQ, dHead].
	 * @param rope Optional rotary position embedding applied to Q and K;
	 * nullptr disables it.
	 * @param masked If true, position i cannot attend to positions j > i
	 * (causal mask for decoder self-attention).
	 * @throws InvalidSizeError If an input is not a [seq, dModel] matrix, or the
	 * key and value lengths differ.
	 */
	template <typename RopeConfig = ModelConfig>
	void forward(const Tensor<T>& inputQ,
		const Tensor<T>& inputK,
		const Tensor<T>& inputV,
		Tensor<T>& out,
		RotaryEmbedding<T, RopeConfig>* rope = nullptr,
		bool masked = false) {

		validation::requireColumns(inputQ, dModel, "Attention query input");
		validation::requireColumns(inputK, dModel, "Attention key input");
		validation::requireColumns(inputV, dModel, "Attention value input");
		validation::requireSameSize(inputV.shape[0], inputK.shape[0], "Attention value length");

		// Query length
		std::size_t seqQ = inputQ.shape[0];
		// Key length (same as V)
		std::size_t seqK = inputK.shape[0];

		// Step - 1. Projections: Project input into different sub-spaces:
		// Q (Query): What I am looking for?
		// K (Key)  : What do I contain?
		// V (Value): What information do I pass if selected?
		Wq.forward(inputQ, Q);
		Wk.forward(inputK, K);
		Wv.forward(inputV, V);

		// Step - 1.5: Apply RoPE if present.
		if (rope) {
			rope->apply(Q);
			rope->apply(K);
			// RotaryEmbedding only reads shared tables, so a fresh instance is
			// equivalent and avoids keeping a pointer that could outlive caller's.
			undoRope = [](Tensor<T>& gradient) {
				RotaryEmbedding<T, RopeConfig> rotation;
				rotation.applyInverse(gradient);
			};
		}
		else {
			undoRope = nullptr;
		}

		// Step - 2: Scores = Q * K^T / sqrt(d)
		// Dot product measures similary.
		// If Q (interest) aligns with K (content), score is high.
		Tensor<T> KT;
		transpose2D(K, KT);
		// [seq_q, seq_k]
		MatMul2D(Q, KT, scores);

		// Divide by sqrt(dHead) so score variance does not grow with
		// head width, which would saturate softmax.
		T scale = T(1.0) / std::sqrt((T)dHead);
		for (auto& x : scores.data) {
			x *= scale;
		}

		// 3. Masking for Decoder Self-Attention: overwriting future
		// positions with a huge negative score makes their softmax weight
		// ~0, so a token cannot see tokens that come after it.
		if (masked) {
			for (std::size_t i = 0; i < seqQ; i++) {
				for (std::size_t j = 0; j < seqK; j++) {
					if (j > i) {
						// Future mask
						scores.data[i * seqK + j] = -1e9;
					}
				}
			}
		}

		// 4. Softmax over keys of each query. Kept in attnWeights
		// because backward() needs it.
		attnWeights = scores;
		softmaxRow(attnWeights);

		// 5. Output = Attn * V: each query's result is attention-weighted
		// mix of value vectors.
		Tensor<T> context;
		MatMul2D(attnWeights, V, context);

		// 6. Final linear projection.
		Wo.forward(context, out);
	}

	/**
	 * @brief Backward pass for last forward() call: accumulates gradients
	 * into Wq/Wk/Wv/Wo and returns gradients w.r.t. three inputs.
	 *
	 * If forward() applied RoPE, Q/K gradients are rotated back through
	 * it before they reach Wq/Wk.
	 *
	 * @param dOut Gradient w.r.t. forward()'s output [seqQ, dHead].
	 * @param dQ Gradient w.r.t. inputQ [seqQ, dModel].
	 * @param dK Gradient w.r.t. inputK [seqK, dModel].
	 * @param dV Gradient w.r.t. inputV [seqK, dModel].
	 */
	void backward(const Tensor<T>& dOut, Tensor<T>& dQ, Tensor<T>& dK, Tensor<T>& dV) {
		if (attnWeights.shape.size() != 2) {
			throw InvalidSizeError("AttentionHead::backward() called before forward()!");
		}
		validation::requireShape(dOut, attnWeights.shape[0], dHead, "Attention output gradient");
		// dOut => Wo => dContext
		Tensor<T> dContext;
		Wo.backward(dOut, dContext);

		// dContext = d(Attn * V)
		// dV = Attn^T * dContext
		Tensor<T> attnT;
		transpose2D(attnWeights, attnT);
		Tensor<T> dVLocal;
		MatMul2D(attnT, dContext, dVLocal);

		// dAttn = dContext * V^T
		Tensor<T> VT;
		transpose2D(V, VT);
		// [Seq, Seq]
		Tensor<T> dAttn;
		MatMul2D(dContext, VT, dAttn);

		// dScores = dSoftmax(dAttn), using softmax Jacobian per row:
		// dS_ij = A_ij * (dA_ij - sum_k(A_ik * dA_ik))
		Tensor<T> dScores = dAttn;
		std::size_t rows = dAttn.shape[0];
		std::size_t cols = dAttn.shape[1];

		for (std::size_t i = 0; i < rows; i++) {
			T sumAda = T(0);
			for (std::size_t j = 0; j < cols; j++) {
				sumAda += attnWeights.data[i * cols + j] * dAttn.data[i * cols + j];
			}

			for (std::size_t j = 0; j < cols; j++) {
				T a = attnWeights.data[i * cols + j];
				dScores.data[i * cols + j] = a * (dAttn.data[i * cols + j] - sumAda);
			}
		}

		// Undo 1/sqrt(dHead) scaling applied to scores in forward().
		T scale = T(1) / std::sqrt((T)dHead);
		for (auto& x : dScores.data) {
			x *= scale;
		}

		// dScores = Q * K^T
		// dQ = dScores * K
		Tensor<T> dQOut;
		MatMul2D(dScores, K, dQOut);

		// dK = dScores^T * Q
		Tensor<T> dScoresT;
		transpose2D(dScores, dScoresT);
		Tensor<T> dKOut;
		MatMul2D(dScoresT, Q, dKOut);

		// Q and K were rotated by RoPE after their projections. dQOut and dKOut
		// are gradients with respect to rotated tensors, so rotate them back
		// (the transpose of a rotation is its inverse) before they reach
		// Wq/Wk, which produced unrotated Q/K.
		if (undoRope) {
			undoRope(dQOut);
			undoRope(dKOut);
		}

		// Backpropagate through Q/K/V projections; this also
		// accumulates their weight gradients.
		Wq.backward(dQOut, dQ);
		Wk.backward(dKOut, dK);
		Wv.backward(dVLocal, dV);
	}

	/**
	 * @brief Applies accumulated gradients to all four projections.
	 *
	 * @param lr Learning rate.
	 * @param rule Optimizer to use; see UpdateRule.
	 */
	void update(T lr, UpdateRule rule = UpdateRule::sgd()) {
		Wq.update(lr, rule);
		Wk.update(lr, rule);
		Wv.update(lr, rule);
		Wo.update(lr, rule);
	}

	/**
	 * @brief Clears accumulated gradients of all four projections.
	 */
	void zeroGrad() {
		Wq.zeroGrad();
		Wk.zeroGrad();
		Wv.zeroGrad();
		Wo.zeroGrad();
	}

private:
	/**
	 * @brief Query
	 */
	Tensor<T> Q;
	/**
	 * @brief Key
	 */
	Tensor<T> K;
	/**
	 * @brief Value
	 */
	Tensor<T> V;

	Tensor<T> scores;

	/**
	 * @brief Set by forward() when RoPE was applied to Q and K; empty otherwise.
	 * backward() uses it to rotate Q/K gradients back, because Q and K
	 * are cached (and used for gradients) in their rotated form.
	 */
	std::function<void(Tensor<T>&)> undoRope;
};