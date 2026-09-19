#pragma once

#include "AttentionHead.h"
#include "FeedForward.h"
#include "RMSNorm.h"
#include "Embedding.h"

/**
 * @brief One BERT encoder layer (pre-norm): bidirectional self-attention
 * followed by a feed-forward network, each with a residual connection.
 * x = x + Attn(Norm(x))
 * x = x + FF(Norm(x))
 *
 * "Bidirectional" means attention is unmasked, so every token sees the
 * whole sequence, before and after it.
 *
 * @remark Stateful: forward() caches its activations and backward() consumes
 * them, so call backward() right after forward() it belongs to.
 */
template <typename T>
class BertLayer {
private:
	// Unmasked self-attention sublayer.
	AttentionHead<T> attn;

	/**
	 * @brief Cache: attention sublayer's output from last forward().
	 */
	Tensor<T> attnOut;

public:
	// Normalization before attention.
	RMSNorm<T> norm1;
	// Position-wise feed-forward sublayer.
	FeedForward<T> ff;
	// Normalization before feed-forward sublayer.
	RMSNorm<T> norm2;

	/**
	 * @param dModel Model width.
	 * @param dFF Inner width of feed-forward sublayer.
	 * @param rng Engine initial weights are drawn from.
	 */
	BertLayer(std::size_t dModel, std::size_t dFF, RandomEngine& rng)
		: attn(dModel, rng), norm1(dModel), ff(dModel, dFF, rng), norm2(dModel) {}

	/**
	 * @brief Clears accumulated gradients of every sublayer.
	 */
	void zeroGrad() {
		attn.zeroGrad();
		norm1.zeroGrad();
		ff.zeroGrad();
		norm2.zeroGrad();
	}

	/**
	 * @brief Forward pass through both sublayers.
	 *
	 * @param x Input [seq, dModel].
	 * @param out Output [seq, dModel].
	 */
	void forward(const Tensor<T>& x, Tensor<T>& out) {
		// 1. Self-Attention.
		// BERT uses Post-Norm usually, but Pre-Norm is also stable.
		// Using Pre-Norm => x = x + Attn(Norm(x))
		Tensor<T> n1;
		norm1.forward(x, n1);

		// No mask for bidirectional.
		// No Rotary Encoding needed for attention head.
		// Self-Attention: Passing same sequence 3 times to find relationships:
		// 1) Query (n1): What is this token looking for? 
		//    e.g "I am bank" looking for 'river' bank or 'money' bank
		// 2) Key (n1)  : What does this token define?
		//    e.g "I am 'river', I define a body of water."
		// 3) Value (n1): What information do I pass?
		//    e.g vector content of 'river'
		Tensor<T> aOut;
		// Omitting RoPE/masked args (rather than passing nullptr, false
		// explicitly) lets forward()'s own defaults apply; passing a bare
		// nullptr can't deduce forward()'s RopeConfig template parameter.
		attn.forward(n1 /* Q */, n1 /* K */, n1 /* V */, aOut);

		// Residual
		// Residual Connection (Skip Connection) is a critical technique used in DL
		// popularized by ResNet to allow training of very deep networks.
		// attn.forward calculated change or 'delta' = aOut
		// Instead of passing data through Attention layer and replacing it,
		// take original input x and add it to result aOut. 
		// Why it is important:
		// 1. Gradient Flow (Superhighway) makes gradients vanished (becomes zero) as there
		// are too many multiplication within layers during backpropagation.
		// "plus" sign allows gradients to flow backwards through network directly bypassing
		// complex attention math if needed.
		// 2. Learning Deltas: It is easier for a model to learn "how to modify existing information"
		// Calculating a small change/residual than to learn "how to recreate information from
		// scratch" at every layer.
		// 3. Identity Mapping: If a layer is unnecessary, model can simply set weights to zero
		// and output becomes x = x + 0. This allways network to ignore layers if they are not
		// helping.
		attnOut = Tensor<T>(x.shape);
		for (std::size_t i = 0; i < x.size(); i++) {
			// Pre-Norm Design Pattern
			// x = x + Attn(Norm(x))
			attnOut[i] = x[i] + aOut[i];
		}

		// 2. Feedforward: normalize attention residual, run FF, and
		// add its output back onto it.
		Tensor<T> n2;
		norm2.forward(attnOut, n2);

		Tensor<T> fOut;
		ff.forward(n2, fOut);

		// Residual
		out = Tensor<T>(x.shape);
		for (std::size_t i = 0; i < x.size(); i++) {
			out[i] = attnOut[i] + fOut[i];
		}
	}

	/**
	 * @brief Backward pass, mirroring forward()'s structure in reverse.
	 * Residual 2: out = attnOut + fOut, so dAttnOut and dFOut both start
	 * as dOut before norm2/ff distribute their own gradients.
	 */
	void backward(const Tensor<T>& dOut, Tensor<T>& dIn) {
		// Backprop FF
		// d_f_out passed directly
		Tensor<T> dn2;
		ff.backward(dOut, dn2); 

		Tensor<T> dAttnOutNorm;
		norm2.backward(dn2, dAttnOutNorm);

		// Sum gradients involved in attn_out (Residual path + Norm path)
		Tensor<T> dAttnOut(dOut.shape);
		for (std::size_t i = 0; i < dOut.size(); i++) {
			dAttnOut[i] = dOut[i] + dAttnOutNorm[i];
		}

		// Backprop Attn
		// attn.backward expects dOut relative to attn output
		Tensor<T> dQ, dK, dV;
		attn.backward(dAttnOut, dQ, dK, dV);

		// Sum Q, K, V grads: input to Attn was n1 for all three, so the
		// gradient w.r.t. n1 is sum of gradients w.r.t. each.
		Tensor<T> dn1(dQ.shape, 0);
		for (std::size_t i = 0; i < dQ.size(); i++) {
			dn1[i] = dQ[i] + dK[i] + dV[i];
		}

		Tensor<T> dxNorm;
		norm1.backward(dn1, dxNorm);

		// Residual 1: attn_out = x + a_out
		// dIn = dAttnOut (from residual) + dxNorm
		dIn = Tensor<T>(dOut.shape);
		for (std::size_t i = 0; i < dOut.size(); i++) {
			dIn[i] = dAttnOut[i] + dxNorm[i];
		}
	}

	/**
	 * @brief Applies accumulated gradients to every sublayer.
	 *
	 * @param lr Learning rate.
	 * @param rule Optimizer to use; see UpdateRule.
	 */
	void update(T lr, UpdateRule rule) {
		attn.update(lr, rule);
		ff.update(lr, rule);
		norm1.update(lr, rule);
		norm2.update(lr, rule);
	}
};