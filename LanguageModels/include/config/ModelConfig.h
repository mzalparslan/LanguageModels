#pragma once

#include <cstddef>
#include <cstdint>


/**
 * @brief Shared architecture + training config, passed as template
 * argument (e.g. `template <typename Config = ModelConfig>`).
 * This shared model configuration is to understand which parameters
 * are mostly used in different execution (training or 
*/
class ModelConfig {
public:
	// ------------------------------------------------------------------
	// Sequence / embedding
	// ------------------------------------------------------------------

	/**
	 * @brief Embedding dimension (vector size) of a single token representation.
	 * Fixed to optimize model runtime performance.
	 * Constexpr can be removed if dynamic embedding dimension
	 * needs to be interpreted for model at run-time.
	 *
	 * @remark BERT paper (Devlin et al., 2018) denotes
	 * parameter as <B>H</B> => Hidden Layer Size => Hidden Size.
	 */
	static constexpr std::size_t d_model = 512;
	static_assert(d_model % 2 == 0, "Model dimension should be even!");

	/**
	 * @brief Context length: maximum sequence length model can handle.
	 * Also known as Max Position Embeddings or Context Window Size.
	 * Literature often denotes this parameter as <B>n_ctx</B> or <B>L</B>.
	 */
	static constexpr std::size_t maxSeqLen = 1024;

	// ------------------------------------------------------------------
	// Attention
	// ------------------------------------------------------------------

	/**
	 * @brief Attention heads operating in parallel.
	 *
	 * @remark Original paper (Vaswani et al., 2017) denotes this parameter as <B>h</B>.
	 * BERT paper (Devlin et al., 2018) denotes parameter as <B>A</B>
	 */
	static constexpr std::size_t h = 8;
	static_assert(h != 0, "Number of heads cannot be zero!");
	static_assert(d_model % h == 0, "Embedding dimension should be a factor of num_heads!");

	/**
	 * @brief Size of each attention vectors (Q, K, V) within specific head.
	 */
	static constexpr std::size_t d_head = d_model / h;
	static_assert(d_head % 2 == 0, "Head dimension should be even!");

	// ------------------------------------------------------------------
	// Feed-forward
	// ------------------------------------------------------------------

	/**
	 * @brief Inner hidden size of Feed-Forward networks.
	 * Literature often sets this to 4 times model dimension.
	 * It is denoted as <B>d_ff</B> in Vaswani et al., 2017.
	 * SwiGLU standard calculates as d_ff = 8/3 * d_model.
	 *
	 * @remark As it is easy to distinguish, naming convention d_ff is preserved
	 * while other parameters used self-descriptive names.
	 */
	static constexpr std::size_t d_ff = 4 * d_model;

	// ------------------------------------------------------------------
	// Mixture of Experts
	// ------------------------------------------------------------------

	/**
	 * @brief Number of Experts in Mixture of Experts (MoE) layers.
	 * Small models may use 4-8 experts, while larger models can use 16, 32 or more.
	 * Trade-off:
	 * More experts = Massive Parameter Count (Memory usage)
	 * but sparse compute (Speed stays same).
	 *
	 * @remark Literature often denotes this parameter as <B>E</B> or <B>num_experts</B>.
	 */
	static constexpr std::size_t numExpertsDefault = 8;

	/**
	 * @brief Top-K Experts to select for each input token in MoE layers.
	 *
	 * @remark Top-2 (K=2) often yields better performance.
	 * Top-1 (K=1) is simplest and fastest. Top-1 also called:
	 * "Hard Routing" or
	 * "Single Expert Selection" or
	 * "Greedy Routing" or
	 * "Single Choice Routing" or
	 * "1-of-N Routing" or
	 * "Winner-Takes-All Routing" or
	 * "Single Path Routing" or
	 * "Switch Routing" (in Switch Transformers).
	 */
	static constexpr std::size_t topKExperts = 2;

	/**
	 * @brief Loss weighting factor (Auxillary Loss Coefficient) for load balancing in MoE layers.
	 * Typically a small value between 0.01 or 0.001 is used.
	 * Formula: L_aux = alpha * N * sum(fi * pi) where N = Number of Experts
	 * Gradient: d(L_aux)/d(P(k|b)) = alpha * N * f_k * (1/B)
	 */
	static constexpr double moeLoadBalancingAlpha = 0.01;

	// ------------------------------------------------------------------
	// Optimization
	// ------------------------------------------------------------------

	/**
	 * @brief lr: Learning Rate for optimizer.
	 * Also known as Step Size in optimization literature.
	 * General notation is <B>alpha</B>. Denoted as <B>eta</B> in many optimization papers.
	 *
	 * @remark MoE also uses same alpha value for load balancing loss coefficient.
	 */
	static constexpr double learningRate = 0.01;

	/**
	 * @brief Seed for initial weights. same seed always gives the
	 * same model, in Debug and Release alike.
	 */
	static constexpr std::uint32_t randomSeed = 42;

	// Prevent instantiation.
	ModelConfig() = delete;
};
