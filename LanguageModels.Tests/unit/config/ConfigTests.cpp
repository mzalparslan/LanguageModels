#include "pch.h"
#include "ModelConfig.h"
#include "BasicGPTConfig.h"
#include "GPTWithUnigramConfig.h"
#include "MiniTransformerConfig.h"
#include <cstddef>

// The configs feed compile-time table sizes (RoPE), so the relationships they
// promise are checked here as well as by static_asserts in the headers.

TEST(ModelConfigTest, HeadDimensionIsModelWidthDividedByHeads) {
    EXPECT_EQ(ModelConfig::d_head, ModelConfig::d_model / ModelConfig::h);
    EXPECT_EQ(ModelConfig::d_head * ModelConfig::h, ModelConfig::d_model);
}

TEST(ModelConfigTest, FeedForwardWidthIsFourTimesModelWidth) {
    EXPECT_EQ(ModelConfig::d_ff, 4 * ModelConfig::d_model);
}

TEST(ModelConfigTest, WidthsAreEvenAsRotaryEmbeddingRequires) {
    EXPECT_EQ(ModelConfig::d_model % 2, 0u);
    EXPECT_EQ(ModelConfig::d_head % 2, 0u);
}

TEST(ModelConfigTest, MixtureOfExpertsDefaultsAreConsistent) {
    EXPECT_GT(ModelConfig::numExpertsDefault, 0u);
    EXPECT_GT(ModelConfig::topKExperts, 0u);
    EXPECT_LE(ModelConfig::topKExperts, ModelConfig::numExpertsDefault);
    EXPECT_GT(ModelConfig::moeLoadBalancingAlpha, 0.0);
}

TEST(ModelConfigTest, TrainingDefaultsArePositive) {
    EXPECT_GT(ModelConfig::learningRate, 0.0);
    EXPECT_GT(ModelConfig::maxSeqLen, 0u);
}

TEST(ModelConfigTest, DefaultSeedIsFortyTwo) {
    EXPECT_EQ(ModelConfig::randomSeed, 42u);
}

TEST(ModelConfigTest, SmallConfigsSatisfyRotaryEmbeddingConstraints) {
    EXPECT_EQ(BasicGPTConfig::d_head % 2, 0u);
    EXPECT_GT(BasicGPTConfig::maxSeqLen, 0u);
    EXPECT_EQ(GPTWithUnigramConfig::d_head % 2, 0u);
    EXPECT_GT(GPTWithUnigramConfig::maxSeqLen, 0u);
    EXPECT_EQ(MiniTransformerConfig::d_head % 2, 0u);
    EXPECT_GT(MiniTransformerConfig::maxSeqLen, 0u);
}

TEST(ModelConfigTest, EveryModelConfigDefaultsToTheSameSeed) {
    EXPECT_EQ(BasicGPTConfig::randomSeed, ModelConfig::randomSeed);
    EXPECT_EQ(GPTWithUnigramConfig::randomSeed, ModelConfig::randomSeed);
}
