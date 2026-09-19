#include "pch.h"
#include "Tensor.h"
#include <limits>

TEST(TensorTest, DefaultConstructedIsEmpty) {
    Tensor<double> tensor;

    EXPECT_EQ(tensor.size(), 0u);
    EXPECT_TRUE(tensor.shape.empty());
}

TEST(TensorTest, ShapeDeterminesSizeAndDefaultFillIsZero) {
    Tensor<double> tensor({ 2, 3 });

    EXPECT_EQ(tensor.size(), 6u);
    EXPECT_EQ(tensor.shape, (std::vector<std::size_t>{ 2, 3 }));
    for (std::size_t i = 0; i < tensor.size(); i++) {
        EXPECT_DOUBLE_EQ(tensor[i], 0.0);
    }
}

TEST(TensorTest, FillValueIsApplied) {
    Tensor<double> tensor({ 4 }, 2.5);

    ASSERT_EQ(tensor.size(), 4u);
    for (std::size_t i = 0; i < tensor.size(); i++) {
        EXPECT_DOUBLE_EQ(tensor[i], 2.5);
    }
}

TEST(TensorTest, ThreeDimensionalShapeMultipliesAllDimensions) {
    Tensor<float> tensor({ 2, 3, 4 });

    EXPECT_EQ(tensor.size(), 24u);
}

TEST(TensorTest, ZeroDimensionGivesEmptyData) {
    Tensor<double> tensor({ 3, 0 });

    EXPECT_EQ(tensor.size(), 0u);
    EXPECT_EQ(tensor.shape.size(), 2u);
}

TEST(TensorTest, IndexOperatorReadsAndWritesFlatElements) {
    Tensor<double> tensor({ 2, 2 });

    tensor[3] = 7.0;

    EXPECT_DOUBLE_EQ(tensor.data[3], 7.0);
    const Tensor<double>& view = tensor;
    EXPECT_DOUBLE_EQ(view[3], 7.0);
}

TEST(TensorTest, CopyIsIndependentOfOriginal) {
    Tensor<double> original({ 2 }, 1.0);
    Tensor<double> copy = original;

    copy[0] = 9.0;

    EXPECT_DOUBLE_EQ(original[0], 1.0);
}

TEST(TensorTest, SizeOverflowThrowsInvalidSizeError) {
    const std::size_t big = std::numeric_limits<std::size_t>::max() / 2 + 1;

    EXPECT_THROW(Tensor<double>({ big, 4 }), InvalidSizeError);
}
