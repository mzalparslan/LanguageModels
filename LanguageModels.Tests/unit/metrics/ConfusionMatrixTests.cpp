#include "pch.h"
#include "ConfusionMatrix.h"

namespace {
    // Three classes, nine examples:
    //   actual    0 0 0 1 1 2 2 2 2
    //   predicted 0 0 1 1 1 2 2 0 2
    // counts (rows actual, columns predicted):
    //        p0 p1 p2
    //   a0    2  1  0
    //   a1    0  2  0
    //   a2    1  0  3
    ConfusionMatrix threeClassMatrix() {
        ConfusionMatrix matrix(3);
        matrix.add({ 0, 0, 0, 1, 1, 2, 2, 2, 2 }, { 0, 0, 1, 1, 1, 2, 2, 0, 2 });
        return matrix;
    }
}

TEST(ConfusionMatrixTest, CountsEachActualPredictedPair) {
    ConfusionMatrix matrix = threeClassMatrix();

    EXPECT_EQ(matrix.total(), 9u);
    EXPECT_EQ(matrix.numClasses(), 3u);
    EXPECT_EQ(matrix.count(0, 0), 2u);
    EXPECT_EQ(matrix.count(0, 1), 1u);
    EXPECT_EQ(matrix.count(1, 1), 2u);
    EXPECT_EQ(matrix.count(2, 0), 1u);
    EXPECT_EQ(matrix.count(2, 2), 3u);
    EXPECT_EQ(matrix.count(1, 0), 0u);
}

TEST(ConfusionMatrixTest, AccuracyIsTheDiagonalShare) {
    EXPECT_DOUBLE_EQ(threeClassMatrix().accuracy(), 7.0 / 9.0);
}

TEST(ConfusionMatrixTest, PerClassPrecisionRecallAndF1) {
    ConfusionMatrix matrix = threeClassMatrix();

    EXPECT_DOUBLE_EQ(matrix.precision(0), 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(matrix.recall(0), 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(matrix.f1(0), 2.0 / 3.0);

    EXPECT_DOUBLE_EQ(matrix.precision(1), 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(matrix.recall(1), 1.0);
    EXPECT_DOUBLE_EQ(matrix.f1(1), 0.8);

    EXPECT_DOUBLE_EQ(matrix.precision(2), 1.0);
    EXPECT_DOUBLE_EQ(matrix.recall(2), 0.75);
    EXPECT_NEAR(matrix.f1(2), 6.0 / 7.0, 1e-15);
}

TEST(ConfusionMatrixTest, MacroAveragesWeightEveryClassEqually) {
    ConfusionMatrix matrix = threeClassMatrix();

    EXPECT_NEAR(matrix.macroPrecision(), (2.0 / 3.0 + 2.0 / 3.0 + 1.0) / 3.0, 1e-15);
    EXPECT_NEAR(matrix.macroRecall(), (2.0 / 3.0 + 1.0 + 0.75) / 3.0, 1e-15);
    EXPECT_NEAR(matrix.macroF1(), (2.0 / 3.0 + 0.8 + 6.0 / 7.0) / 3.0, 1e-15);
}

TEST(ConfusionMatrixTest, ABinaryExampleMatchesTheTextbookFormulas) {
    // TP = 40, FN = 10, FP = 5, TN = 45, with class 1 as "positive".
    ConfusionMatrix matrix(2);
    for (int i = 0; i < 40; i++) matrix.add(1, 1);
    for (int i = 0; i < 10; i++) matrix.add(1, 0);
    for (int i = 0; i < 5; i++) matrix.add(0, 1);
    for (int i = 0; i < 45; i++) matrix.add(0, 0);

    EXPECT_DOUBLE_EQ(matrix.accuracy(), 0.85);
    EXPECT_DOUBLE_EQ(matrix.precision(1), 40.0 / 45.0);
    EXPECT_DOUBLE_EQ(matrix.recall(1), 0.8);
    EXPECT_NEAR(matrix.f1(1), 2.0 * (40.0 / 45.0) * 0.8 / (40.0 / 45.0 + 0.8), 1e-15);
}

TEST(ConfusionMatrixTest, AClassNeverPredictedOrPresentScoresZeroButStillCountsInMacroAverages) {
    ConfusionMatrix matrix(3);
    matrix.add({ 0, 0, 1, 1 }, { 0, 0, 1, 1 });

    EXPECT_DOUBLE_EQ(matrix.accuracy(), 1.0);
    EXPECT_DOUBLE_EQ(matrix.precision(2), 0.0);
    EXPECT_DOUBLE_EQ(matrix.recall(2), 0.0);
    EXPECT_DOUBLE_EQ(matrix.f1(2), 0.0);
    EXPECT_DOUBLE_EQ(matrix.macroF1(), 2.0 / 3.0);
}

TEST(ConfusionMatrixTest, RejectsInvalidInput) {
    EXPECT_THROW(ConfusionMatrix(0), InvalidParameterSizeError);

    ConfusionMatrix matrix(2);
    EXPECT_THROW(matrix.accuracy(), DivisionByZeroError);
    EXPECT_THROW(matrix.add(2, 0), InvalidParameterError);
    EXPECT_THROW(matrix.add(0, 2), InvalidParameterError);
    EXPECT_THROW(matrix.add({ 0, 1 }, { 0 }), InvalidSizeError);
    EXPECT_THROW(matrix.count(0, 2), InvalidParameterError);
    EXPECT_THROW(matrix.precision(2), InvalidParameterError);
    EXPECT_THROW(matrix.recall(2), InvalidParameterError);
}
