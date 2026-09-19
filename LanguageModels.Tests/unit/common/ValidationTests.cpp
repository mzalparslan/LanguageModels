#include "pch.h"
#include "Validation.h"
#include <limits>
#include <string>

namespace {
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    const double kInf = std::numeric_limits<double>::infinity();
}

// ------------------------------------------------------------ requireFinite

TEST(ValidationTest, RequireFiniteAcceptsOrdinaryValues) {
    EXPECT_NO_THROW(validation::requireFinite(0.0, "x"));
    EXPECT_NO_THROW(validation::requireFinite(-1e300, "x"));
    EXPECT_NO_THROW(validation::requireFinite(1.5f, "x"));
}

TEST(ValidationTest, RequireFiniteThrowsNaNErrorForNaN) {
    EXPECT_THROW(validation::requireFinite(kNaN, "x"), NaNError);
}

TEST(ValidationTest, RequireFiniteThrowsNonFiniteErrorForInfinity) {
    EXPECT_THROW(validation::requireFinite(kInf, "x"), NonFiniteError);
    EXPECT_THROW(validation::requireFinite(-kInf, "x"), NonFiniteError);
}

TEST(ValidationTest, InfinityIsNotReportedAsNaN) {
    try {
        validation::requireFinite(kInf, "x");
        FAIL() << "expected a throw";
    }
    catch (const NaNError&) {
        FAIL() << "infinity must not be a NaNError";
    }
    catch (const NonFiniteError&) {
        SUCCEED();
    }
}

TEST(ValidationTest, MessageNamesTheOffendingValue) {
    try {
        validation::requireFinite(kNaN, "Learning rate");
        FAIL() << "expected a throw";
    }
    catch (const NaNError& error) {
        EXPECT_NE(std::string(error.what()).find("Learning rate"), std::string::npos);
    }
}

TEST(ValidationTest, RequireAllFiniteScansEveryElement) {
    Tensor<double> good({ 2, 2 }, 1.0);
    EXPECT_NO_THROW(validation::requireAllFinite(good, "tensor"));

    Tensor<double> hasNaN({ 2, 2 }, 1.0);
    hasNaN[3] = kNaN;
    EXPECT_THROW(validation::requireAllFinite(hasNaN, "tensor"), NaNError);

    Tensor<double> hasInf({ 2, 2 }, 1.0);
    hasInf[0] = kInf;
    EXPECT_THROW(validation::requireAllFinite(hasInf, "tensor"), NonFiniteError);
}

// ----------------------------------------------- positive / non-negative floats

TEST(ValidationTest, RequirePositiveFiniteRejectsZeroAndNegative) {
    EXPECT_NO_THROW(validation::requirePositiveFinite(1e-12, "lr"));
    EXPECT_THROW(validation::requirePositiveFinite(0.0, "lr"), InvalidParameterError);
    EXPECT_THROW(validation::requirePositiveFinite(-0.1, "lr"), InvalidParameterError);
}

TEST(ValidationTest, RequirePositiveFiniteRejectsNonFinite) {
    EXPECT_THROW(validation::requirePositiveFinite(kNaN, "lr"), NaNError);
    EXPECT_THROW(validation::requirePositiveFinite(kInf, "lr"), NonFiniteError);
}

TEST(ValidationTest, RequireNonNegativeFiniteAcceptsZero) {
    EXPECT_NO_THROW(validation::requireNonNegativeFinite(0.0, "scale"));
    EXPECT_NO_THROW(validation::requireNonNegativeFinite(2.0, "scale"));
    EXPECT_THROW(validation::requireNonNegativeFinite(-1e-9, "scale"), InvalidParameterError);
    EXPECT_THROW(validation::requireNonNegativeFinite(kNaN, "scale"), NaNError);
    EXPECT_THROW(validation::requireNonNegativeFinite(-kInf, "scale"), NonFiniteError);
}

// ---------------------------------------------------------------- denominators

TEST(ValidationTest, RequireNonZeroDenominatorOnlyRejectsExactZero) {
    EXPECT_NO_THROW(validation::requireNonZeroDenominator(1e-300, "d"));
    EXPECT_NO_THROW(validation::requireNonZeroDenominator(-3.0, "d"));
    EXPECT_THROW(validation::requireNonZeroDenominator(0.0, "d"), DivisionByZeroError);
    EXPECT_THROW(validation::requireNonZeroDenominator(std::size_t(0), "d"), DivisionByZeroError);
}

// ---------------------------------------------------------------------- sizes

TEST(ValidationTest, RequirePositiveSizeRejectsZeroWithParameterSizeError) {
    EXPECT_NO_THROW(validation::requirePositiveSize(1, "width"));
    EXPECT_THROW(validation::requirePositiveSize(0, "width"), InvalidParameterSizeError);
}

TEST(ValidationTest, RequireValidShapeRejectsEmptyAndZeroDimensions) {
    EXPECT_NO_THROW(validation::requireValidShape({ 3 }, "shape"));
    EXPECT_NO_THROW(validation::requireValidShape({ 2, 3, 4 }, "shape"));
    EXPECT_THROW(validation::requireValidShape({}, "shape"), InvalidParameterSizeError);
    EXPECT_THROW(validation::requireValidShape({ 2, 0 }, "shape"), InvalidParameterSizeError);
}

TEST(ValidationTest, RequireDivisibleChecksRemainderAndZeroDivisor) {
    EXPECT_NO_THROW(validation::requireDivisible(64, 8, "width", "heads"));
    EXPECT_THROW(validation::requireDivisible(10, 3, "width", "heads"), InvalidParameterSizeError);
    EXPECT_THROW(validation::requireDivisible(10, 0, "width", "heads"), InvalidParameterSizeError);
}

TEST(ValidationTest, RequireBelowIsAnExclusiveUpperBound) {
    EXPECT_NO_THROW(validation::requireBelow(0, 5, "id"));
    EXPECT_NO_THROW(validation::requireBelow(4, 5, "id"));
    EXPECT_THROW(validation::requireBelow(5, 5, "id"), InvalidParameterError);
    EXPECT_THROW(validation::requireBelow(100, 5, "id"), InvalidParameterError);
}

TEST(ValidationTest, RequireBelowMessageContainsIndexAndLimit) {
    try {
        validation::requireBelow(9, 5, "Token id");
        FAIL() << "expected a throw";
    }
    catch (const InvalidParameterError& error) {
        std::string message = error.what();
        EXPECT_NE(message.find("9"), std::string::npos);
        EXPECT_NE(message.find("5"), std::string::npos);
    }
}

TEST(ValidationTest, RequireNonEmptyRejectsZeroWithInvalidSizeError) {
    EXPECT_NO_THROW(validation::requireNonEmpty(1, "input"));
    EXPECT_THROW(validation::requireNonEmpty(0, "input"), InvalidSizeError);
}

TEST(ValidationTest, RequireAtMostAllowsTheLimitItself) {
    EXPECT_NO_THROW(validation::requireAtMost(10, 10, "length"));
    EXPECT_THROW(validation::requireAtMost(11, 10, "length"), InvalidSizeError);
}

TEST(ValidationTest, RequireSameSizeRequiresEquality) {
    EXPECT_NO_THROW(validation::requireSameSize(4, 4, "vector"));
    EXPECT_THROW(validation::requireSameSize(3, 4, "vector"), InvalidSizeError);
    EXPECT_THROW(validation::requireSameSize(5, 4, "vector"), InvalidSizeError);
}

// ------------------------------------------------------------------- matrices

TEST(ValidationTest, RequireMatrixAcceptsOnlyNonEmpty2D) {
    EXPECT_NO_THROW(validation::requireMatrix(Tensor<double>({ 2, 3 }), "m"));
    EXPECT_THROW(validation::requireMatrix(Tensor<double>({ 3 }), "m"), InvalidSizeError);
    EXPECT_THROW(validation::requireMatrix(Tensor<double>({ 2, 2, 2 }), "m"), InvalidSizeError);
    EXPECT_THROW(validation::requireMatrix(Tensor<double>({ 0, 3 }), "m"), InvalidSizeError);
    EXPECT_THROW(validation::requireMatrix(Tensor<double>({ 3, 0 }), "m"), InvalidSizeError);
    EXPECT_THROW(validation::requireMatrix(Tensor<double>(), "m"), InvalidSizeError);
}

TEST(ValidationTest, RequireColumnsChecksWidth) {
    Tensor<double> m({ 4, 3 });

    EXPECT_NO_THROW(validation::requireColumns(m, 3, "m"));
    EXPECT_THROW(validation::requireColumns(m, 4, "m"), InvalidSizeError);
    EXPECT_THROW(validation::requireColumns(Tensor<double>({ 3 }), 3, "m"), InvalidSizeError);
}

TEST(ValidationTest, RequireShapeChecksBothDimensions) {
    Tensor<double> m({ 4, 3 });

    EXPECT_NO_THROW(validation::requireShape(m, 4, 3, "m"));
    EXPECT_THROW(validation::requireShape(m, 5, 3, "m"), InvalidSizeError);
    EXPECT_THROW(validation::requireShape(m, 4, 2, "m"), InvalidSizeError);
}
