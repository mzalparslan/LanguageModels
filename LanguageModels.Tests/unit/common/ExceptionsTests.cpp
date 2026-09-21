#include "pch.h"
#include "Exceptions.h"
#include <string>
#include <type_traits>

// catch-by-base behavior is part of library's contract: callers may
// catch specific type, family, or std base.

TEST(ExceptionsTest, HierarchyMatchesDocumentedDesign) {
    static_assert(std::is_base_of_v<std::domain_error, NonFiniteError>);
    static_assert(std::is_base_of_v<NonFiniteError, NaNError>);
    static_assert(std::is_base_of_v<std::domain_error, DivisionByZeroError>);
    static_assert(std::is_base_of_v<std::invalid_argument, InvalidParameterError>);
    static_assert(std::is_base_of_v<std::invalid_argument, InvalidSizeError>);
    static_assert(std::is_base_of_v<InvalidSizeError, InvalidParameterSizeError>);

    // Value errors and size errors are distinct families.
    static_assert(!std::is_base_of_v<InvalidSizeError, InvalidParameterError>);
    static_assert(!std::is_base_of_v<InvalidParameterError, InvalidSizeError>);
    static_assert(!std::is_base_of_v<NonFiniteError, DivisionByZeroError>);
    SUCCEED();
}

TEST(ExceptionsTest, NaNErrorIsCaughtAsNonFiniteError) {
    try {
        throw NaNError("nan");
        FAIL() << "expected a throw";
    }
    catch (const NonFiniteError& error) {
        EXPECT_STREQ(error.what(), "nan");
    }
}

TEST(ExceptionsTest, NonFiniteFamilyIsCaughtAsDomainError) {
    EXPECT_THROW(throw NonFiniteError("inf"), std::domain_error);
    EXPECT_THROW(throw NaNError("nan"), std::domain_error);
    EXPECT_THROW(throw DivisionByZeroError("zero"), std::domain_error);
}

TEST(ExceptionsTest, ParameterAndSizeErrorsAreCaughtAsInvalidArgument) {
    EXPECT_THROW(throw InvalidParameterError("p"), std::invalid_argument);
    EXPECT_THROW(throw InvalidSizeError("s"), std::invalid_argument);
    EXPECT_THROW(throw InvalidParameterSizeError("ps"), std::invalid_argument);
}

TEST(ExceptionsTest, InvalidParameterSizeErrorIsCaughtAsInvalidSizeError) {
    EXPECT_THROW(throw InvalidParameterSizeError("ps"), InvalidSizeError);
}

TEST(ExceptionsTest, MessageIsPreserved) {
    try {
        throw InvalidParameterError(std::string("learning rate must be positive"));
    }
    catch (const std::exception& error) {
        EXPECT_STREQ(error.what(), "learning rate must be positive");
    }
}
