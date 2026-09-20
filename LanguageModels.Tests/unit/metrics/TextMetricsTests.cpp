#include "pch.h"
#include "TextMetrics.h"
#include <cmath>

using evaluation::Tokens;
using evaluation::splitWords;

namespace {
    // The worked example of Papineni et al., "BLEU: a Method for Automatic
    // Evaluation of Machine Translation" (ACL 2002), section 2.
    const char* paperReference1 = "It is a guide to action that ensures that the military will forever heed Party commands.";
    const char* paperReference2 = "It is the guiding principle which guarantees the military forces always being under the command of the Party.";
    const char* paperReference3 = "It is the practical guide for the army always to heed the directions of the party.";
    const char* paperCandidate1 = "It is a guide to action which ensures that the military always obeys the commands of the party.";
    const char* paperCandidate2 = "It is to insure the troops forever hearing the activity guidebook that party direct.";

    std::vector<Tokens> paperReferences() {
        return { splitWords(paperReference1), splitWords(paperReference2), splitWords(paperReference3) };
    }
}

// ------------------------------------------------------------------- splitWords

TEST(SplitWordsTest, LowerCasesAndTrimsPunctuation) {
    EXPECT_EQ(splitWords("The Party, (it said)."), (Tokens{ "the", "party", "it", "said" }));
}

TEST(SplitWordsTest, KeepsInnerPunctuationAndSkipsExtraWhitespace) {
    EXPECT_EQ(splitWords("  don't   stop\t-- now \n"), (Tokens{ "don't", "stop", "now" }));
}

TEST(SplitWordsTest, EmptyAndPunctuationOnlyTextGiveNoWords) {
    EXPECT_TRUE(splitWords("").empty());
    EXPECT_TRUE(splitWords(" ... !? ").empty());
}

// ----------------------------------------------------------------- edit distance

TEST(EditDistanceTest, KittenBecomesSittingInThreeEdits) {
    // The standard textbook example: k->s, e->i, insert g.
    EXPECT_EQ(evaluation::editDistance(std::string("kitten"), std::string("sitting")), 3u);
}

TEST(EditDistanceTest, HandlesEmptyAndIdenticalSequences) {
    EXPECT_EQ(evaluation::editDistance(std::string(""), std::string("abc")), 3u);
    EXPECT_EQ(evaluation::editDistance(std::string("abc"), std::string("")), 3u);
    EXPECT_EQ(evaluation::editDistance(std::string("abc"), std::string("abc")), 0u);
    EXPECT_EQ(evaluation::editDistance(std::string(""), std::string("")), 0u);
}

TEST(EditDistanceTest, IsSymmetric) {
    EXPECT_EQ(evaluation::editDistance(std::string("flaw"), std::string("lawn")),
        evaluation::editDistance(std::string("lawn"), std::string("flaw")));
    EXPECT_EQ(evaluation::editDistance(std::string("flaw"), std::string("lawn")), 2u);
}

TEST(EditDistanceTest, WorksOnAnySequenceOfComparableItems) {
    std::vector<int> a = { 1, 2, 3, 4 };
    std::vector<int> b = { 1, 3, 4, 5 };

    EXPECT_EQ(evaluation::editDistance(a, b), 2u);
}

// ----------------------------------------------------------- word / character error rate

TEST(WordErrorRateTest, CountsSubstitutionsInsertionsAndDeletions) {
    // "sat" -> "sit" (substitution) and "the" deleted before "mat": 2 edits / 6 words.
    Tokens reference = splitWords("the cat sat on the mat");
    Tokens hypothesis = splitWords("the cat sit on mat");

    EXPECT_DOUBLE_EQ(evaluation::wordErrorRate(reference, hypothesis), 2.0 / 6.0);
}

TEST(WordErrorRateTest, PerfectMatchIsZeroAndAnOverlongHypothesisCanExceedOne) {
    Tokens reference = splitWords("hello world");

    EXPECT_DOUBLE_EQ(evaluation::wordErrorRate(reference, reference), 0.0);
    EXPECT_DOUBLE_EQ(evaluation::wordErrorRate(reference, splitWords("hello there big wide world")), 1.5);
}

TEST(WordErrorRateTest, EmptyReferenceThrows) {
    EXPECT_THROW(evaluation::wordErrorRate({}, splitWords("hello")), DivisionByZeroError);
}

TEST(CharacterErrorRateTest, IsCharacterEditsOverReferenceLength) {
    EXPECT_DOUBLE_EQ(evaluation::characterErrorRate("kitten", "sitting"), 3.0 / 6.0);
    EXPECT_THROW(evaluation::characterErrorRate("", "x"), DivisionByZeroError);
}

// ------------------------------------------------------------------------- BLEU

TEST(BleuTest, ModifiedPrecisionClipsRepeatedWords) {
    // Paper: "the the the the the the the" against two references has modified
    // unigram precision 2/7, because "the" appears at most twice in one reference.
    Tokens candidate = splitWords("the the the the the the the");
    std::vector<Tokens> references = { splitWords("the cat is on the mat"),
                                       splitWords("there is a cat on the mat") };

    auto score = evaluation::sentenceBleu(candidate, references, 1);

    ASSERT_EQ(score.precisions.size(), 1u);
    EXPECT_DOUBLE_EQ(score.precisions[0], 2.0 / 7.0);
    EXPECT_NEAR(score.bleu, 2.0 / 7.0, 1e-12);
}

TEST(BleuTest, ReproducesThePapersPrecisionsForTheGoodCandidate) {
    auto score = evaluation::sentenceBleu(splitWords(paperCandidate1), paperReferences());

    // Paper: unigram precision 17/18 and bigram precision 10/17.
    ASSERT_EQ(score.precisions.size(), 4u);
    EXPECT_DOUBLE_EQ(score.precisions[0], 17.0 / 18.0);
    EXPECT_DOUBLE_EQ(score.precisions[1], 10.0 / 17.0);
    EXPECT_DOUBLE_EQ(score.precisions[2], 7.0 / 16.0);
    EXPECT_DOUBLE_EQ(score.precisions[3], 4.0 / 15.0);
    EXPECT_DOUBLE_EQ(score.brevityPenalty, 1.0);
}

TEST(BleuTest, ReproducesThePapersPrecisionsForTheBadCandidate) {
    auto score = evaluation::sentenceBleu(splitWords(paperCandidate2), paperReferences());

    // Paper: unigram precision 8/14 and bigram precision 1/13.
    EXPECT_DOUBLE_EQ(score.precisions[0], 8.0 / 14.0);
    EXPECT_DOUBLE_EQ(score.precisions[1], 1.0 / 13.0);
    // No matching trigram exists, so plain BLEU-4 is exactly zero.
    EXPECT_DOUBLE_EQ(score.precisions[2], 0.0);
    EXPECT_DOUBLE_EQ(score.bleu, 0.0);
}

TEST(BleuTest, GoodCandidateScoresFarAboveBadCandidate) {
    // Values checked against an independent implementation.
    auto good = evaluation::sentenceBleu(splitWords(paperCandidate1), paperReferences());
    auto bad = evaluation::sentenceBleu(splitWords(paperCandidate2), paperReferences(), 4, true);

    EXPECT_NEAR(good.bleu, 0.5045666840058485, 1e-12);
    EXPECT_NEAR(bad.bleu, 0.13111209575157434, 1e-12);
    EXPECT_GT(good.bleu, bad.bleu);
}

TEST(BleuTest, CorpusBleuSumsCountsInsteadOfAveragingSentences) {
    std::vector<Tokens> candidates = { splitWords(paperCandidate1), splitWords(paperCandidate2) };
    std::vector<std::vector<Tokens>> references = { paperReferences(), paperReferences() };

    auto score = evaluation::corpusBleu(candidates, references);

    // Pooled: unigrams (17 + 8) / (18 + 14), bigrams (10 + 1) / (17 + 13), ...
    EXPECT_DOUBLE_EQ(score.precisions[0], 25.0 / 32.0);
    EXPECT_DOUBLE_EQ(score.precisions[1], 11.0 / 30.0);
    EXPECT_NEAR(score.bleu, 0.3043537261305561, 1e-12);
    EXPECT_EQ(score.candidateLength, 32u);
}

TEST(BleuTest, IdenticalSentenceScoresOne) {
    Tokens sentence = splitWords("the quick brown fox jumps over the lazy dog");

    auto score = evaluation::sentenceBleu(sentence, { sentence });

    EXPECT_DOUBLE_EQ(score.bleu, 1.0);
    EXPECT_DOUBLE_EQ(score.brevityPenalty, 1.0);
}

TEST(BleuTest, ShortCandidatesArePenalisedByTheBrevityPenalty) {
    Tokens candidate = splitWords("the cat is on the");
    Tokens reference = splitWords("the cat is on the mat");

    auto score = evaluation::sentenceBleu(candidate, { reference });

    // Every n-gram of the candidate matches (precision 1), so BLEU is exactly the penalty.
    EXPECT_DOUBLE_EQ(score.brevityPenalty, std::exp(1.0 - 6.0 / 5.0));
    EXPECT_DOUBLE_EQ(score.bleu, std::exp(1.0 - 6.0 / 5.0));
    EXPECT_EQ(score.referenceLength, 6u);
}

TEST(BleuTest, BrevityPenaltyUsesTheClosestReferenceLength) {
    Tokens candidate = splitWords("a b c d");
    // Reference lengths 2 and 5 are 2 and 1 away from 4, so 5 is the closest.
    std::vector<Tokens> references = { splitWords("a b"), splitWords("a b c d e") };

    auto score = evaluation::sentenceBleu(candidate, references, 1);

    EXPECT_EQ(score.referenceLength, 5u);
    EXPECT_DOUBLE_EQ(score.brevityPenalty, std::exp(1.0 - 5.0 / 4.0));
}

TEST(BleuTest, ATieBetweenReferenceLengthsPicksTheShorter) {
    Tokens candidate = splitWords("a b c d");
    std::vector<Tokens> references = { splitWords("a b c d e"), splitWords("a b c") };

    auto score = evaluation::sentenceBleu(candidate, references, 1);

    EXPECT_EQ(score.referenceLength, 3u);
    EXPECT_DOUBLE_EQ(score.brevityPenalty, 1.0);
}

TEST(BleuTest, AnEmptyCandidateScoresZero) {
    auto score = evaluation::sentenceBleu({}, { splitWords("the cat") });

    EXPECT_DOUBLE_EQ(score.bleu, 0.0);
}

TEST(BleuTest, RejectsInvalidInput) {
    std::vector<std::vector<Tokens>> oneReferenceSet = { paperReferences() };

    EXPECT_THROW(evaluation::corpusBleu({}, {}), InvalidSizeError);
    EXPECT_THROW(evaluation::corpusBleu({ splitWords("a") }, {}), InvalidSizeError);
    EXPECT_THROW(evaluation::corpusBleu({ splitWords("a") }, { {} }), InvalidSizeError);
    EXPECT_THROW(evaluation::corpusBleu({ splitWords("a"), splitWords("b") }, oneReferenceSet), InvalidSizeError);
    EXPECT_THROW(evaluation::corpusBleu({ splitWords("a") }, oneReferenceSet, 0), InvalidParameterSizeError);
}

// ------------------------------------------------------------------------ ROUGE

TEST(RougeTest, RougeOneCountsSharedWords) {
    Tokens candidate = splitWords("the cat was found under the bed");
    Tokens reference = splitWords("the cat was under the bed");

    auto score = evaluation::rougeN(candidate, reference, 1);

    EXPECT_DOUBLE_EQ(score.precision, 6.0 / 7.0);
    EXPECT_DOUBLE_EQ(score.recall, 1.0);
    EXPECT_DOUBLE_EQ(score.f1, 12.0 / 13.0);
}

TEST(RougeTest, RougeTwoCountsSharedWordPairs) {
    Tokens candidate = splitWords("the cat was found under the bed");
    Tokens reference = splitWords("the cat was under the bed");

    auto score = evaluation::rougeN(candidate, reference, 2);

    // Shared pairs: "the cat", "cat was", "under the", "the bed".
    EXPECT_DOUBLE_EQ(score.precision, 4.0 / 6.0);
    EXPECT_DOUBLE_EQ(score.recall, 4.0 / 5.0);
}

TEST(RougeTest, RepeatedWordsAreClippedToTheReferenceCount) {
    auto score = evaluation::rougeN(splitWords("the the the"), splitWords("the cat"), 1);

    EXPECT_DOUBLE_EQ(score.precision, 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(score.recall, 1.0 / 2.0);
}

TEST(RougeTest, RougeLUsesTheLongestCommonSubsequence) {
    Tokens candidate = splitWords("the cat was found under the bed");
    Tokens reference = splitWords("the cat was under the bed");

    auto score = evaluation::rougeL(candidate, reference);

    // "the cat was under the bed" is a subsequence of the candidate: length 6.
    EXPECT_DOUBLE_EQ(score.precision, 6.0 / 7.0);
    EXPECT_DOUBLE_EQ(score.recall, 1.0);
}

TEST(RougeTest, RougeLRewardsOrderThatRougeOneIgnores) {
    Tokens reference = splitWords("a b c d");
    Tokens shuffled = splitWords("d c b a");

    EXPECT_DOUBLE_EQ(evaluation::rougeN(shuffled, reference, 1).f1, 1.0);
    EXPECT_DOUBLE_EQ(evaluation::rougeL(shuffled, reference).f1, 0.25);
}

TEST(RougeTest, DisjointOrEmptyTextScoresZero) {
    auto disjoint = evaluation::rougeN(splitWords("a b"), splitWords("c d"), 1);
    auto empty = evaluation::rougeL({}, splitWords("c d"));

    EXPECT_DOUBLE_EQ(disjoint.f1, 0.0);
    EXPECT_DOUBLE_EQ(empty.precision, 0.0);
    EXPECT_DOUBLE_EQ(empty.recall, 0.0);
    EXPECT_DOUBLE_EQ(empty.f1, 0.0);
}

TEST(RougeTest, ShorterThanNGramGivesZeroInsteadOfDividingByZero) {
    auto score = evaluation::rougeN(splitWords("a"), splitWords("a b"), 2);

    EXPECT_DOUBLE_EQ(score.f1, 0.0);
}

TEST(RougeTest, ZeroOrderThrows) {
    EXPECT_THROW(evaluation::rougeN(splitWords("a"), splitWords("a"), 0), InvalidParameterSizeError);
}
