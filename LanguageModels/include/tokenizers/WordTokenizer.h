#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "Exceptions.h"
#include "Validation.h"
#include "TextNormalizer.h"

/**
 * @brief Word-level tokenizer: every distinct whitespace-separated word gets
 * an integer id.
 *
 * vocabulary starts with a fixed list of special tokens (for example
 * <UNK>, <SOS>, <EOS> for translation, or [PAD], [CLS], [SEP], [MASK] for BERT),
 * whose ids are their positions in that list, followed by words seen
 * during fit() in order of first appearance. Words never seen while fitting
 * map to unknown token, so a model can be tested on text with new words.
 *
 * Fit it on training data only: a vocabulary built from test data too
 * would leak test set into training.
 */
class WordTokenizer {
public:
	/**
	 * @param specialTokens Tokens that always exist, with ids 0, 1, 2, ... in
	 * order given.
	 * @param unknownToken Token used for words outside vocabulary; must be
	 * one of specialTokens.
	 * @throws InvalidParameterError If unknownToken is not in specialTokens, or
	 * a special token appears twice.
	 */
	explicit WordTokenizer(std::vector<std::string> specialTokens, std::string unknownToken)
		: unknownWord(std::move(unknownToken)) {
		for (const std::string& token : specialTokens) {
			if (wordToId.count(token)) {
				throw InvalidParameterError("WordTokenizer: Special token '" + token + "' is listed twice!");
			}
			addWord(token);
		}
		if (!wordToId.count(unknownWord)) {
			throw InvalidParameterError("WordTokenizer: unknown token must be one of special tokens!");
		}
		specialCount = idToWord.size();
	}

	/**
	 * @brief Builds vocabulary from sentences, replacing any earlier one
	 * (the special tokens are kept).
	 *
	 * @param sentences Sentences to learn words from, split on whitespace.
	 * Normalize them first (text::normalize) if text is raw.
	 * @param maxVocabSize Largest vocabulary, special tokens included; when
	 * there are more words, most frequent ones are kept (earlier words win
	 * ties). 0 keeps every word.
	 * @throws InvalidParameterSizeError If maxVocabSize is nonzero but smaller
	 * than number of special tokens.
	 */
	void fit(const std::vector<std::string>& sentences, std::size_t maxVocabSize = 0) {
		if (maxVocabSize != 0 && maxVocabSize < specialCount) {
			throw InvalidParameterSizeError("WordTokenizer: maxVocabSize is smaller than special tokens!");
		}

		// Count words, remembering order they first appear in.
		class WordCount {
		public:
			std::string word;
			std::size_t count;
			std::size_t firstSeen;
		};
		std::map<std::string, std::size_t> position;
		std::vector<WordCount> words;
		for (const std::string& sentence : sentences) {
			for (const std::string& word : text::splitOnWhitespace(sentence)) {
				auto found = position.find(word);
				if (found == position.end()) {
					position[word] = words.size();
					words.push_back({ word, 1, words.size() });
				}
				else {
					words[found->second].count++;
				}
			}
		}

		// Too many words: keep most frequent, then restore first-seen order.
		if (maxVocabSize != 0 && words.size() > maxVocabSize - specialCount) {
			std::stable_sort(words.begin(), words.end(), [](const WordCount& a, const WordCount& b) {
				return a.count > b.count;
			});
			words.resize(maxVocabSize - specialCount);
			std::sort(words.begin(), words.end(), [](const WordCount& a, const WordCount& b) {
				return a.firstSeen < b.firstSeen;
			});
		}

		// Keep special tokens, then add words (a special token that also
		// occurs as a word in text stays a single entry).
		idToWord.resize(specialCount);
		for (auto it = wordToId.begin(); it != wordToId.end();) {
			it = (it->second >= specialCount) ? wordToId.erase(it) : std::next(it);
		}
		for (const WordCount& entry : words) {
			if (!wordToId.count(entry.word)) {
				addWord(entry.word);
			}
		}
	}

	/**
	 * @brief Number of ids: special tokens plus fitted words.
	 */
	std::size_t size() const { return idToWord.size(); }

	/**
	 * @brief Id of a word, or of unknown token if it is not in vocabulary.
	 */
	std::size_t id(const std::string& word) const {
		auto found = wordToId.find(word);
		return found == wordToId.end() ? wordToId.at(unknownWord) : found->second;
	}

	/**
	 * @brief Id of a token that must exist (typically a special token such as
	 * <SOS>).
	 *
	 * @throws InvalidParameterError If token is not in vocabulary.
	 */
	std::size_t requireId(const std::string& token) const {
		auto found = wordToId.find(token);
		if (found == wordToId.end()) {
			throw InvalidParameterError("WordTokenizer: '" + token + "' is not in vocabulary!");
		}
		return found->second;
	}

	/**
	 * @brief word an id stands for.
	 *
	 * @throws InvalidParameterError If id is outside vocabulary.
	 */
	const std::string& word(std::size_t tokenId) const {
		validation::requireBelow(tokenId, idToWord.size(), "Token id");
		return idToWord[tokenId];
	}

	/**
	 * @brief Converts a sentence to word ids (unknown words become unknown id).
	 */
	std::vector<std::size_t> encode(const std::string& sentence) const {
		std::vector<std::size_t> ids;
		for (const std::string& word : text::splitOnWhitespace(sentence)) {
			ids.push_back(id(word));
		}
		return ids;
	}

	/**
	 * @brief Converts ids back to words.
	 *
	 * @param ids Token ids.
	 * @param skipSpecial Leave out special tokens such as <SOS> and <EOS>.
	 * @throws InvalidParameterError If an id is outside vocabulary.
	 */
	std::vector<std::string> decode(const std::vector<std::size_t>& ids, bool skipSpecial = true) const {
		std::vector<std::string> words;
		for (std::size_t tokenId : ids) {
			validation::requireBelow(tokenId, idToWord.size(), "Token id");
			if (skipSpecial && tokenId < specialCount) {
				continue;
			}
			words.push_back(idToWord[tokenId]);
		}
		return words;
	}

	/**
	 * @brief Whether an id belongs to a special token.
	 */
	bool isSpecial(std::size_t tokenId) const { return tokenId < specialCount; }

private:
	std::map<std::string, std::size_t> wordToId;
	std::vector<std::string> idToWord;
	std::string unknownWord;
	std::size_t specialCount = 0;

	void addWord(const std::string& word) {
		wordToId[word] = idToWord.size();
		idToWord.push_back(word);
	}
};
