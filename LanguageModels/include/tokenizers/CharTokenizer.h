#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "Validation.h"

/**
 * @brief Character-level tokenizer: every distinct character of training
 * text gets an integer id.
 *
 * Id 0 is reserved for "unknown", so a character that never occurred in the
 * training text (which can happen in a held-out text) still has an id. The
 * characters then get ids 1, 2, 3, ... in ascending byte order.
 */
class CharTokenizer {
public:
	// Id given to characters that were not seen while fitting.
	static constexpr std::size_t unknownId = 0;

	/**
	 * @brief Builds vocabulary from a text, replacing any earlier one.
	 */
	void fit(const std::vector<char>& characters) {
		std::set<unsigned char> distinct(characters.begin(), characters.end());

		charToId.clear();
		idToChar.assign(1, '\0');
		for (unsigned char character : distinct) {
			charToId[character] = idToChar.size();
			idToChar.push_back(static_cast<char>(character));
		}
	}

	/**
	 * @brief Number of ids: unknown id plus one per distinct character.
	 */
	std::size_t size() const { return idToChar.size(); }

	/**
	 * @brief Converts characters to ids (unseen characters become unknownId).
	 */
	std::vector<std::size_t> encode(const std::vector<char>& characters) const {
		std::vector<std::size_t> ids;
		ids.reserve(characters.size());
		for (char character : characters) {
			auto found = charToId.find(static_cast<unsigned char>(character));
			ids.push_back(found == charToId.end() ? unknownId : found->second);
		}
		return ids;
	}

	/**
	 * @brief Converts ids back to text (the unknown id becomes '?').
	 *
	 * @throws InvalidParameterError If an id is outside vocabulary.
	 */
	std::string decode(const std::vector<std::size_t>& ids) const {
		std::string text;
		for (std::size_t tokenId : ids) {
			validation::requireBelow(tokenId, idToChar.size(), "Token id");
			text += (tokenId == unknownId) ? '?' : idToChar[tokenId];
		}
		return text;
	}

private:
	std::map<unsigned char, std::size_t> charToId;
	std::vector<char> idToChar = std::vector<char>(1, '\0');
};
