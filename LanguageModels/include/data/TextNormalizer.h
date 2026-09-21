#pragma once

#include <cctype>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Small text helpers shared by data loaders and tokenizers.
 */
namespace text {

	/**
	 * @brief Lowercases text and keeps only ASCII letters, spaces, apostrophes
	 * and hyphens. Accented letters (multi-byte UTF-8), digits and other
	 * punctuation are dropped.
	 *
	 * Deliberately simple, so word vocabulary of a translation corpus stays
	 * small enough for toy models in this library.
	 */
	inline std::string normalize(const std::string& text) {
		std::string normalized;
		for (char character : text) {
			// <cctype> functions are undefined for negative values, which UTF-8
			// continuation bytes (accented characters) are when stored in a char.
			const unsigned char byteValue = static_cast<unsigned char>(character);
			if (std::isalpha(byteValue) || byteValue == ' ') {
				normalized += static_cast<char>(std::tolower(byteValue));
			}
			else if (byteValue == '\'' || byteValue == '-') {
				normalized += character;
			}
		}
		return normalized;
	}

	/**
	 * @brief Splits text into words on whitespace.
	 */
	inline std::vector<std::string> splitOnWhitespace(const std::string& text) {
		std::vector<std::string> words;
		std::stringstream stream(text);
		std::string word;
		while (stream >> word) {
			words.push_back(word);
		}
		return words;
	}

} // namespace text
