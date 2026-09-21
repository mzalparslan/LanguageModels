#pragma once

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Exceptions.h"
#include "SentencePair.h"
#include "TextNormalizer.h"

/**
 * @brief Loads text data pipelines train on.
 *
 * Every loader reads a whole file into memory and throws DataLoadError if the
 * file cannot be opened or holds nothing usable.
 */
class DataLoader {
	// Prevent instantiation of this class.
	DataLoader() = delete;

public:
	/**
	 * @brief Loads tab-separated sentence pairs (the Tatoeba format: source,
	 * target, and an optional attribution column that is ignored).
	 *
	 * Both sides are passed through text::normalize(). Lines with no tab, or
	 * whose source or target is empty after normalizing, are skipped.
	 *
	 * @param filePath File to read.
	 * @param limit Most pairs to load; 0 loads them all.
	 * @throws DataLoadError If file cannot be opened, or has no usable pair.
	 */
	[[nodiscard]]
	static std::vector<SentencePair> loadSentencePairs(const std::filesystem::path& filePath,
		std::size_t limit = 0) {
		std::ifstream file = openFile(filePath);

		std::vector<SentencePair> pairs;
		std::string line;
		while (std::getline(file, line) && (limit == 0 || pairs.size() < limit)) {
			const std::size_t firstTab = line.find('\t');
			if (firstTab == std::string::npos) {
				continue;
			}

			std::string source = text::normalize(line.substr(0, firstTab));
			const std::string remainder = line.substr(firstTab + 1);
			// Anything after a second tab is attribution column.
			std::string target = text::normalize(remainder.substr(0, remainder.find('\t')));

			if (source.empty() || target.empty()) {
				continue;
			}

			pairs.push_back({ std::move(source), std::move(target) });
		}

		requireReadable(file, filePath);
		if (pairs.empty()) {
			throw DataLoadError("DataLoader: " + filePath.string() + " contains no sentence pairs.");
		}
		return pairs;
	}

	/**
	 * @brief Loads a whole text file. Carriage returns are dropped, so a file
	 * written on Windows loads same as on Linux.
	 *
	 * @throws DataLoadError If file cannot be opened, or is empty.
	 */
	[[nodiscard]]
	static std::string loadText(const std::filesystem::path& filePath) {
		std::ifstream file = openFile(filePath);

		std::ostringstream contents;
		contents << file.rdbuf();
		requireReadable(file, filePath);

		std::string result = contents.str();
		result.erase(std::remove(result.begin(), result.end(), '\r'), result.end());
		if (result.empty()) {
			throw DataLoadError("DataLoader: " + filePath.string() + " is empty.");
		}
		return result;
	}

	/**
	 * @brief Loads a text file as a sequence of characters, sample type of
	 * character-level language models.
	 *
	 * @throws DataLoadError See loadText().
	 */
	[[nodiscard]]
	static std::vector<char> loadCharacters(const std::filesystem::path& filePath) {
		const std::string contents = loadText(filePath);
		return std::vector<char>(contents.begin(), contents.end());
	}

	/**
	 * @brief Loads a text file as normalized lines, skipping ones that are
	 * empty after text::normalize(). sample type of BERT: one sentence per line.
	 *
	 * @param limit Most lines to load; 0 loads them all.
	 * @throws DataLoadError If file cannot be opened, or has no usable line.
	 */
	[[nodiscard]]
	static std::vector<std::string> loadLines(const std::filesystem::path& filePath,
		std::size_t limit = 0) {
		std::ifstream file = openFile(filePath);

		std::vector<std::string> lines;
		std::string line;
		while (std::getline(file, line) && (limit == 0 || lines.size() < limit)) {
			std::string normalized = text::normalize(line);
			// A line of only spaces is as empty as a blank one.
			if (normalized.find_first_not_of(' ') == std::string::npos) {
				continue;
			}
			lines.push_back(std::move(normalized));
		}

		requireReadable(file, filePath);
		if (lines.empty()) {
			throw DataLoadError("DataLoader: " + filePath.string() + " contains no usable lines.");
		}
		return lines;
	}

private:
	static std::ifstream openFile(const std::filesystem::path& filePath) {
		std::ifstream file(filePath, std::ios::binary);
		if (!file.is_open()) {
			throw DataLoadError("DataLoader: Unable to open file: " + filePath.string());
		}
		return file;
	}

	static void requireReadable(const std::ifstream& file, const std::filesystem::path& filePath) {
		if (file.bad()) {
			throw DataLoadError("DataLoader: Error reading file: " + filePath.string());
		}
	}
};
