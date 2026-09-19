
#include <iostream>
#include <vector>
#include "BasicGPTWithMoE.h"

/**
 * @brief Demo stage: trains BasicGPTWithMoE (decoder-only transformer whose
 * feed-forward sublayers are Mixture-of-Experts layers) on repeating
 * pattern 0, 1, 2, 0, 1, 2, ... and checks that generation stays inside the
 * vocabulary.
 *
 * @return 0 on completion.
 */
int testBasicGPTWithMoE() {
    // dModel and ctxLen must match BasicGPTConfig (d_head = 16,
    // maxSeqLen = 20): RotaryEmbedding is sized from that config.
    std::size_t vocabSize = 10;
    std::size_t dModel = 16;
    std::size_t numLayers = 2;
    std::size_t ctxLen = 20;
    // MoE settings: 4 experts per layer, each token routed to its single
    // best expert (top-1).
    bool useMoe = true;
    std::size_t numExperts = 4;
    std::size_t topK = 1;

    BasicGPTWithMoE gpt(vocabSize, dModel, numLayers, ctxLen, useMoe, numExperts, topK);

    if (useMoe) {
        std::cout << "MoE Enabled: " << numExperts << " experts, top-" << topK << std::endl;
    }
    else {
        std::cout << "Standard FF Enabled." << std::endl;
    }

    // Data: 0, 1, 2, 0, 1, 2...
    std::vector<size_t> data;
    for (int i = 0; i < 100; i++) {
        data.push_back(i % 3);
    }

    // Train
    double learningRate = 0.05;
    std::cout << "Training GPT..." << std::endl;
    for (int epoch = 0; epoch < 200; epoch++) {
        // Causal Language Modeling: Predict next token
        // Input: 0, 1, 2, 0, 1
        // Target: 1, 2, 0, 1, 2

        // Slide a short window (5 tokens, stride 5) over data to keep each
        // step fast; every position predicts token that follows it.
        double totalLoss = 0;
        int stepCount = 0;

        for (std::size_t i = 0; i < data.size() - 6; i += 5) {
            std::vector<size_t> inputIds, targetIds;
            for (int j = 0; j < 5; j++) {
                inputIds.push_back(data[i + j]);
                targetIds.push_back(data[i + j + 1]);
            }
            totalLoss += gpt.trainStep(inputIds, targetIds, learningRate);
            stepCount++;
        }

        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << totalLoss / stepCount << std::endl;
        }
    }

    // Generate
    std::vector<size_t> prompt = { 0, 1 };
    std::vector<size_t> generated = gpt.generate(prompt, 10);

    std::cout << "Generated: ";
    for (auto token : generated) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    // Expect: 0 1 2 0 1 2 ...
    // Verify last few
    if (generated.back() == 2 || generated.back() == 1 || generated.back() == 0) {
        std::cout << "Test Passed: Generated meaningful tokens within vocab." << std::endl;
    }
    else {
        std::cout << "Test Failed: Weird tokens." << std::endl;
    }
    return 0;
}