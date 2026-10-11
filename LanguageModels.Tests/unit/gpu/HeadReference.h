#pragma once

#include "LinearLayer.h"
#include "Parameter.h"
#include <algorithm>
#include <cmath>
#include <vector>

// What the tests of a GPU vocabulary head (CUDA or Metal) compare it with: the CPU code
// the head replaces, in double precision, where the two must agree to within the
// precision of the head (rounding for double, about 1e-4 for float; the GPU sums in a
// different order and may fuse multiply-adds, so they are not bit-identical).
namespace headreference {

    const std::size_t sourceVocab = 50;
    const std::size_t targetVocab = 700;   // large enough for several blocks per row
    const std::size_t width = 8;

    // Deterministic values of a chosen scale.
    inline std::vector<double> pattern(std::size_t count, double scale, double phase) {
        std::vector<double> values(count);
        for (std::size_t i = 0; i < count; i++) {
            values[i] = scale * std::sin(0.37 * static_cast<double>(i) + phase);
        }
        return values;
    }

    template <typename T>
    std::vector<T> convert(const std::vector<double>& values) {
        std::vector<T> converted;
        converted.reserve(values.size());
        for (double value : values) {
            converted.push_back(static_cast<T>(value));
        }
        return converted;
    }

    // The four parameters of a head, with some starting values.
    class Weights {
    public:
        std::vector<double> source = pattern(sourceVocab * width, 0.5, 0.1);
        std::vector<double> target = pattern(targetVocab * width, 0.5, 0.2);
        std::vector<double> projection = pattern(width * targetVocab, 0.3, 0.3);
        std::vector<double> bias = pattern(targetVocab, 0.1, 0.4);
    };

    // The CPU's output layer: LinearLayer, then softmax + cross-entropy exactly as
    // MiniTransformer::trainStep() does it. Fills dx and leaves the gradients in layer.
    inline double referenceOutputLayer(LinearLayer<double>& layer, const std::vector<double>& x,
        const std::vector<std::size_t>& labels, std::vector<double>& dx) {
        const std::size_t rows = labels.size();
        Tensor<double> input({ rows, width });
        input.data = x;
        Tensor<double> logits;
        layer.forward(input, logits);

        Tensor<double> dlogits(logits.shape, 0);
        double loss = 0;
        for (std::size_t i = 0; i < rows; i++) {
            double maxValue = -1e9;
            for (std::size_t j = 0; j < targetVocab; j++) {
                maxValue = std::max(maxValue, logits.data[i * targetVocab + j]);
            }
            double sum = 0;
            std::vector<double> probs(targetVocab);
            for (std::size_t j = 0; j < targetVocab; j++) {
                probs[j] = std::exp(logits.data[i * targetVocab + j] - maxValue);
                sum += probs[j];
            }
            for (std::size_t j = 0; j < targetVocab; j++) {
                probs[j] /= sum;
            }
            loss -= std::log(probs[labels[i]]);
            for (std::size_t j = 0; j < targetVocab; j++) {
                dlogits.data[i * targetVocab + j] = probs[j];
            }
            dlogits.data[i * targetVocab + labels[i]] -= 1.0;
        }

        Tensor<double> dxTensor;
        layer.backward(dlogits, dxTensor);
        dx = dxTensor.data;
        return loss;
    }

    inline LinearLayer<double> makeReferenceLayer(const Weights& weights) {
        RandomEngine rng(1);
        LinearLayer<double> layer(width, targetVocab, rng);
        layer.W.value.data = weights.projection;
        layer.b.value.data = weights.bias;
        layer.zeroGrad();
        return layer;
    }

    // The CPU update of one embedding table given per-row gradients, for comparison.
    inline std::vector<double> referenceTableUpdate(const std::vector<double>& table, std::size_t vocab,
        const std::vector<std::size_t>& ids, const std::vector<double>& rowGradients,
        std::size_t steps, bool adam, double learningRate) {
        Parameter<double> parameter;
        RandomEngine rng(1);
        parameter.init({ vocab, width }, 0.1, rng);
        parameter.value.data = table;

        for (std::size_t step = 1; step <= steps; step++) {
            parameter.zeroGrad();
            for (std::size_t r = 0; r < ids.size(); r++) {
                for (std::size_t c = 0; c < width; c++) {
                    parameter.grad.data[ids[r] * width + c] += rowGradients[r * width + c];
                }
            }
            parameter.update(learningRate, adam ? UpdateRule::adam(step) : UpdateRule::sgd());
        }
        return parameter.value.data;
    }

}
