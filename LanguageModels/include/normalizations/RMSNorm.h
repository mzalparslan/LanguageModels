#pragma once

#include "Parameter.h"

/**
 * @brief Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
 *
 * Rescales every row (one position's vector) by its root mean square, then
 * applies a learned per-feature scale:
 * y = x / sqrt(mean(x^2) + eps) * weight
 *
 * Unlike LayerNorm it does not subtract mean and has no bias, which is
 * cheaper and works about as well in transformers.
 *
 * @remark Stateful: forward() caches its input and per-row RMS, and
 * backward() consumes them, so call backward() right after forward() it
 * belongs to.
 */
template <typename T>
class RMSNorm {
public:
    // Learned per-feature scale (g), initialized to 1
    Parameter<T> weight; 
    // Feature width of each row.
    std::size_t dModel;

    /**
     * @param dim Feature width of each row.
     * @param epsilon Numerical-stability constant added before square root.
     */
    RMSNorm(std::size_t dim, double epsilon = 1e-5) : dModel(dim), eps(epsilon) {
        validation::requirePositiveSize(dim, "RMSNorm width");
        // A zero epsilon would divide by zero on an all-zero row.
        validation::requirePositiveFinite(epsilon, "RMSNorm epsilon");
        // Initialize scale to 1.0, so layer starts as pure normalization.
        // No randomness is needed.
        weight.initConstant({ dim }, T(1));
    }

    /**
     * @brief Clears accumulated gradient of scale parameter.
     */
    void zeroGrad() {
        weight.zeroGrad();
    }

    /**
     * @brief Normalizes each row of x independently.
     *
     * @param x Input [Seq, dModel].
     * @param out Output, same shape as x.
     */
    void forward(const Tensor<T>& x, Tensor<T>& out) {
        validation::requireColumns(x, dModel, "RMSNorm input");
        // x: [Seq, dModel]
        std::size_t seq = x.shape[0];
        std::size_t d = x.shape[1];

        inputCache = x;
        out = Tensor<T>(x.shape);
        rmsCache = Tensor<T>({ seq }, 0);

        for (std::size_t i = 0; i < seq; i++) {
            // Root mean square of this row: sqrt(mean(x^2) + eps).
            T sumSq = 0;
            for (std::size_t j = 0; j < d; j++) {
                sumSq += x.data[i * d + j] * x.data[i * d + j];
            }
            T rms = static_cast<T>(std::sqrt(sumSq / d + eps));
            // A NaN/Inf input shows up here; a zero rms would divide by zero.
            validation::requireFinite(rms, "RMSNorm root mean square");
            validation::requireNonZeroDenominator(rms, "RMSNorm root mean square");
            rmsCache.data[i] = rms;

            // Scale to unit RMS, then apply learned per-feature scale.
            for (std::size_t j = 0; j < d; j++) {
                out.data[i * d + j] = (x.data[i * d + j] / rms) * weight.value[j];
            }
        }
    }

    /**
     * @brief Backward pass: accumulates gradient of scale parameter
     * and returns gradient with respect to input.
     *
     * y = (x / RMS) * w
     * dy/dx is not simply w / RMS because RMS depends on every x_j in the
     * row; extra term below accounts for that coupling.
     *
     * @param dout Gradient w.r.t. forward()'s output [Seq, dModel].
     * @param dx Gradient w.r.t. forward()'s input, same shape.
     */
    void backward(const Tensor<T>& dout, Tensor<T>& dx) {
        if (inputCache.shape.size() != 2) {
            throw InvalidSizeError("RMSNorm::backward() called before forward()!");
        }
        validation::requireShape(dout, inputCache.shape[0], dModel, "RMSNorm output gradient");
        std::size_t seq = dout.shape[0];
        std::size_t d = dout.shape[1];
        dx = Tensor<T>(dout.shape);

        for (std::size_t i = 0; i < seq; i++) {
            T rms = rmsCache.data[i];
            T invRms = T(1) / rms;
            T sumGradX = 0; // Dot product of (dout * w) and x

            // Pass 1: accumulate scale gradient and dot product that
            // couples all features of row through shared RMS.

            for (std::size_t j = 0; j < d; j++) {
                T w = weight.value[j];
                T dz = dout.data[i * d + j];

                // Accumulate gradient for weight
                // dw = dout * (x / RMS)
                weight.grad.data[j] += dz * (inputCache.data[i * d + j] * invRms);

                sumGradX += (dz * w) * inputCache.data[i * d + j];
            }

            T val = sumGradX * static_cast<T>(1.0 / (d * rms * rms)); // factor for dRMS/dx part

            // Pass 2: dx_j = (dout_j * w_j) * invRms - x_j * val * invRms
            // This is simplified gradient form for RMSNorm
            for (std::size_t j = 0; j < d; j++) {
                T w = weight.value[j];
                T dz = dout.data[i * d + j];
                dx.data[i * d + j] = (dz * w * invRms) - (inputCache.data[i * d + j] * val * invRms);
            }
        }
    }

    /**
     * @brief Applies accumulated gradient to scale parameter.
     *
     * @param lr Learning rate.
     * @param rule Optimizer to use; see UpdateRule.
     */
    void update(T lr, UpdateRule rule = UpdateRule::sgd()) {
        weight.update(lr, rule);
    }

private:
    // Small constant added inside square root to avoid dividing by zero.
    double eps;

    /**
     * @brief Caches from last forward() call, used by backward().
     */
    Tensor<T> inputCache;
    // RMS of each row (not its inverse)
    Tensor<T> rmsCache; 
};
