#pragma once

#include "matrix.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace matmul {

template <typename T>
struct VerifySummary {
  std::size_t checked = 0;
  std::size_t mismatches = 0;
  double max_abs = 0.0;
  double max_rel = 0.0;
};

template <typename T>
inline VerifySummary<T> verify_full(const Matrix<T>& ref, const Matrix<T>& test, double atol,
                                    double rtol) {
  VerifySummary<T> summary;
  const std::size_t rows = ref.rows;
  const std::size_t cols = ref.cols;
  summary.checked = rows * cols;
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      const double a = static_cast<double>(ref(i, j));
      const double b = static_cast<double>(test(i, j));
      const double diff = std::fabs(a - b);
      const double rel = diff / (std::fabs(a) + 1e-12);
      if (diff > atol && rel > rtol) {
        ++summary.mismatches;
      }
      if (diff > summary.max_abs) {
        summary.max_abs = diff;
      }
      if (rel > summary.max_rel) {
        summary.max_rel = rel;
      }
    }
  }
  return summary;
}

template <typename T>
inline VerifySummary<T> verify_sampled_gemm(const Matrix<T>& a, const Matrix<T>& b,
                                            const Matrix<T>& c, std::size_t samples,
                                            double atol, double rtol, std::uint64_t seed) {
  VerifySummary<T> summary;
  if (samples == 0) {
    return summary;
  }
  std::uint64_t state = seed;
  summary.checked = samples;
  for (std::size_t s = 0; s < samples; ++s) {
    state = state * 6364136223846793005ULL + 1ULL;
    const std::size_t i = static_cast<std::size_t>(state % a.rows);
    state = state * 6364136223846793005ULL + 1ULL;
    const std::size_t j = static_cast<std::size_t>(state % b.cols);

    double ref = 0.0;
    for (std::size_t k = 0; k < a.cols; ++k) {
      ref += static_cast<double>(a(i, k)) * static_cast<double>(b(k, j));
    }
    const double test = static_cast<double>(c(i, j));
    const double diff = std::fabs(ref - test);
    const double rel = diff / (std::fabs(ref) + 1e-12);
    if (diff > atol && rel > rtol) {
      ++summary.mismatches;
    }
    if (diff > summary.max_abs) {
      summary.max_abs = diff;
    }
    if (rel > summary.max_rel) {
      summary.max_rel = rel;
    }
  }
  return summary;
}

}  // namespace matmul
