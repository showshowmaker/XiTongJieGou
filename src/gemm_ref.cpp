#include "kernels.hpp"

namespace matmul {

namespace {

template <typename T>
void gemm_ref_impl(const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c) {
  const std::size_t m = a.rows;
  const std::size_t n = b.cols;
  const std::size_t k = a.cols;
  c.zero();
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      T sum = static_cast<T>(0);
      for (std::size_t kk = 0; kk < k; ++kk) {
        sum += a(i, kk) * b(kk, j);
      }
      c(i, j) = sum;
    }
  }
}

}  // namespace

void gemm_ref_f32(const Matrix<float>& a, const Matrix<float>& b, Matrix<float>& c) {
  gemm_ref_impl(a, b, c);
}

void gemm_ref_f64(const Matrix<double>& a, const Matrix<double>& b, Matrix<double>& c) {
  gemm_ref_impl(a, b, c);
}

}  // namespace matmul
