#include "kernels.hpp"

namespace matmul {

namespace {

template <typename T>
void gemm_base_impl(const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c) {
  const std::size_t m = a.rows;
  const std::size_t n = b.cols;
  const std::size_t k = a.cols;
  c.zero();
  for (std::size_t i = 0; i < m; ++i) {
    T* c_row = c.row_ptr(i);
    for (std::size_t kk = 0; kk < k; ++kk) {
      const T a_val = a(i, kk);
      const T* b_row = b.row_ptr(kk);
      for (std::size_t j = 0; j < n; ++j) {
        c_row[j] += a_val * b_row[j];
      }
    }
  }
}

}  // namespace

void gemm_base_f32(const Matrix<float>& a, const Matrix<float>& b, Matrix<float>& c) {
  gemm_base_impl(a, b, c);
}

void gemm_base_f64(const Matrix<double>& a, const Matrix<double>& b, Matrix<double>& c) {
  gemm_base_impl(a, b, c);
}

}  // namespace matmul
