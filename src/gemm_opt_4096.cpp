#include "kernels.hpp"

#include <algorithm>

#if defined(MATMUL_ENABLE_AVX2)
#include <immintrin.h>
#endif

namespace matmul {

namespace {
constexpr std::size_t kBlockI = 64;
constexpr std::size_t kBlockJ = 64;
constexpr std::size_t kBlockK = 64;
}

void gemm_opt_4096_f64(const Matrix<double>& a, const Matrix<double>& b, Matrix<double>& c) {
  if (a.rows != 4096 || a.cols != 4096 || b.rows != 4096 || b.cols != 4096 || c.rows != 4096 ||
      c.cols != 4096) {
    gemm_base_f64(a, b, c);
    return;
  }

#if defined(MATMUL_ENABLE_AVX2) && (defined(__AVX2__) || defined(_MSC_VER))
  c.zero();
  const std::size_t m = a.rows;
  const std::size_t n = b.cols;
  const std::size_t k = a.cols;

  for (std::size_t ii = 0; ii < m; ii += kBlockI) {
    const std::size_t i_max = std::min(ii + kBlockI, m);
    for (std::size_t jj = 0; jj < n; jj += kBlockJ) {
      const std::size_t j_max = std::min(jj + kBlockJ, n);
      for (std::size_t kk = 0; kk < k; kk += kBlockK) {
        const std::size_t k_max = std::min(kk + kBlockK, k);
        std::size_t i = ii;
        for (; i + 3 < i_max; i += 4) {
          const double* a_row0 = a.row_ptr(i);
          const double* a_row1 = a.row_ptr(i + 1);
          const double* a_row2 = a.row_ptr(i + 2);
          const double* a_row3 = a.row_ptr(i + 3);

          std::size_t j = jj;
          for (; j + 3 < j_max; j += 4) {
            __m256d c0 = _mm256_load_pd(c.row_ptr(i) + j);
            __m256d c1 = _mm256_load_pd(c.row_ptr(i + 1) + j);
            __m256d c2 = _mm256_load_pd(c.row_ptr(i + 2) + j);
            __m256d c3 = _mm256_load_pd(c.row_ptr(i + 3) + j);

            for (std::size_t kk2 = kk; kk2 < k_max; ++kk2) {
              const __m256d b_vec = _mm256_load_pd(b.row_ptr(kk2) + j);
              const __m256d a0 = _mm256_broadcast_sd(a_row0 + kk2);
              const __m256d a1 = _mm256_broadcast_sd(a_row1 + kk2);
              const __m256d a2 = _mm256_broadcast_sd(a_row2 + kk2);
              const __m256d a3 = _mm256_broadcast_sd(a_row3 + kk2);
#if defined(__FMA__) || defined(_MSC_VER)
              c0 = _mm256_fmadd_pd(a0, b_vec, c0);
              c1 = _mm256_fmadd_pd(a1, b_vec, c1);
              c2 = _mm256_fmadd_pd(a2, b_vec, c2);
              c3 = _mm256_fmadd_pd(a3, b_vec, c3);
#else
              c0 = _mm256_add_pd(c0, _mm256_mul_pd(a0, b_vec));
              c1 = _mm256_add_pd(c1, _mm256_mul_pd(a1, b_vec));
              c2 = _mm256_add_pd(c2, _mm256_mul_pd(a2, b_vec));
              c3 = _mm256_add_pd(c3, _mm256_mul_pd(a3, b_vec));
#endif
            }

            _mm256_store_pd(c.row_ptr(i) + j, c0);
            _mm256_store_pd(c.row_ptr(i + 1) + j, c1);
            _mm256_store_pd(c.row_ptr(i + 2) + j, c2);
            _mm256_store_pd(c.row_ptr(i + 3) + j, c3);
          }

          for (; j < j_max; ++j) {
            double sum0 = c(i, j);
            double sum1 = c(i + 1, j);
            double sum2 = c(i + 2, j);
            double sum3 = c(i + 3, j);
            for (std::size_t kk2 = kk; kk2 < k_max; ++kk2) {
              const double b_val = b(kk2, j);
              sum0 += a_row0[kk2] * b_val;
              sum1 += a_row1[kk2] * b_val;
              sum2 += a_row2[kk2] * b_val;
              sum3 += a_row3[kk2] * b_val;
            }
            c(i, j) = sum0;
            c(i + 1, j) = sum1;
            c(i + 2, j) = sum2;
            c(i + 3, j) = sum3;
          }
        }

        for (; i < i_max; ++i) {
          for (std::size_t j = jj; j < j_max; ++j) {
            double sum = c(i, j);
            for (std::size_t kk2 = kk; kk2 < k_max; ++kk2) {
              sum += a(i, kk2) * b(kk2, j);
            }
            c(i, j) = sum;
          }
        }
      }
    }
  }
#else
  gemm_base_f64(a, b, c);
#endif
}

}  // namespace matmul
