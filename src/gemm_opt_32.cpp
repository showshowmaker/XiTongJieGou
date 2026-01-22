#include "kernels.hpp"

#if defined(MATMUL_ENABLE_AVX2)
#include <immintrin.h>
#endif

namespace matmul {

void gemm_opt_32_f32(const Matrix<float>& a, const Matrix<float>& b, Matrix<float>& c) {
  if (a.rows != 32 || a.cols != 32 || b.rows != 32 || b.cols != 32 || c.rows != 32 ||
      c.cols != 32) {
    gemm_base_f32(a, b, c);
    return;
  }

#if defined(MATMUL_ENABLE_AVX2) && (defined(__AVX2__) || defined(_MSC_VER))
  c.zero();
  for (std::size_t i = 0; i < 32; ++i) {
    const float* a_row = a.row_ptr(i);
    float* c_row = c.row_ptr(i);
    for (std::size_t j = 0; j < 32; j += 8) {
      __m256 c_vec = _mm256_setzero_ps();
      for (std::size_t k = 0; k < 32; ++k) {
        const __m256 b_vec = _mm256_load_ps(b.row_ptr(k) + j);
        const __m256 a_vec = _mm256_broadcast_ss(a_row + k);
#if defined(__FMA__) || defined(_MSC_VER)
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
#else
        c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, b_vec));
#endif
      }
      _mm256_store_ps(c_row + j, c_vec);
    }
  }
#else
  gemm_base_f32(a, b, c);
#endif
}

}  // namespace matmul
