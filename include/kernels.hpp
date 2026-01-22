#pragma once

#include "matrix.hpp"

namespace matmul {

enum class KernelKind {
  Ref,
  Base,
  Opt
};

void gemm_ref_f32(const Matrix<float>& a, const Matrix<float>& b, Matrix<float>& c);
void gemm_ref_f64(const Matrix<double>& a, const Matrix<double>& b, Matrix<double>& c);

void gemm_base_f32(const Matrix<float>& a, const Matrix<float>& b, Matrix<float>& c);
void gemm_base_f64(const Matrix<double>& a, const Matrix<double>& b, Matrix<double>& c);

void gemm_opt_32_f32(const Matrix<float>& a, const Matrix<float>& b, Matrix<float>& c);
void gemm_opt_4096_f64(const Matrix<double>& a, const Matrix<double>& b, Matrix<double>& c);

}  // namespace matmul
