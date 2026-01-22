#include "matrix.hpp"

#include <cmath>

namespace matmul {

namespace {
inline double lcg_next(std::uint64_t& state) {
  state = state * 6364136223846793005ULL + 1ULL;
  const std::uint64_t x = state >> 11;  // 53 bits
  return static_cast<double>(x) * (1.0 / 9007199254740992.0);
}
}

void init_matrix(Matrix<float>& m, std::uint64_t seed) {
  std::uint64_t state = seed;
  for (std::size_t i = 0; i < m.rows; ++i) {
    for (std::size_t j = 0; j < m.cols; ++j) {
      const double u = lcg_next(state);
      const double v = (u - 0.5) * 0.2;
      m(i, j) = static_cast<float>(v);
    }
  }
}

void init_matrix(Matrix<double>& m, std::uint64_t seed) {
  std::uint64_t state = seed;
  for (std::size_t i = 0; i < m.rows; ++i) {
    for (std::size_t j = 0; j < m.cols; ++j) {
      const double u = lcg_next(state);
      const double v = (u - 0.5) * 0.2;
      m(i, j) = v;
    }
  }
}

void transpose(const Matrix<float>& src, Matrix<float>& dst) {
  if (dst.rows != src.cols || dst.cols != src.rows) {
    dst.resize(src.cols, src.rows);
  }
  for (std::size_t i = 0; i < src.rows; ++i) {
    for (std::size_t j = 0; j < src.cols; ++j) {
      dst(j, i) = src(i, j);
    }
  }
}

void transpose(const Matrix<double>& src, Matrix<double>& dst) {
  if (dst.rows != src.cols || dst.cols != src.rows) {
    dst.resize(src.cols, src.rows);
  }
  for (std::size_t i = 0; i < src.rows; ++i) {
    for (std::size_t j = 0; j < src.cols; ++j) {
      dst(j, i) = src(i, j);
    }
  }
}

}  // namespace matmul
