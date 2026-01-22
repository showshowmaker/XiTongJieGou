#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace matmul {

inline void* aligned_malloc(std::size_t size, std::size_t alignment) {
#if defined(_MSC_VER)
  return _aligned_malloc(size, alignment);
#else
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
#endif
}

inline void aligned_free(void* ptr) {
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

template <typename T>
struct Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t ld = 0;
  T* data = nullptr;

  Matrix() = default;

  Matrix(std::size_t r, std::size_t c) { resize(r, c); }

  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;

  Matrix(Matrix&& other) noexcept {
    rows = other.rows;
    cols = other.cols;
    ld = other.ld;
    data = other.data;
    other.rows = 0;
    other.cols = 0;
    other.ld = 0;
    other.data = nullptr;
  }

  Matrix& operator=(Matrix&& other) noexcept {
    if (this != &other) {
      release();
      rows = other.rows;
      cols = other.cols;
      ld = other.ld;
      data = other.data;
      other.rows = 0;
      other.cols = 0;
      other.ld = 0;
      other.data = nullptr;
    }
    return *this;
  }

  ~Matrix() { release(); }

  void resize(std::size_t r, std::size_t c) {
    release();
    rows = r;
    cols = c;
    ld = c;
    if (rows == 0 || cols == 0) {
      data = nullptr;
      return;
    }
    const std::size_t bytes = rows * ld * sizeof(T);
    data = static_cast<T*>(aligned_malloc(bytes, 64));
    if (!data) {
      throw std::bad_alloc();
    }
  }

  void release() {
    if (data) {
      aligned_free(data);
      data = nullptr;
    }
    rows = 0;
    cols = 0;
    ld = 0;
  }

  inline T& operator()(std::size_t r, std::size_t c) {
    return data[r * ld + c];
  }

  inline const T& operator()(std::size_t r, std::size_t c) const {
    return data[r * ld + c];
  }

  inline T* row_ptr(std::size_t r) { return data + r * ld; }
  inline const T* row_ptr(std::size_t r) const { return data + r * ld; }

  void zero() {
    if (!data) {
      return;
    }
    std::fill(data, data + rows * ld, static_cast<T>(0));
  }
};

void init_matrix(Matrix<float>& m, std::uint64_t seed);
void init_matrix(Matrix<double>& m, std::uint64_t seed);

void transpose(const Matrix<float>& src, Matrix<float>& dst);
void transpose(const Matrix<double>& src, Matrix<double>& dst);

}  // namespace matmul
