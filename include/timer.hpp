#pragma once

#include <chrono>

namespace matmul {

class Timer {
 public:
  using clock = std::chrono::steady_clock;

  void reset() { start_ = clock::now(); }

  double elapsed_ms() const {
    const auto end = clock::now();
    const std::chrono::duration<double, std::milli> diff = end - start_;
    return diff.count();
  }

 private:
  clock::time_point start_ = clock::now();
};

}  // namespace matmul
