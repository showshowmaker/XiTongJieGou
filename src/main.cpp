#include "kernels.hpp"
#include "matrix.hpp"
#include "timer.hpp"
#include "verify.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace matmul {

enum class DType {
  F32,
  F64
};

struct Options {
  std::size_t size = 32;
  DType dtype = DType::F32;
  KernelKind kernel = KernelKind::Base;
  std::size_t repeat = 0;
  std::size_t warmup = 0;
  bool verify = false;
  std::size_t samples = 64;
  std::string csv_path = "report/results.csv";
  std::string case_label = "small32";
};

std::string to_lower(std::string s) {
  for (char& ch : s) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return s;
}

std::size_t default_repeat(std::size_t size) {
  if (size <= 64) {
    return 100000;
  }
  if (size <= 512) {
    return 20;
  }
  return 3;
}

std::size_t default_warmup(std::size_t size) {
  if (size <= 64) {
    return 10;
  }
  return 1;
}

void print_usage() {
  std::cout
      << "Usage: matmul_bench [--case small|large] [--size N] [--dtype fp32|fp64]"
         " [--kernel ref|base|opt] [--repeat N] [--warmup N] [--verify]"
         " [--sample N] [--csv path]\n";
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--case" && i + 1 < argc) {
      std::string val = to_lower(argv[++i]);
      if (val == "small") {
        opts.size = 32;
        opts.dtype = DType::F32;
        opts.case_label = "small32";
      } else if (val == "large") {
        opts.size = 4096;
        opts.dtype = DType::F64;
        opts.case_label = "large4096";
      }
    } else if (arg == "--size" && i + 1 < argc) {
      opts.size = static_cast<std::size_t>(std::stoull(argv[++i]));
      opts.case_label = "custom";
    } else if (arg == "--dtype" && i + 1 < argc) {
      std::string val = to_lower(argv[++i]);
      if (val == "fp32") {
        opts.dtype = DType::F32;
      } else if (val == "fp64") {
        opts.dtype = DType::F64;
      }
    } else if (arg == "--kernel" && i + 1 < argc) {
      std::string val = to_lower(argv[++i]);
      if (val == "ref") {
        opts.kernel = KernelKind::Ref;
      } else if (val == "base") {
        opts.kernel = KernelKind::Base;
      } else if (val == "opt") {
        opts.kernel = KernelKind::Opt;
      }
    } else if (arg == "--repeat" && i + 1 < argc) {
      opts.repeat = static_cast<std::size_t>(std::stoull(argv[++i]));
    } else if (arg == "--warmup" && i + 1 < argc) {
      opts.warmup = static_cast<std::size_t>(std::stoull(argv[++i]));
    } else if (arg == "--verify") {
      opts.verify = true;
    } else if (arg == "--sample" && i + 1 < argc) {
      opts.samples = static_cast<std::size_t>(std::stoull(argv[++i]));
    } else if (arg == "--csv" && i + 1 < argc) {
      opts.csv_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    } else {
      std::cerr << "Unknown arg: " << arg << "\n";
      print_usage();
      std::exit(1);
    }
  }

  if (opts.repeat == 0) {
    opts.repeat = default_repeat(opts.size);
  }
  if (opts.warmup == 0) {
    opts.warmup = default_warmup(opts.size);
  }
  return opts;
}

const char* kernel_name(KernelKind k) {
  switch (k) {
    case KernelKind::Ref:
      return "ref";
    case KernelKind::Base:
      return "base";
    case KernelKind::Opt:
      return "opt";
  }
  return "unknown";
}

const char* dtype_name(DType d) { return d == DType::F32 ? "fp32" : "fp64"; }

bool file_has_content(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  return in.peek() != std::ifstream::traits_type::eof();
}

}  // namespace matmul

int main(int argc, char** argv) {
  using namespace matmul;

  Options opts = parse_args(argc, argv);
  const std::size_t n = opts.size;

  std::cout << "case=" << opts.case_label << " size=" << n << " dtype=" << dtype_name(opts.dtype)
            << " kernel=" << kernel_name(opts.kernel) << "\n";

  Timer timer;
  std::vector<double> times_ms;
  times_ms.reserve(opts.repeat);

  VerifySummary<double> verify_summary_f64;
  VerifySummary<float> verify_summary_f32;

  if (opts.dtype == DType::F32) {
    Matrix<float> a(n, n);
    Matrix<float> b(n, n);
    Matrix<float> c(n, n);

    init_matrix(a, 1234);
    init_matrix(b, 5678);

    auto kernel = gemm_base_f32;
    if (opts.kernel == KernelKind::Ref) {
      kernel = gemm_ref_f32;
    } else if (opts.kernel == KernelKind::Opt) {
      if (n == 32) {
        kernel = gemm_opt_32_f32;
      } else {
        std::cerr << "opt kernel not available for size " << n << ", using base\n";
        kernel = gemm_base_f32;
      }
    }

    for (std::size_t w = 0; w < opts.warmup; ++w) {
      kernel(a, b, c);
    }

    for (std::size_t r = 0; r < opts.repeat; ++r) {
      timer.reset();
      kernel(a, b, c);
      times_ms.push_back(timer.elapsed_ms());
    }

    if (opts.verify) {
      if (n <= 128) {
        Matrix<float> cref(n, n);
        gemm_ref_f32(a, b, cref);
        verify_summary_f32 = verify_full(cref, c, 1e-4, 1e-4);
      } else {
        verify_summary_f32 = verify_sampled_gemm(a, b, c, opts.samples, 1e-4, 1e-4, 777);
      }
    }
  } else {
    Matrix<double> a(n, n);
    Matrix<double> b(n, n);
    Matrix<double> c(n, n);

    init_matrix(a, 1234);
    init_matrix(b, 5678);

    auto kernel = gemm_base_f64;
    if (opts.kernel == KernelKind::Ref) {
      kernel = gemm_ref_f64;
    } else if (opts.kernel == KernelKind::Opt) {
      if (n == 4096) {
        kernel = gemm_opt_4096_f64;
      } else {
        std::cerr << "opt kernel not available for size " << n << ", using base\n";
        kernel = gemm_base_f64;
      }
    }

    for (std::size_t w = 0; w < opts.warmup; ++w) {
      kernel(a, b, c);
    }

    for (std::size_t r = 0; r < opts.repeat; ++r) {
      timer.reset();
      kernel(a, b, c);
      times_ms.push_back(timer.elapsed_ms());
    }

    if (opts.verify) {
      if (n <= 256) {
        Matrix<double> cref(n, n);
        gemm_ref_f64(a, b, cref);
        verify_summary_f64 = verify_full(cref, c, 1e-10, 1e-10);
      } else {
        verify_summary_f64 = verify_sampled_gemm(a, b, c, opts.samples, 1e-10, 1e-10, 777);
      }
    }
  }

  if (times_ms.empty()) {
    std::cerr << "No timing samples collected.\n";
    return 1;
  }

  std::sort(times_ms.begin(), times_ms.end());
  const double median_ms = times_ms[times_ms.size() / 2];
  const double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) *
                       static_cast<double>(n);
  const double gflops = flops / (median_ms * 1.0e6);

  std::cout << "median_ms=" << median_ms << " gflops=" << gflops << "\n";

  if (opts.verify) {
    if (opts.dtype == DType::F32) {
      std::cout << "verify_checked=" << verify_summary_f32.checked
                << " mismatches=" << verify_summary_f32.mismatches
                << " max_abs=" << verify_summary_f32.max_abs
                << " max_rel=" << verify_summary_f32.max_rel << "\n";
    } else {
      std::cout << "verify_checked=" << verify_summary_f64.checked
                << " mismatches=" << verify_summary_f64.mismatches
                << " max_abs=" << verify_summary_f64.max_abs
                << " max_rel=" << verify_summary_f64.max_rel << "\n";
    }
  }

  if (!opts.csv_path.empty()) {
    const std::filesystem::path csv_path(opts.csv_path);
    if (csv_path.has_parent_path()) {
      std::error_code ec;
      std::filesystem::create_directories(csv_path.parent_path(), ec);
    }
    const bool has_content = file_has_content(opts.csv_path);
    std::ofstream out(opts.csv_path, std::ios::app);
    if (!out) {
      std::cerr << "Failed to open CSV: " << opts.csv_path << "\n";
      return 1;
    }
    if (!has_content) {
      out << "case,dtype,kernel,size,time_ms,gflops,repeat,warmup,verify_checked,verify_mismatches,"
             "verify_max_abs,verify_max_rel\n";
    }
    if (opts.dtype == DType::F32) {
      out << opts.case_label << "," << dtype_name(opts.dtype) << "," << kernel_name(opts.kernel)
          << "," << n << "," << median_ms << "," << gflops << "," << opts.repeat << ","
          << opts.warmup << "," << verify_summary_f32.checked << ","
          << verify_summary_f32.mismatches << "," << verify_summary_f32.max_abs << ","
          << verify_summary_f32.max_rel << "\n";
    } else {
      out << opts.case_label << "," << dtype_name(opts.dtype) << "," << kernel_name(opts.kernel)
          << "," << n << "," << median_ms << "," << gflops << "," << opts.repeat << ","
          << opts.warmup << "," << verify_summary_f64.checked << ","
          << verify_summary_f64.mismatches << "," << verify_summary_f64.max_abs << ","
          << verify_summary_f64.max_rel << "\n";
    }
  }

  return 0;
}
