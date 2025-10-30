#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <random>


namespace {
static std::size_t AVX_FLOAT_COUNT = 8;

std::vector<float> make_matrix(std::size_t n) {
  std::vector<float> matrix(n*n);

  static std::random_device ran_dev;
  static std::mt19937 ran_eng(ran_dev());
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::generate(matrix.begin(), matrix.end(), [&]() { return dist(ran_eng); });

  return matrix;
}

static void row_calculation_amount(const float* input_row, 
                                    std::size_t n,
                                    float* output_row)
{
    float row_sum = 0.0f;
    for (std::size_t item = 0; item < n; item++) { row_sum += exp(input_row[item]); }
    const float divider = 1 / row_sum;
    for (std::size_t item = 0; item < n; item++) { output_row[item] = exp(input_row[item]) / row_sum; }
}

std::vector<float> run_sequential(const std::vector<float> &matrix,
                                  std::size_t n) {
  //throw std::runtime_error("Sequential method not implemented");

  std::vector<float> res_matrix(n*n);
  
  for (std::size_t row = 0; row < n; row++)
  {
    row_calculation_amount(&matrix[row*n], n, &res_matrix[row*n]);
  }
  return res_matrix;
}

std::vector<float> run_openmp(const std::vector<float> &matrix, std::size_t n) {
  //throw std::runtime_error("OpenMP method not implemented");
  std::vector<float> res_matrix(n*n);

  #pragma omp parallel for
  for (int row = 0; row < n; row++)
  {
    row_calculation_amount(&matrix[row*n], n, &res_matrix[row*n]);
  }
  return res_matrix;
}

double measure_seconds(const std::function<std::vector<float>()> &work,
                       std::vector<float> &result_store) {
  const auto start = std::chrono::high_resolution_clock::now();
  result_store = work();
  const auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

float max_abs_diff(const std::vector<float> &baseline,
                   const std::vector<float> &candidate) {
  if (baseline.size() != candidate.size()) {
    throw std::runtime_error(
        "Result size mismatch while validating correctness");
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < baseline.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(baseline[i] - candidate[i]));
  }
  return max_diff;
}

// TODO: Create basic utils file
struct RunResult {
  std::vector<float> result;
  double seconds = 0.0;
  float diff = 0.0f;
  bool success = false;
  explicit operator bool() const noexcept { return success; }
};

std::string format_time(double seconds) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << seconds;
  return oss.str();
}

std::string format_diff(float diff) {
  std::ostringstream oss;
  oss << std::defaultfloat << std::setprecision(1) << diff;
  return oss.str();
}

void print_report(std::string_view testName, const RunResult &result) {
  if (result) {
    std::cout << testName << ": " << format_time(result.seconds)
              << " sec (diff: " << format_diff(result.diff) << ")\n";
  } else {
    std::cout << testName << ": n/a (diff: n/a)\n";
  }
}

RunResult run_test_case(const std::function<std::vector<float>()> &runner,
                        const std::vector<float> &baseline,
                        std::string_view methodName) {
  RunResult result;
  try {
    result.seconds = measure_seconds(runner, result.result);
    result.diff = max_abs_diff(baseline, result.result);
    result.success = true;
  } catch (const std::exception &ex) {
    std::cerr << methodName << " method failed: " << ex.what() << '\n';
  }
  return result;
}
}  // namespace

int main(int argc, char *argv[]) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <matrix_size_n>\n";
      return EXIT_FAILURE;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (n == 0) {
      throw std::invalid_argument("Matrix size must be positive");
    }

    const auto input = make_matrix(n);

    std::vector<float> sequential_result;
    const double sequential_seconds = measure_seconds(
        [&]() { return run_sequential(input, n); }, sequential_result);

    auto omp_res = run_test_case([&] { return run_openmp(input, n); },
                                 sequential_result, "OpenMP");
  
    std::cout << "Sequential: " << format_time(sequential_seconds) << " sec\n";
    print_report("OpenMP", omp_res);
 

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}