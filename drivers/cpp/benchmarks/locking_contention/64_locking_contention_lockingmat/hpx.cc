// Driver for 64_locking_contention_lockingmat using HPX per-entry spinlocks
// #include <hpx/hpx_main.hpp>
// template<typename GO, typename SC>
// struct SimpleMatrix {
//   using rows_type = std::vector<GO>;
//   using row_indices_type = std::vector<std::vector<GO>>;
//   using row_coefs_type = std::vector<std::vector<SC>>;

//   std::vector<GO> rows;                       // global row ids stored locally
//   std::vector<std::vector<GO>> row_indices;   // per-row sorted column indices
//   std::vector<std::vector<SC>> row_coefs;     // per-row coefficients

//   // find local index for a global row id; return -1 if not present
//   int find_row_idx(GO row) const {
//     auto it = std::lower_bound(rows.begin(), rows.end(), row);
//     if (it == rows.end() || *it != row) return -1;
//     return static_cast<int>(it - rows.begin());
//   }
// };
// template<typename MatrixType>
// class LockingMatrix {
// public:
//   using spinlock = hpx::lcos::local::spinlock;

//   explicit LockingMatrix(MatrixType& matrix)
//     : matrix_(matrix)
//   {
//     // allocate per-entry unique_ptr<spinlock> to match matrix_.row_indices shape
//     row_locks_.reserve(matrix_.row_indices.size());
//     for (std::size_t r = 0; r < matrix_.row_indices.size(); ++r) {
//       std::size_t cols = matrix_.row_indices[r].size();
//       row_locks_.emplace_back();
//       row_locks_.back().reserve(cols);
//       for (std::size_t c = 0; c < cols; ++c) {
//         row_locks_.back().push_back(std::make_unique<spinlock>());
//       }
//     }
//   }
//   MatrixType& matrix_;
//   // per-row array of per-entry locks (unique_ptr to avoid non-movable spinlock issues)
//   std::vector<std::vector<std::unique_ptr<spinlock>>> row_locks_;
// };
// Accumulate contributions from multiple sparse input vectors into one row of
//    a sparse matrix concurrently. 
// Use HPX to compute in parallel.  Assume HPX has already been initialized.
//
//    Example:
//
//    input:
//       row_indices = [0, 1, 2, 3]
//       row_coefs   = [0.0, 0.0, 0.0, 0.0]
//
//       in1_idx  = [1, 3]
//       in1_coef = [1.5, 2.5]
//
//       in2_idx  = [0, 2]
//       in2_coef = [0.5, 3.0]
//
//    Exmplanation of computation:
//       entry 0: 0.0 + 0.5 = 0.5   (from in2)
//       entry 1: 0.0 + 1.5 = 1.5   (from in1)
//       entry 2: 0.0 + 3.0 = 3.0   (from in2)
//       entry 3: 0.0 + 2.5 = 2.5   (from in1)
//
//    output:
//       updated_row_coefs = [0.5, 1.5, 3.0, 2.5]
// 
//using Matrix = SimpleMatrix<int, double>;
//Matrix matrix;
//LockingMatrix<Matrix> lm(matrix);
// template<typename LM, typename MatrixT, typename GO = int, typename SC = double>
// std::vector<SC> locking_mat_add(
//     LM& lm,
//     MatrixT& matrix,
//     GO target_row,
//     const std::vector<GO>& in1_idx,
//     const std::vector<SC>& in1_coef,
//     const std::vector<GO>& in2_idx,
//     const std::vector<SC>& in2_coef){

#include <cstddef>
#include <algorithm>
#include <vector>
#include <thread>
#include <iostream>
#include <memory>
#include <cassert>
#include <mutex>                       // std::lock_guard
#include <hpx/lcos/local/spinlock.hpp> // HPX spinlock

#include <hpx/config/version.hpp>
#if HPX_VERSION_MAJOR > 1 || (HPX_VERSION_MAJOR == 1 && HPX_VERSION_MINOR >= 10)
#  include "1_10_hpx.hpp"
#else
#  include "hpx-includes.hpp"
#endif
#include "matrix_types.hpp"
#include "generated-code.hpp"
#include "baseline.hpp"
#include "utilities_old.hpp"

/*
TO RUN:
c++ -o try.o cpp/benchmarks/locking_contention/64_locking_contention_lockingmat/hpx.cc cpp/models/hpx-driver.cc -DUSE_HPX -O3 `pkg-config --cflags --libs hpx_application`    -lhpx_iostreams -Icpp  -Icpp/models -DDRIVER_PROBLEM_SIZE="(1<<9)" -w -std=c++17 -lboost_program_options  -L"$BOOST_ROOT/lib" -lhpx -lhpx_wrap  -pthread -lboost_system -lboost_filesystem

Needs libstd and boost extra flags
*/
// Context for the driver
struct Context {
  SimpleMatrix<int, double> matrix;
  std::unique_ptr< LockingMatrix< SimpleMatrix<int,double> > > lm;

  // input sets (kept as vectors so pointers remain valid)
  std::vector<int> in1_idx;
  std::vector<double> in1_coef;
  std::vector<int> in2_idx;
  std::vector<double> in2_coef;

  int target_row = 0; // global row id we update

  std::vector<double> best_vec;
  std::vector<double> compute_vec;
};

// Initialize context: build one local row with indices [0,1,2,3], zero coefficients.
// Set in1 and in2 as in the example. Create LockingMatrix wrapper.

// Context* init()
// {
//     Context* ctx = new Context();

//     // Ensure the row has at least four entries so the sample inputs remain valid.
//     std::size_t requested    = static_cast<std::size_t>(DRIVER_PROBLEM_SIZE);
//     std::size_t problem_size = std::max<std::size_t>(4u, requested);

//     ctx->matrix.rows = {0};
//     ctx->matrix.row_indices.assign(1, {});
//     ctx->matrix.row_coefs.assign(1, {});

//     auto& idx_row = ctx->matrix.row_indices[0];
//     idx_row.resize(problem_size);
//     std::iota(idx_row.begin(), idx_row.end(), 0);
//     // Make the stored column ids less uniform (still strictly increasing).
//     std::transform(idx_row.begin(), idx_row.end(), idx_row.begin(),
//                    [](int v) { return 5 + 4 * v + (v % 3); });

//     auto& coef_row = ctx->matrix.row_coefs[0];
//     coef_row.resize(problem_size);
//     std::generate(coef_row.begin(), coef_row.end(),
//                   [n = 0]() mutable {
//                       double value = 0.35 + 0.95 * static_cast<double>(n);
//                       ++n;
//                       return value;
//                   });

//     auto select_col = [&](std::size_t pos) -> int {
//         pos = std::min(pos, problem_size - 1);
//         return idx_row[pos];
//     };

//     ctx->in1_idx = {
//         select_col(std::min<std::size_t>(1, problem_size - 1)),
//         select_col(std::min<std::size_t>(3, problem_size - 1))
//     };
//     ctx->in1_coef = {1.50, 2.75};

//     std::vector<int> in2_cols;
//     for (std::size_t desired : {0u, 2u, 4u}) {
//         if (problem_size == 0) break;
//         int col = select_col(std::min(desired, problem_size - 1));
//         if (std::find(in2_cols.begin(), in2_cols.end(), col) == in2_cols.end())
//             in2_cols.push_back(col);
//         if (in2_cols.size() == 3) break;
//     }
//     for (std::size_t pos = 0; in2_cols.size() < 3 && pos < problem_size; ++pos) {
//         int col = select_col(pos);
//         if (std::find(in2_cols.begin(), in2_cols.end(), col) == in2_cols.end())
//             in2_cols.push_back(col);
//     }

//     ctx->in2_idx = in2_cols;
//     static constexpr std::array<double, 3> candidate_coefs = {0.55, 1.10, 3.45};
//     ctx->in2_coef.clear();
//     for (std::size_t i = 0; i < ctx->in2_idx.size(); ++i) {
//         ctx->in2_coef.push_back(candidate_coefs[i]);
//     }

//     ctx->lm = std::make_unique< LockingMatrix< SimpleMatrix<int, double> > >(ctx->matrix);
//     ctx->target_row = 0;

//     return ctx;
// }

Context* init()
{
    Context* ctx = new Context();

    // Ensure we always have at least four columns so the example data remains valid.
    std::size_t requested    = static_cast<std::size_t>(DRIVER_PROBLEM_SIZE);
    std::size_t problem_size = std::max<std::size_t>(4u, requested);

    ctx->matrix.rows = {0};
    ctx->matrix.row_indices.assign(1, {});
    ctx->matrix.row_coefs.assign(1, {});

    auto& idx_row = ctx->matrix.row_indices[0];
    idx_row.resize(problem_size);
    std::iota(idx_row.begin(), idx_row.end(), 0);

    ctx->matrix.row_coefs[0].assign(problem_size, 0.0);

    auto clamp_index = [&](std::size_t desired) -> int {
        return static_cast<int>(std::min<std::size_t>(desired, problem_size - 1));
    };

    ctx->in1_idx   = { clamp_index(1), clamp_index(4) };
    ctx->in1_coef  = { 1.5, 2.5 };

    ctx->in2_idx   = { clamp_index(0), clamp_index(8), clamp_index(2) };
    ctx->in2_coef  = { 0.5, 1.0, 3.0 };

    ctx->lm = std::make_unique< LockingMatrix< SimpleMatrix<int,double> > >(ctx->matrix);
    ctx->target_row = 0;
    return ctx;
}


void reset(Context* ctx) {
  if (!ctx) return;
  int local = ctx->matrix.find_row_idx(ctx->target_row);
  if (local >= 0) {
    std::fill(ctx->matrix.row_coefs[static_cast<std::size_t>(local)].begin(),
              ctx->matrix.row_coefs[static_cast<std::size_t>(local)].end(),
              0.0);
  }
}



// compute: run the parallel implementation and return final coefficients
void NO_OPTIMIZE compute(Context* ctx) {
  //assert(ctx);
  reset(ctx);                    // ensure fresh state
  ctx->compute_vec=locking_mat_add< LockingMatrix<SimpleMatrix<int,double>>, SimpleMatrix<int,double>, int, double >(
      *ctx->lm,
      ctx->matrix,
      ctx->target_row,
      ctx->in1_idx,
      ctx->in1_coef,
      ctx->in2_idx,
      ctx->in2_coef);
}

// best: run the serial (correct) implementation and return final coefficients
void NO_OPTIMIZE best(Context* ctx) {
  //assert(ctx);
  reset(ctx);                    // ensure fresh state
  ctx->best_vec=serial_run< SimpleMatrix<int,double>, int, double >(
     // *ctx->lm,
      ctx->matrix,
      ctx->target_row,
      ctx->in1_idx,
      ctx->in1_coef,
      ctx->in2_idx,
      ctx->in2_coef);

}

// -----------------------------------------------------------------------------
bool validate(Context* ctx)
{
    if (!ctx) return false;

    constexpr double tol = 1.0e-12;
    const std::array<int, 4> thread_cases{1, 2, 4, 8};
    const std::size_t trials = MAX_VALIDATION_ATTEMPTS;
    for (std::size_t attempt = 0; attempt < trials; ++attempt)
    {
        best(ctx);
        const auto& reference = ctx->best_vec;

        compute(ctx);
        const auto& trial = ctx->compute_vec;

        if (reference.size() != trial.size())
        {
            return false;
        }

        for (std::size_t i = 0; i < reference.size(); ++i)
        {
            if (std::abs(reference[i] - trial[i]) > tol)
            {
                return false;
            }
        }
    }

    return true;
}



// // Simple demonstration main: calls compute and best and prints the outputs.
// int main() {
//   Context* ctx = init_context(4);

//   std::vector<double> parallel_result = compute(ctx);
//   std::cout << "Parallel final coefficients:\n";
//   for (std::size_t i = 0; i < parallel_result.size(); ++i)
//     std::cout << " col " << ctx->matrix.row_indices[0][i] << " -> " << parallel_result[i] << "\n";

//   // Need to reset before running serial to get a fair comparison
//   reset_context(ctx);
//   std::vector<double> serial_result = best(ctx);
//   std::cout << "Serial final coefficients:\n";
//   for (std::size_t i = 0; i < serial_result.size(); ++i)
//     std::cout << " col " << ctx->matrix.row_indices[0][i] << " -> " << serial_result[i] << "\n";

void destroy(Context* ctx)
{
    delete ctx;
}