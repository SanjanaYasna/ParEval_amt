// lockingmat_demo_parallelrun.cpp
// Single-file demo that uses a free parallel_run (no sum_in member).
// It uses per-entry HPX spinlocks stored as std::unique_ptr so vectors can reallocate.
//
// Compile example (adjust HPX/Boost paths as needed):
// g++ -std=c++17 -O3 lockingmat_demo_parallelrun.cpp -o locking_demo \
//   `pkg-config --cflags --libs hpx_application` -pthread
//
// If pkg-config is incomplete, add -L/path/to/hpx/lib -lhpx -lhpx_iostreams
// and Boost libs (-lboost_system -lboost_program_options -lboost_filesystem) and
// group static libs with -Wl,--start-group / --end-group as discussed earlier.

#include <cstddef>
#include <algorithm>
#include <vector>
#include <thread>
#include <iostream>
#include <memory>
#include <cassert>
#include <mutex>                       // std::lock_guard
#include <type_traits>                 // std::remove_reference_t
#include <hpx/lcos/local/spinlock.hpp> // HPX spinlock

// Minimal matrix used by the demo
template<typename GO, typename SC>
struct SimpleMatrix {
  using rows_type = std::vector<GO>;
  using row_indices_type = std::vector<std::vector<GO>>;
  using row_coefs_type = std::vector<std::vector<SC>>;

  std::vector<GO> rows;                       // global row ids stored locally
  std::vector<std::vector<GO>> row_indices;   // per-row sorted column indices
  std::vector<std::vector<SC>> row_coefs;     // per-row coefficients

  // find local index for a global row id; return -1 if not present
  int find_row_idx(GO row) const {
    auto it = std::lower_bound(rows.begin(), rows.end(), row);
    if (it == rows.end() || *it != row) return -1;
    return static_cast<int>(it - rows.begin());
  }
};

// LockingMatrix: per-entry HPX spinlocks.
// Note: row_locks_ is public so the free parallel_run can access it.
template<typename MatrixType>
class LockingMatrix {
public:
  using spinlock = hpx::lcos::local::spinlock;

  explicit LockingMatrix(MatrixType& matrix)
    : matrix_(matrix)
  {
    // allocate per-entry unique_ptr<spinlock> to match matrix_.row_indices shape
    row_locks_.reserve(matrix_.row_indices.size());
    for (std::size_t r = 0; r < matrix_.row_indices.size(); ++r) {
      std::size_t cols = matrix_.row_indices[r].size();
      row_locks_.emplace_back();
      row_locks_.back().reserve(cols);
      for (std::size_t c = 0; c < cols; ++c) {
        row_locks_.back().push_back(std::make_unique<spinlock>());
      }
    }
  }

  // Expose locks so the free parallel_run can use them.
  // (Keeping them public is the simplest change; you can provide a const accessor instead.)
  std::vector<std::vector<std::unique_ptr<spinlock>>> row_locks_;

private:
  MatrixType& matrix_;
};

// Free parallel_run that inlines the previous sum_in logic and launches threads.
// Combines the logic of sum_in into one function that launches threads
// and updates the target row's coefficients in parallel.
template<typename LM, typename MatrixT, typename GO = int, typename SC = double>
std::vector<SC> parallel_run(
    LM& lm,
    MatrixT& matrix,
    GO target_row,
    int num_threads,
    const std::vector<GO>& in1_idx,
    const std::vector<SC>& in1_coef,
    const std::vector<GO>& in2_idx,
    const std::vector<SC>& in2_coef)
{
    assert(num_threads >= 1);
    int n = std::max(1, num_threads);

    // Find the local row index once.
    int local_int = matrix.find_row_idx(target_row);
    if (local_int < 0) return {};
    std::size_t local = static_cast<std::size_t>(local_int);

    // References to the row data and locks used by the threads.
    auto& row_idx_vec = matrix.row_indices[local];
    auto& row_coefs   = matrix.row_coefs[local];
    auto& locks_row   = lm.row_locks_[local];

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(n));

    for (int t = 0; t < n; ++t) {
        const std::vector<GO>* idxs  = ((t % 2) == 0) ? &in1_idx : &in2_idx;
        const std::vector<SC>* coefs = ((t % 2) == 0) ? &in1_coef : &in2_coef;

        threads.emplace_back([&row_idx_vec, &row_coefs, &locks_row, idxs, coefs]() {
            int num_indices = static_cast<int>(idxs->size());
            for (int k = 0; k < num_indices; ++k) {
                GO col = (*idxs)[k];
                SC val = (*coefs)[k];

                auto it = std::lower_bound(row_idx_vec.begin(), row_idx_vec.end(), col);
                if (it == row_idx_vec.end() || *it != col) continue;
                std::size_t pos = static_cast<std::size_t>(it - row_idx_vec.begin());

                // Obtain raw pointer to the spinlock
                auto lock_ptr = locks_row[pos].get();
                std::lock_guard<std::remove_reference_t<decltype(*lock_ptr)>> guard(*lock_ptr);
                row_coefs[pos] += val;
            }
        });
    }

    for (auto& th : threads) th.join();

    return matrix.row_coefs[local];
}

// Helper: sequential apply of a single input vector into the matrix row (no locking required)
template<typename MatrixT, typename GO = int, typename SC = double>
void apply_input_seq(MatrixT& matrix, GO target_row, const std::vector<GO>& idxs, const std::vector<SC>& coefs) {
  int local_int = matrix.find_row_idx(target_row);
  if (local_int < 0) return;
  std::size_t local = static_cast<std::size_t>(local_int);
  auto& row_idx_vec = matrix.row_indices[local];
  auto& row_coefs   = matrix.row_coefs[local];

  int num = static_cast<int>(idxs.size());
  for (int k = 0; k < num; ++k) {
    GO col = idxs[k];
    SC val = coefs[k];
    auto it = std::lower_bound(row_idx_vec.begin(), row_idx_vec.end(), col);
    if (it == row_idx_vec.end() || *it != col) continue;
    std::size_t pos = static_cast<std::size_t>(it - row_idx_vec.begin());
    row_coefs[pos] += val;
  }
}

// Context for the driver
struct Context {
  SimpleMatrix<int, double> matrix;
  std::unique_ptr< LockingMatrix< SimpleMatrix<int,double> > > lm;

  // input sets (kept as vectors so pointers remain valid)
  std::vector<int> in1_idx;
  std::vector<double> in1_coef;
  std::vector<int> in2_idx;
  std::vector<double> in2_coef;

  int num_threads = 4;
  int target_row = 0; // global row id we update
};

// Initialize context: build one local row with indices [0,1,2,3], zero coefficients.
// Set in1 and in2 as in the example. Create LockingMatrix wrapper.
Context* init_context(int num_threads = 4) {
  Context* ctx = new Context();
  ctx->num_threads = num_threads;

  // one local row: global id 0
  ctx->matrix.rows = {0};
  ctx->matrix.row_indices.resize(1);
  ctx->matrix.row_coefs.resize(1);
  ctx->matrix.row_indices[0] = {0, 1, 2, 3};
  ctx->matrix.row_coefs[0]   = {0.0, 0.0, 0.0, 0.0};

  // inputs
  ctx->in1_idx = {1, 3};
  ctx->in1_coef = {1.5, 2.5};

  ctx->in2_idx = {0, 1, 2};
  ctx->in2_coef = {0.5, 1.0, 3.0};

  ctx->lm = std::make_unique< LockingMatrix< SimpleMatrix<int,double> > >(ctx->matrix);
  ctx->target_row = 0;
  return ctx;
}

void reset_context(Context* ctx) {
  if (!ctx) return;
  int local = ctx->matrix.find_row_idx(ctx->target_row);
  if (local >= 0) {
    std::fill(ctx->matrix.row_coefs[static_cast<std::size_t>(local)].begin(),
              ctx->matrix.row_coefs[static_cast<std::size_t>(local)].end(),
              0.0);
  }
}

// Serial implementation: apply inputs sequentially (matching the same t-loop)
std::vector<double> serial_run(Context* ctx) {
  assert(ctx && ctx->lm);
  int n = std::max(1, ctx->num_threads);
  for (int t = 0; t < n; ++t) {
    if ((t % 2) == 0) {
      apply_input_seq<SimpleMatrix<int,double>, int, double>(ctx->matrix, ctx->target_row, ctx->in1_idx, ctx->in1_coef);
    } else {
      apply_input_seq<SimpleMatrix<int,double>, int, double>(ctx->matrix, ctx->target_row, ctx->in2_idx, ctx->in2_coef);
    }
  }
  int local = ctx->matrix.find_row_idx(ctx->target_row);
  if (local < 0) return {};
  return ctx->matrix.row_coefs[static_cast<std::size_t>(local)];
}

// Parallel implementation: use the free parallel_run (no sum_in member).
std::vector<double> parallel_run_driver(Context* ctx) {
  assert(ctx && ctx->lm);
  return parallel_run< LockingMatrix<SimpleMatrix<int,double>>, SimpleMatrix<int,double>, int, double >(
      *ctx->lm,
      ctx->matrix,
      ctx->target_row,
      ctx->num_threads,
      ctx->in1_idx,
      ctx->in1_coef,
      ctx->in2_idx,
      ctx->in2_coef);
}

// compute: run the parallel implementation and return final coefficients
std::vector<double> compute(Context* ctx) {
  assert(ctx);
  reset_context(ctx);                    // ensure fresh state
  return parallel_run_driver(ctx);
}

// best: run the serial (correct) implementation and return final coefficients
std::vector<double> best(Context* ctx) {
  assert(ctx);
  reset_context(ctx);                    // ensure fresh state
  return serial_run(ctx);
}

void destroy_context(Context* ctx) {
  if (!ctx) return;
  delete ctx;
}

// Simple demonstration main: calls compute and best and prints the outputs.
int main() {
  Context* ctx = init_context(4);

  std::vector<double> parallel_result = compute(ctx);
  std::cout << "Parallel final coefficients:\n";
  for (std::size_t i = 0; i < parallel_result.size(); ++i)
    std::cout << " col " << ctx->matrix.row_indices[0][i] << " -> " << parallel_result[i] << "\n";

  // Need to reset before running serial to get a fair comparison
  reset_context(ctx);
  std::vector<double> serial_result = best(ctx);
  std::cout << "Serial final coefficients:\n";
  for (std::size_t i = 0; i < serial_result.size(); ++i)
    std::cout << " col " << ctx->matrix.row_indices[0][i] << " -> " << serial_result[i] << "\n";

  destroy_context(ctx);
  return 0;
}