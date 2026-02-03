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

// LockingMatrix: per-entry HPX spinlocks and sum_in implementation
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
  // Thread-safe accumulation into a single row described by global_row.
  // indices/coefs describe an input vector of length num_indices.
  template<typename GO, typename SC>
  void sum_in(GO global_row,
              int num_indices,
              const GO* indices,
              const SC* coefs)
  {
    int local = matrix_.find_row_idx(global_row);
    if (local < 0) return;

    auto& row_idx_vec = matrix_.row_indices[static_cast<std::size_t>(local)];
    auto& row_coefs   = matrix_.row_coefs[static_cast<std::size_t>(local)];
    auto& locks_row   = row_locks_[static_cast<std::size_t>(local)];

    for (int k = 0; k < num_indices; ++k) {
      GO col = indices[k];
      SC val = coefs[k];

      // find position in the sorted column index list
      auto it = std::lower_bound(row_idx_vec.begin(), row_idx_vec.end(), col);
      if (it == row_idx_vec.end() || *it != col) {
        // not present in sparsity pattern; skip
        continue;
      }
      std::size_t pos = static_cast<std::size_t>(it - row_idx_vec.begin());

      // protect the single entry using the per-entry spinlock (owned by unique_ptr)
      std::lock_guard<spinlock> guard(*locks_row[pos]);
      row_coefs[pos] += val;
    }
  }

//private:
  MatrixType& matrix_;
  // per-row array of per-entry locks (unique_ptr to avoid non-movable spinlock issues)
  std::vector<std::vector<std::unique_ptr<spinlock>>> row_locks_;
};