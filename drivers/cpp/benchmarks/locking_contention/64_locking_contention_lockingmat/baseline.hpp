template<typename MatrixT, typename GO = int, typename SC = double>
std::vector<SC> serial_run(
  MatrixT& matrix,
  GO target_row,
  const std::vector<GO>& in1_idx,
  const std::vector<SC>& in1_coef,
  const std::vector<GO>& in2_idx,
  const std::vector<SC>& in2_coef)
  {
  int local_int = matrix.find_row_idx(target_row);
  if (local_int < 0) return {};  // row not present locally
  std::size_t local = static_cast<std::size_t>(local_int);  
  auto& row_idx_vec = matrix.row_indices[local];
  auto& row_coefs   = matrix.row_coefs[local];

  auto accumulate = [&](const std::vector<GO>& idxs,
                        const std::vector<SC>& coefs)
  {
      std::size_t num = idxs.size();
      for (std::size_t k = 0; k < num; ++k)
      {
          GO col = idxs[k];
          SC val = coefs[k];

          auto it = std::lower_bound(row_idx_vec.begin(), row_idx_vec.end(), col);
          if (it == row_idx_vec.end() || *it != col) continue;  // not in sparsity pattern

          std::size_t pos = static_cast<std::size_t>(it - row_idx_vec.begin());
          row_coefs[pos] += val;
      }
  };

  // serially apply both input vectors
  accumulate(in1_idx, in1_coef);
  accumulate(in2_idx, in2_coef);

  return row_coefs;  // return a copy of the updated row coefficients
  }