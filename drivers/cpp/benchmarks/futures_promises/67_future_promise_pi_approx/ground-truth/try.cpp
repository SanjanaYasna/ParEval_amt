template<typename LM, typename MatrixT, typename GO = int, typename SC = double>
std::vector<SC> locking_mat_add(
    LM& lm,
    MatrixT& matrix,
    GO target_row,
    const std::vector<GO>& in1_idx,
    const std::vector<SC>& in1_coef,
    const std::vector<GO>& in2_idx,
    const std::vector<SC>& in2_coef){
  std::vector<SC> updated_row_coefs(matrix.row_coefs[target_row].size());
  std::vector<SC> local_row_coefs(matrix.row_coefs[target_row].size());
  std::vector<GO> local_row_indices(matrix.row_indices[target_row].size());
  std::vector<std::unique_ptr<hpx::lcos::local::spinlock>> local_row_locks;
  local_row_locks.reserve(matrix.row_indices[target_row].size());
  for (std::size_t i = 0; i < matrix.row_indices[target_row].size(); ++i) {
    local_row_indices[i] = matrix.row_indices[target_row][i];
    local_row_coefs[i] = matrix.row_coefs[target_row][i];
    local_row_locks.push_back(std::make_unique<hpx::lcos::local::spinlock>());
  }
  hpx::parallel::for_loop(
      hpx::parallel::execution::par,
      0,
      local_row_indices.size(),
      [&](std::size_t i) {
        std::size_t idx = local_row_indices[i];
        SC coef = local_row_coefs[i];
        if (std::find(in1_idx.begin(), in1_idx.end(), idx) != in1_idx.end()) {
          coef += in1_coef[std::distance(in1_idx.begin(), std::find(in1_idx.begin(), in1_idx.end(), idx))];
        }
        if (std::find(in2_idx.begin(), in2_idx.end(), idx) != in2_idx.end()) {
          coef += in2_coef[std::distance(in2_idx.begin(), std::find(in2_idx.begin(), in2_idx.end(), idx))];
        }
        local_row_coefs[i] = coef;
      });
  for (std::size_t i = 0; i < local_row_indices.size(); ++i) {
    std::size_t idx = local_row_indices[i];
    SC coef = local_row_coefs[i];
    lm.row_locks_[target_row][idx]->lock();
    updated_row_coefs[idx] = coef;
    lm.row_locks_[target_row][idx]->unlock();
  }
  return updated_row_coefs;
}


Starcoder:
template<typename LM, typename MatrixT, typename GO = int, typename SC = double>
std::vector<SC> locking_mat_add(
    LM& lm,
    MatrixT& matrix,
    GO target_row,
    const std::vector<GO>& in1_idx,
    const std::vector<SC>& in1_coef,
    const std::vector<GO>& in2_idx,
    const std::vector<SC>& in2_coef){
  // find local index for target row
  int row_idx = lm.matrix_.find_row_idx(target_row);
  if (row_idx == -1) {
    // target row not found locally
    return {};
  }
  // get row indices and coefficients
  auto& row_indices = lm.matrix_.row_indices[row_idx];
  auto& row_coefs = lm.matrix_.row_coefs[row_idx];
  // get row locks
  auto& row_locks = lm.row_locks_[row_idx];
  // compute number of entries in target row
  std::size_t num_entries = row_indices.size();
  // compute number of entries in input vectors
  std::size_t num_in1_entries = in1_idx.size();
  std::size_t num_in2_entries = in2_idx.size();
  // allocate output vector
  std::vector<SC> updated_row_coefs(num_entries);
  // compute number of tasks
  std::size_t num_tasks = num_in1_entries + num_in2_entries;
  // allocate tasks
  std::vector<hpx::future<void>> tasks;
  tasks.reserve(num_tasks);
  // add tasks for in1
  for (std::size_t i = 0; i < num_in1_entries; ++i) {
    tasks.push_back(hpx::async([&, i] {
      // find index of entry in target row
      auto it = std::lower_bound(row_indices.begin(), row_indices.end(), in1_idx[i]);
      if (it == row_indices.end() || *it!= in1_idx[i]) {
        // entry not found in target row
        return;
      }
      // compute index of entry in target row
      std::size_t entry_idx = static_cast<std::size_t>(it - row_indices.begin());
      // get entry lock
      auto& entry_lock = row_locks[entry_idx];
      // lock entry
      entry_lock->lock();
      // update entry
      row_coefs[entry_idx] += in1_coef[i];
      // unlock entry
      entry_lock->unlock();
    }));
  }
  // add tasks for in2
  for (std::size_t i = 0; i < num_in2_entries; ++i) {
    tasks.push_back(hpx::async([&, i] {
      // find index of entry in target row
      auto it = std::lower_bound(row_indices.begin(), row_indices.end(), in2_idx[i]);
      if (it == row_indices.end() || *it!= in2_idx[i]) {
        // entry not found in target row
        return;
      }
      // compute index of entry in target row
      std::size_t entry_idx = static_cast<std::size_t>(it - row_indices.begin());
      // get entry lock
      auto& entry_lock = row_locks[entry_idx];
      // lock entry
      entry_lock->lock();
      // update entry
      row_coefs[entry_idx] += in2_coef[i];
      // unlock entry
      entry_lock->unlock();
    }));
  }
  // wait for all tasks to complete
  hpx::wait_all(tasks);
  // copy updated row coefficients to output vector
  std::copy(row_coefs.begin(), row_coefs.end(), updated_row_coefs.begin());
  // return updated row coefficients
  return updated_row_coefs;
}