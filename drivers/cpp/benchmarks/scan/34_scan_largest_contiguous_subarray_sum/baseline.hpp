#pragma once
#include <vector>
#include <limits>

/* Return the largest sum of any contiguous subarray in the vector x.
   i.e. if x=[−2, 1, −3, 4, −1, 2, 1, −5, 4] then [4, −1, 2, 1] is the contiguous
   subarray with the largest sum of 6.
   Example:

   input: [−2, 1, −3, 4, −1, 2, 1, −5, 4]
   output: 6
*/
int NO_INLINE correctMaximumSubarray(std::vector<int> const& x) {
    int largestSum = std::numeric_limits<int>::lowest();
    for (int i = 0; i < x.size(); i++) {
        int currSum = 0;
        for (int j = i; j < x.size(); j++) {
            currSum += x[j];
            if (currSum > largestSum) largestSum = currSum;
        }
    }
    return largestSum;
}



template<typename LM, typename MatrixT, typename GO = int, typename SC = double>\nstd::vector<SC> locking_mat_add(\n    LM& lm,\n    MatrixT& matrix,\n    GO target_row,\n    const std::vector<GO>& in1_idx,\n    const std::vector<SC>& in1_coef,\n    const std::vector<GO>& in2_idx,\n    const std::vector<SC>& in2_coef){\n  // find local index for target row\n  int row_idx = lm.matrix_.find_row_idx(target_row);\n  if (row_idx == -1) {\n    // target row not found locally\n    return {};\n  }\n  // get row indices and coefficients\n  auto& row_indices = lm.matrix_.row_indices[row_idx];\n  auto& row_coefs = lm.matrix_.row_coefs[row_idx];\n  // get row locks\n  auto& row_locks = lm.row_locks_[row_idx];\n  // compute number of entries in target row\n  std::size_t num_entries = row_indices.size();\n  // compute number of entries in input vectors\n  std::size_t num_in1_entries = in1_idx.size();\n  std::size_t num_in2_entries = in2_idx.size();\n  // allocate output vector\n  std::vector<SC> updated_row_coefs(num_entries);\n  // compute number of tasks\n  std::size_t num_tasks = num_in1_entries + num_in2_entries;\n  // allocate tasks\n  std::vector<hpx::future<void>> tasks;\n  tasks.reserve(num_tasks);\n  // add tasks for in1\n  for (std::size_t i = 0; i < num_in1_entries; ++i) {\n    tasks.push_back(hpx::async([&, i] {\n      // find index of entry in target row\n      auto it = std::lower_bound(row_indices.begin(), row_indices.end(), in1_idx[i]);\n      if (it == row_indices.end() || *it!= in1_idx[i]) {\n        // entry not found in target row\n        return;\n      }\n      // compute index of entry in target row\n      std::size_t entry_idx = static_cast<std::size_t>(it - row_indices.begin());\n      // get entry lock\n      auto& entry_lock = row_locks[entry_idx];\n      // lock entry\n      entry_lock->lock();\n      // update entry\n      row_coefs[entry_idx] += in1_coef[i];\n      // unlock entry\n      entry_lock->unlock();\n    }));\n  }\n  // add tasks for in2\n  for (std::size_t i = 0; i < num_in2_entries; ++i) {\n    tasks.push_back(hpx::async([&, i] {\n      // find index of entry in target row\n      auto it = std::lower_bound(row_indices.begin(), row_indices.end(), in2_idx[i]);\n      if (it == row_indices.end() || *it!= in2_idx[i]) {\n        // entry not found in target row\n        return;\n      }\n      // compute index of entry in target row\n      std::size_t entry_idx = static_cast<std::size_t>(it - row_indices.begin());\n      // get entry lock\n      auto& entry_lock = row_locks[entry_idx];\n      // lock entry\n      entry_lock->lock();\n      // update entry\n      row_coefs[entry_idx] += in2_coef[i];\n      // unlock entry\n      entry_lock->unlock();\n    }));\n  }\n  // wait for all tasks to complete\n  hpx::wait_all(tasks);\n  // copy updated row coefficients to output vector\n  std::copy(row_coefs.begin(), row_coefs.end(), updated_row_coefs.begin());\n  // return updated row coefficients\n  return updated_row_coefs;\n}",
    