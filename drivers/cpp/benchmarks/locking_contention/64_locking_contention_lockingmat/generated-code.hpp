// sum_into_row: performs atomic additions into a sparse matrix row using a per-entry HPX spinlock array.
// - row: contains pointers to the row's sorted column indices and coefficients
// - inputs: contains the input indices and coefficients to add into the row
// - entry_locks: vector of spinlocks with size == row.row_len; one lock per stored column entry
//
// The function finds each input index in the row (binary search) and, if present, locks the corresponding
// entry_lock and updates row_coefs[idx] += input_coefs[i].
//
// This function is safe to call concurrently from multiple threads as long as all threads share the
// same entry_locks instance aligned with the same row (i.e., locks[i] protects row_coefs[i]).
template<typename GlobalOrdinal, typename Scalar>
void sum_into_row(const RowData<GlobalOrdinal, Scalar>& row,
                  const InputData<GlobalOrdinal, Scalar>& inputs,
                  std::vector<hpx::lcos::local::spinlock>& entry_locks)
{
  assert(row.row_len >= 0);
  assert(static_cast<std::size_t>(row.row_len) == entry_locks.size());

  for (int i = 0; i < inputs.num_inputs; ++i) {
    const GlobalOrdinal key = inputs.input_indices[i];
    GlobalOrdinal* begin = row.row_indices;
    GlobalOrdinal* end = row.row_indices + row.row_len;
    GlobalOrdinal* loc = std::lower_bound(begin, end, key);
    std::ptrdiff_t idx = loc - begin;
    if (idx < row.row_len && *loc == key) {
      // lock the per-entry spinlock, update the coefficient, then release
      std::lock_guard<hpx::lcos::local::spinlock> guard(entry_locks[static_cast<std::size_t>(idx)]);
      row.row_coefs[static_cast<std::size_t>(idx)] += inputs.input_coefs[i];
    }
  }
}