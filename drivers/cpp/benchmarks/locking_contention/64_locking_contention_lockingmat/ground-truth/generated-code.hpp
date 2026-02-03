
// // Parallel implementation: spawn threads which call LockingMatrix::sum_in on the context.
// std::vector<double> locking_mat_add(
//       //LM& lm,
// //     MatrixT& matrix,
// //     GO target_row,
// //     int num_threads,
// //     const std::vector<GO>& in1_idx,
// //     const std::vector<SC>& in1_coef,
// //     const std::vector<GO>& in2_idx,
// //     const std::vector<SC>& in2_coef
//       Context* ctx) {
//   using GO = int;
//   using SC = double;
//   assert(ctx && ctx->lm);

//   int n = std::max(1, ctx->num_threads);
//   std::vector<std::thread> threads;
//   threads.reserve(static_cast<std::size_t>(n));
//   for (int t = 0; t < n; ++t) {
//     if ((t % 2) == 0) {
//       threads.emplace_back([ctx]() {
//         ctx->lm->template sum_in<int,double>(ctx->target_row,
//                                              static_cast<int>(ctx->in1_idx.size()),
//                                              ctx->in1_idx.data(),
//                                              ctx->in1_coef.data());
//       });
//     } else {
//       threads.emplace_back([ctx]() {
//         ctx->lm->template sum_in<int,double>(ctx->target_row,
//                                              static_cast<int>(ctx->in2_idx.size()),
//                                              ctx->in2_idx.data(),
//                                              ctx->in2_coef.data());
//       });
//     }
//   }
//   for (auto& th : threads) th.join();

//   int local = ctx->matrix.find_row_idx(ctx->target_row);
//   if (local < 0) return {};
//   return ctx->matrix.row_coefs[static_cast<std::size_t>(local)];
// }

//NO sum_in

template<typename LM, typename MatrixT, typename GO = int, typename SC = double>
std::vector<SC> locking_mat_add(
    LM& lm,
    MatrixT& matrix,
    GO target_row,
    const std::vector<GO>& in1_idx,
    const std::vector<SC>& in1_coef,
    const std::vector<GO>& in2_idx,
    const std::vector<SC>& in2_coef)
{
    // find the target row
    int local_int = matrix.find_row_idx(target_row);
    if (local_int < 0) return {};
    std::size_t local = static_cast<std::size_t>(local_int);

    auto& row_idx_vec = matrix.row_indices[local];
    auto& row_coefs   = matrix.row_coefs[local];
    auto& locks_row   = lm.row_locks_[local];

    auto worker = [&](const std::vector<GO>& idxs,
                      const std::vector<SC>& coefs)
    {
        std::size_t num_indices = idxs.size();
        for (std::size_t k = 0; k < num_indices; ++k) {
            GO col = idxs[k];
            SC val = coefs[k];

            auto it = std::lower_bound(row_idx_vec.begin(), row_idx_vec.end(), col);
            if (it == row_idx_vec.end() || *it != col) continue;

            std::size_t pos = static_cast<std::size_t>(it - row_idx_vec.begin());
            auto lock_ptr = locks_row[pos].get();
            std::lock_guard<std::remove_reference_t<decltype(*lock_ptr)>> guard(*lock_ptr);
            row_coefs[pos] += val;
        }
    };

    std::vector<std::thread> threads;

    if (!in1_idx.empty())
        threads.emplace_back(worker, std::cref(in1_idx), std::cref(in1_coef));
    if (!in2_idx.empty())
        threads.emplace_back(worker, std::cref(in2_idx), std::cref(in2_coef));


    for (auto& th : threads) th.join();

    return row_coefs;
}