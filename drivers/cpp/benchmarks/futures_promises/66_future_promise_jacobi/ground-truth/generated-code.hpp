// //
// // -----------------------------------------------------------------------------
// // Parallel asynchronous Jacobi (no helper dependencies)
// // -----------------------------------------------------------------------------
// std::vector<double> jacobi_parallel(
//     std::size_t tiles,
//     std::size_t points_per_tile,
//     std::size_t iterations,
//     double left_boundary,
//     double right_boundary,
//     std::vector<double> const& initial,
//     std::vector<double> const& rhs,
//     std::vector<std::vector<double>>* history = nullptr)
// {
//     HPX_ASSERT(tiles > 0);
//     HPX_ASSERT(points_per_tile >= 2);
//     HPX_ASSERT(initial.size() == tiles * points_per_tile);
//     HPX_ASSERT(rhs.size() == initial.size());

//     std::vector<jacobi_tile> blocks;
//     blocks.reserve(tiles);

//     for (std::size_t t = 0; t < tiles; ++t)
//     {
//         blocks.emplace_back(points_per_tile);
//         jacobi_tile& blk = blocks.back();

//         auto begin_u = initial.begin() + t * points_per_tile;
//         std::copy(begin_u, begin_u + points_per_tile, blk.u.begin());

//         auto begin_rhs = rhs.begin() + t * points_per_tile;
//         std::copy(begin_rhs, begin_rhs + points_per_tile, blk.rhs.begin());

//         blk.left_bc_  = left_boundary;
//         blk.right_bc_ = right_boundary;
//     }

//     for (std::size_t t = 0; t < tiles; ++t)
//     {
//         jacobi_tile& blk = blocks[t];
//         blk.left_neighbor  = (t == 0) ? nullptr : &blocks[t - 1];
//         blk.right_neighbor = (t + 1 == tiles) ? nullptr : &blocks[t + 1];
//         blk.reset_promises();
//     }

//     for (std::size_t t = 0; t < tiles; ++t)
//     {
//         jacobi_tile& blk = blocks[t];

//         jacobi_tile::boundary left_halo;
//         if (blk.left_neighbor)
//         {
//             jacobi_tile const& neighbor = *blk.left_neighbor;
//             left_halo = {neighbor.u[neighbor.u.size() - 2],
//                          neighbor.u.back()};
//         }
//         else
//         {
//             left_halo = {blk.left_bc_, blk.left_bc_};
//         }

//         jacobi_tile::boundary right_halo;
//         if (blk.right_neighbor)
//         {
//             jacobi_tile const& neighbor = *blk.right_neighbor;
//             right_halo = {neighbor.u.front(),
//                           neighbor.u.size() >= 2 ? neighbor.u[1]
//                                                  : neighbor.u.front()};
//         }
//         else
//         {
//             right_halo = {blk.right_bc_, blk.right_bc_};
//         }

//         blk.left_in_  =
//             hpx::make_ready_future<jacobi_tile::boundary>(left_halo).share();
//         blk.right_in_ =
//             hpx::make_ready_future<jacobi_tile::boundary>(right_halo).share();
//     }

//     if (history)
//         history->clear();

//     std::vector<hpx::future<void>> phase;
//     phase.reserve(tiles);

//     for (std::size_t iter = 0; iter < iterations; ++iter)
//     {
//         phase.clear();

//         for (std::size_t t = 0; t < tiles; ++t)
//         {
//             jacobi_tile* tile_ptr = &blocks[t];

//             phase.emplace_back(
//                 hpx::dataflow(
//                     hpx::launch::async,
//                     hpx::util::unwrapping(
//                     [tile_ptr](jacobi_tile::boundary left,
//                                jacobi_tile::boundary right)
//                     {
//                         jacobi_tile& tile = *tile_ptr;
//                         std::size_t const N = tile.u.size();
//                         HPX_ASSERT(N >= 2);

//                         jacobi_tile::buffer next(N);

//                         const double ghost_left  = left[1];
//                         const double ghost_right = right[0];

//                         for (std::size_t i = 1; i + 1 < N; ++i)
//                         {
//                             next[i] = 0.5 *
//                                 (tile.u[i - 1] + tile.u[i + 1] - tile.rhs[i]);
//                         }

//                         next.front() = 0.5 *
//                             (ghost_left + tile.u[1] - tile.rhs.front());
//                         next.back()  = 0.5 *
//                             (tile.u[N - 2] + ghost_right - tile.rhs.back());

//                         hpx::lcos::local::promise<jacobi_tile::boundary> next_left_out;
//                         hpx::lcos::local::promise<jacobi_tile::boundary> next_right_out;

//                         auto next_left_future  =
//                             next_left_out.get_future().share();
//                         auto next_right_future =
//                             next_right_out.get_future().share();

//                         if (tile.left_neighbor)
//                         {
//                             tile.left_neighbor->right_in_ = next_left_future;
//                         }
//                         else
//                         {
//                             tile.left_in_ =
//                                 hpx::make_ready_future<jacobi_tile::boundary>(
//                                     jacobi_tile::boundary{
//                                         tile.left_bc_, tile.left_bc_})
//                                     .share();
//                         }

//                         if (tile.right_neighbor)
//                         {
//                             tile.right_neighbor->left_in_ = next_right_future;
//                         }
//                         else
//                         {
//                             tile.right_in_ =
//                                 hpx::make_ready_future<jacobi_tile::boundary>(
//                                     jacobi_tile::boundary{
//                                         tile.right_bc_, tile.right_bc_})
//                                     .share();
//                         }

//                         jacobi_tile::boundary outgoing_left =
//                             {next.front(), next[1]};
//                         jacobi_tile::boundary outgoing_right =
//                             {next[N - 2], next.back()};

//                         tile.left_out_.set_value(outgoing_left);
//                         tile.right_out_.set_value(outgoing_right);

//                         tile.u.swap(next);

//                         tile.left_out_  = std::move(next_left_out);
//                         tile.right_out_ = std::move(next_right_out);

//                         tile.left_future_  = next_left_future;
//                         tile.right_future_ = next_right_future;
//                     } ),
//                     tile_ptr->left_in_, tile_ptr->right_in_));
//         }

//         hpx::wait_all(phase);

//         if (history)
//         {
//             history->emplace_back();
//             auto& entry = history->back();
//             entry.reserve(tiles * points_per_tile);

//             for (auto const& blk : blocks)
//                 entry.insert(entry.end(), blk.u.begin(), blk.u.end());
//         }
//     }

//     std::vector<double> result;
//     result.reserve(tiles * points_per_tile);
//     for (auto const& blk : blocks)
//         result.insert(result.end(), blk.u.begin(), blk.u.end());

//     return result;
// }




// -----------------------------------------------------------------------------
std::vector<double> jacobi_parallel(
    std::size_t tiles,
    std::size_t points_per_tile,
    std::size_t iterations,
    double left_boundary,
    double right_boundary,
    std::vector<double> const& initial,
    std::vector<double> const& rhs_vec)
{
    std::size_t const total_points = tiles * points_per_tile;

    if (points_per_tile == 0 || tiles == 0)
        return {};

    if (initial.size() != total_points || rhs_vec.size() != total_points)
        throw std::invalid_argument("initial/rhs sizes do not match domain size");

    // Construct tiles and initialize their state.
    std::vector<jacobi_tile> domain;
    domain.reserve(tiles);

    for (std::size_t t = 0; t < tiles; ++t)
    {
        domain.emplace_back(points_per_tile);
        auto& tile = domain.back();

        auto const offset = t * points_per_tile;
        std::copy_n(initial.begin() + offset, points_per_tile, tile.u.begin());
        std::copy_n(rhs_vec.begin() + offset, points_per_tile, tile.rhs.begin());

        if (t == 0)
            tile.left_bc_ = left_boundary;
        if (t + 1 == tiles)
            tile.right_bc_ = right_boundary;
    }

    // Set neighbor pointers.
    for (std::size_t t = 0; t < tiles; ++t)
    {
        if (t > 0)
            domain[t].left_neighbor = &domain[t - 1];
        if (t + 1 < tiles)
            domain[t].right_neighbor = &domain[t + 1];
    }

    // Storage for the concatenated (global) solution.
    std::vector<double> solution(total_points);
    for (std::size_t t = 0; t < tiles; ++t)
    {
        std::copy(domain[t].u.begin(), domain[t].u.end(),
                  solution.begin() + t * points_per_tile);
    }

    // Perform Jacobi iterations.
    for (std::size_t iter = 0; iter < iterations; ++iter)
    {
        // Prepare communication primitives for the new iteration.
        for (auto& tile : domain)
            tile.reset_promises();

        for (std::size_t t = 0; t < tiles; ++t)
        {
            auto& tile = domain[t];

            if (tile.left_neighbor)
            {
                tile.left_in_ = tile.left_neighbor->right_future_;
            }
            else
            {
                jacobi_tile::boundary const b{{tile.left_bc_, tile.left_bc_}};
                tile.left_in_ = hpx::make_ready_future(b).share();
            }

            if (tile.right_neighbor)
            {
                tile.right_in_ = tile.right_neighbor->left_future_;
            }
            else
            {
                jacobi_tile::boundary const b{{tile.right_bc_, tile.right_bc_}};
                tile.right_in_ = hpx::make_ready_future(b).share();
            }
        }

        // Launch one HPX task per tile.
        std::vector<hpx::future<void>> tasks;
        tasks.reserve(tiles);

        for (std::size_t t = 0; t < tiles; ++t)
        {
            tasks.emplace_back(hpx::async([&, t]() {
                auto& tile = domain[t];

                // Publish our boundary values to neighbors before we start waiting.
                jacobi_tile::boundary const boundary_values{{tile.u.front(), tile.u.back()}};
                tile.left_out_.set_value(boundary_values);
                tile.right_out_.set_value(boundary_values);

                // Acquire neighbor boundary information.
                jacobi_tile::boundary const left  = tile.left_in_.get();
                jacobi_tile::boundary const right = tile.right_in_.get();

                std::size_t const n = tile.u.size();
                jacobi_tile::buffer new_values(n);

                for (std::size_t j = 0; j < n; ++j)
                {
                    double const left_val  = (j == 0)      ? left[1]        : tile.u[j - 1];
                    double const right_val = (j + 1 == n)  ? right[0]       : tile.u[j + 1];
                    new_values[j] = 0.5 * (left_val + right_val - tile.rhs[j]);
                }

                tile.u.swap(new_values);
            }));
        }

        hpx::wait_all(tasks);

        // Gather the full solution after this iteration.
        for (std::size_t t = 0; t < tiles; ++t)
        {
            auto const& tile = domain[t];
            std::copy(tile.u.begin(), tile.u.end(),
                      solution.begin() + t * points_per_tile);
        }

        // Print the iteration result (matches the requested sample output format).
        // hpx::cout << "iteration " << iter + 1 << ": [";
        // for (std::size_t i = 0; i < solution.size(); ++i)
        // {
        //     hpx::cout << std::fixed << std::setprecision(4) << solution[i];
        //     if (i + 1 != solution.size())
        //         hpx::cout << ", ";
        // }
        // hpx::cout << "]\n" << hpx::flush;
    }

    return solution;
}