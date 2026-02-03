std::vector<double> wave_parallel(
    std::vector<double> const& initial,
    std::size_t np, std::size_t nx, std::size_t nt,
    double c, double dt, double dx)
{
    HPX_ASSERT(np >= 1);
    HPX_ASSERT(nx >= 2);
    HPX_ASSERT(initial.size() == np * nx);

    std::vector<wave_tile> tiles;
    tiles.reserve(np);

    for (std::size_t p = 0; p < np; ++p)
    {
        tiles.emplace_back(nx);
        wave_tile& tile = tiles.back();

        auto begin = initial.begin() + p * nx;
        std::copy(begin, begin + nx, tile.prev.begin());
        std::copy(begin, begin + nx, tile.curr.begin());
    }

    // periodic neighbour links
    for (std::size_t p = 0; p < np; ++p)
    {
        tiles[p].left_neighbor  = &tiles[(p + np - 1) % np];
        tiles[p].right_neighbor = &tiles[(p + 1) % np];
    }

    // initial halos from neighbours (periodic)
    for (auto& tile : tiles)
    {
        double const left_value  = tile.left_neighbor->curr.back();
        double const right_value = tile.right_neighbor->curr.front();
        tile.set_initial_halos(left_value, right_value);
    }

    double const coeff = std::pow(c * dt / dx, 2);

    std::vector<hpx::future<void>> phase;
    phase.reserve(np);

    for (std::size_t iter = 0; iter < nt; ++iter)
    {
        phase.clear();

        for (std::size_t p = 0; p < np; ++p)
        {
            wave_tile* tile_ptr = &tiles[p];

            phase.emplace_back(
                hpx::dataflow(
                    hpx::launch::async,
                    hpx::util::unwrapping(
                        [tile_ptr, coeff](double left_ghost,
                                          double right_ghost)
                        {
                            wave_tile& tile = *tile_ptr;
                            std::size_t const N = tile.curr.size();
                            auto& next = tile.next;

                            tile.reset_promises();

                            for (std::size_t i = 1; i + 1 < N; ++i)
                            {
                                next[i] = 2.0 * tile.curr[i] - tile.prev[i] +
                                          coeff * (tile.curr[i - 1]
                                                 - 2.0 * tile.curr[i]
                                                 + tile.curr[i + 1]);
                            }

                            next.front() = 2.0 * tile.curr.front()
                                - tile.prev.front()
                                + coeff * (left_ghost - 2.0 * tile.curr.front()
                                           + tile.curr[1]);

                            next.back() = 2.0 * tile.curr.back()
                                - tile.prev.back()
                                + coeff * (tile.curr[N - 2]
                                           - 2.0 * tile.curr.back()
                                           + right_ghost);

                            tile.left_out_.set_value(next.front());
                            tile.right_out_.set_value(next.back());

                            tile.prev.swap(tile.curr);
                            tile.curr.swap(next);
                        }),
                    tile_ptr->left_in_, tile_ptr->right_in_));
        }

        hpx::wait_all(phase);

        // hand over the newly created futures to neighbours for the next step
        for (auto& tile : tiles)
        {
            tile.left_in_  = tile.left_neighbor->right_future_;
            tile.right_in_ = tile.right_neighbor->left_future_;
        }
    }

    std::vector<double> result;
    result.reserve(np * nx);
    for (auto const& tile : tiles)
        result.insert(result.end(), tile.curr.begin(), tile.curr.end());

    return result;
}