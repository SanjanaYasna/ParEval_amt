
// ---------------------------------------------------------------------------
// Dataflow-based solver.  Everything (initialisation + time stepping) happens
// inside this function.
// ---------------------------------------------------------------------------
template <typename DataflowFactory>
std::vector<double> parallel_heat(
    std::vector<double> const& initial,
    std::size_t np, std::size_t nx, std::size_t nt,
    double k_in, double dt_in, double dx_in,
    DataflowFactory make_step)
{
    if (np == 0 || nx == 0)
        return {};

    std::size_t const total_points = np * nx;
    if (initial.size() != total_points)
    {
        HPX_THROW_EXCEPTION(hpx::invalid_status, "parallel_heat",
            "size of initial state does not match np*nx");
    }

    // update globals for the heat kernel
    k  = k_in;
    dt = dt_in;
    dx = dx_in;

    using space = std::vector<partition>;
    std::vector<space> layers(2, space(np));

    // initialise t = 0
    for (std::size_t p = 0; p < np; ++p)
    {
        partition::tile_type tile(nx);
        for (std::size_t j = 0; j < nx; ++j)
            tile[j] = initial[p * nx + j];

        layers[0][p] =
            partition(hpx::make_ready_future(std::move(tile)).share());
    }

    // time steps
    for (std::size_t t = 0; t < nt; ++t)
    {
        space const& current = layers[t % 2];
        space& next          = layers[(t + 1) % 2];

        for (std::size_t i = 0; i < np; ++i)
        {
            next[i] = partition(make_step(
                current[idx(i, -1, np)].get_future(),
                current[i].get_future(),
                current[idx(i, +1, np)].get_future()));
        }
    }

    // gather final state (t = nt)
    std::vector<double> result(total_points);
    space const& final_layer = layers[nt % 2];

    for (std::size_t p = 0; p < np; ++p)
    {
        partition::tile_type tile = final_layer[p].get_future().get();
        for (std::size_t j = 0; j < nx; ++j)
            result[p * nx + j] = tile[j];
    }

    return result;
}
