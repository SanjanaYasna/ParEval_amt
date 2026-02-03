// -----------------------------------------------------------------------------
// Serial reference wave solver (periodic)
// -----------------------------------------------------------------------------
 std::vector<double> wave_serial(
    std::vector<double> const& initial,
    std::size_t np, std::size_t nx, std::size_t nt,
    double c, double dt, double dx)
{
    HPX_ASSERT(np >= 1);
    HPX_ASSERT(nx >= 2);
    HPX_ASSERT(initial.size() == np * nx);

    std::vector<double> prev = initial;
    std::vector<double> curr = initial;
    std::vector<double> next(curr.size(), 0.0);

    std::size_t const total = curr.size();
    double const coeff = std::pow(c * dt / dx, 2);

    for (std::size_t iter = 0; iter < nt; ++iter)
    {
        for (std::size_t i = 0; i < total; ++i)
        {
            std::size_t left  = (i == 0) ? total - 1 : i - 1;
            std::size_t right = (i + 1 == total) ? 0 : i + 1;

            next[i] = 2.0 * curr[i] - prev[i]
                + coeff * (curr[left] - 2.0 * curr[i] + curr[right]);
        }

        prev.swap(curr);
        curr.swap(next);
    }

    return curr;
}