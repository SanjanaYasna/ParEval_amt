// -----------------------------------------------------------------------------
// Serial Jacobi reference implementation
// -----------------------------------------------------------------------------
std::vector<double> jacobi_serial(
    std::size_t tiles,
    std::size_t points_per_tile,
    std::size_t iterations,
    double left_boundary,
    double right_boundary,
    std::vector<double> const& initial,
    std::vector<double> const& rhs,
    std::vector<std::vector<double>>* history = nullptr)
{
    HPX_ASSERT(tiles > 0);
    HPX_ASSERT(points_per_tile >= 2);
    HPX_ASSERT(initial.size() == tiles * points_per_tile);
    HPX_ASSERT(rhs.size() == initial.size());

    std::vector<double> current = initial;
    std::vector<double> next(current.size(), 0.0);

    if (history)
        history->clear();

    const std::size_t total_points = current.size();

    for (std::size_t iter = 0; iter < iterations; ++iter)
    {
        for (std::size_t i = 0; i < total_points; ++i)
        {
            double left  = (i == 0) ? left_boundary : current[i - 1];
            double right = (i + 1 == total_points)
                ? right_boundary : current[i + 1];

            next[i] = 0.5 * (left + right - rhs[i]);
        }

        current.swap(next);

        if (history)
            history->push_back(current);
    }

    return current;
}