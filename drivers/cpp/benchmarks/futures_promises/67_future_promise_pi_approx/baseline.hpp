// -----------------------------------------------------------------------------
// Serial reference Monte Carlo
// -----------------------------------------------------------------------------
monte_carlo_result monte_carlo_serial(
    std::uint64_t total_samples,
    std::uint64_t seed)
{
    if (total_samples == 0)
        return {};

    monte_carlo_result res{};
    res.samples = total_samples;

    for (std::uint64_t idx = 0; idx < total_samples; ++idx)
    {
        double x = sample_unit_interval(2 * idx, seed);
        double y = sample_unit_interval(2 * idx + 1, seed);

        if ((x * x + y * y) <= 1.0)
            ++res.hits;
    }

    return res;
}