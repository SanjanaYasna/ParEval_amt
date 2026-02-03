
/* ---- Reference (serial) integrator ------------------------------------- */
template <typename F>
integral_result monte_carlo_integral_serial(
    std::uint64_t total_samples,
    double a,
    double b,
    std::uint64_t seed,
    F&& f)
{
    if (total_samples == 0)
        return {};

    integral_result result;
    result.samples = total_samples;

    for (std::uint64_t idx = 0; idx < total_samples; ++idx)
    {
        double const u = sample_unit_interval(idx, seed);
        double const x = a + (b - a) * u;
        result.accumulator += f(x);
    }

    return result;
}