
// ---------------------------------------------------------------------------
// Serial baseline (unchanged) for validation.
// ---------------------------------------------------------------------------
inline std::vector<double> correct_parallel_heat(
    std::vector<double> const& initial,
    std::size_t np, std::size_t nx, std::size_t nt,
    double k_in, double dt_in, double dx_in)
{
    if (np == 0 || nx == 0)
        return {};

    std::size_t const total_points = np * nx;
    if (initial.size() != total_points)
    {
        HPX_THROW_EXCEPTION(hpx::invalid_status, "correct_parallel_heat",
            "size of initial state does not match np*nx");
    }

    auto heat = [k_in, dt_in, dx_in](double left, double middle, double right) {
        return middle + (k_in * dt_in / (dx_in * dx_in)) *
            (left - 2.0 * middle + right);
    };

    std::vector<double> U0 = initial;
    std::vector<double> U1(total_points);

    for (std::size_t t = 0; t < nt; ++t)
    {
        std::vector<double>& current = (t % 2 == 0) ? U0 : U1;
        std::vector<double>& next    = (t % 2 == 0) ? U1 : U0;

        for (std::size_t i = 0; i < total_points; ++i)
        {
            std::size_t const left  = (i == 0) ? (total_points - 1) : (i - 1);
            std::size_t const right = (i + 1) % total_points;

            next[i] = heat(current[left], current[i], current[right]);
        }
    }

    return (nt % 2 == 0) ? U0 : U1;
}