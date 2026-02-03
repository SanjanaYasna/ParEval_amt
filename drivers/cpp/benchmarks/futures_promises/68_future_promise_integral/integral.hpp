
/* ---- RNG helpers -------------------------------------------------------- */
struct integral_result
{
    double        accumulator = 0.0;
    std::uint64_t samples     = 0;

    double estimate(double a, double b) const
    {
        return (samples == 0)
            ? 0.0
            : (b - a) * (accumulator / static_cast<double>(samples));
    }
};

std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= (x >> 31);
    return x;
}

double sample_unit_interval(std::uint64_t index, std::uint64_t seed)
{
    std::uint64_t value = splitmix64(index + seed);
    return static_cast<double>(value >> 11) *
        (1.0 / static_cast<double>(1ull << 53));
}

