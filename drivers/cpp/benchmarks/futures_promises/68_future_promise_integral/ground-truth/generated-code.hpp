#include <hpx/include/async.hpp>

#include <algorithm>   // std::min
#include <cstdint>
#include <vector>

template <typename F>
integral_result monte_carlo_integral_parallel(
    std::uint64_t total_samples,
    std::uint64_t chunk_size,
    double a,
    double b,
    std::uint64_t seed,
    F&& f)
{
    if (chunk_size == 0)
        throw std::invalid_argument("chunk_size must be positive");

    if (total_samples == 0)
        return {};

    std::uint64_t const num_chunks =
        (total_samples + chunk_size - 1) / chunk_size;

    std::vector<hpx::future<integral_result>> tasks;
    tasks.reserve(num_chunks);

    for (std::uint64_t chunk = 0; chunk < num_chunks; ++chunk)
    {
        std::uint64_t const start = chunk * chunk_size;
        if (start >= total_samples)
            break;

        std::uint64_t const count =
            std::min(chunk_size, total_samples - start);

        tasks.emplace_back(hpx::async(
            [start, count, a, b, seed,
             func = std::forward<F>(f)]() mutable
            {
                integral_result partial{};
                partial.samples = count;

                for (std::uint64_t local = 0; local < count; ++local)
                {
                    std::uint64_t const sample_index = start + local;
                    double const u  = sample_unit_interval(sample_index, seed);
                    double const x  = a + (b - a) * u;

                    partial.accumulator += func(x);
                }

                return partial;
            }));
    }

    integral_result total{};
    for (auto& fut : tasks)
    {
        integral_result partial = fut.get();
        total.samples     += partial.samples;
        total.accumulator += partial.accumulator;
    }

    return total;
}