
monte_carlo_result monte_carlo_parallel(
    std::uint64_t total_samples,
    std::uint64_t chunk_size,
    std::uint64_t seed)
{
    if (chunk_size == 0)
        throw std::invalid_argument("chunk_size must be positive");
    if (total_samples == 0)
        return {};

    std::uint64_t num_chunks =
        (total_samples + chunk_size - 1) / chunk_size;

    std::vector<hpx::future<monte_carlo_result>> futures;
    futures.reserve(num_chunks);

    for (std::uint64_t chunk = 0; chunk < num_chunks; ++chunk)
    {
        std::uint64_t const start =
            std::min(chunk * chunk_size, total_samples);

        std::uint64_t const samples_here =
            std::min(chunk_size, total_samples - start);

        futures.emplace_back(
            hpx::async(
                [start, samples_here, seed]() {
                    monte_carlo_result res{};
                    res.samples = samples_here;

                    for (std::uint64_t j = 0; j < samples_here; ++j)
                    {
                        std::uint64_t const idx = start + j;
                        double const x = sample_unit_interval(2 * idx,     seed);
                        double const y = sample_unit_interval(2 * idx + 1, seed);

                        if ((x * x + y * y) <= 1.0)
                            ++res.hits;
                    }

                    return res;
                }));
    }

    monte_carlo_result total{};
    for (auto& f : futures)
    {
        monte_carlo_result part = f.get();
        total.hits    += part.hits;
        total.samples += part.samples;
    }

    return total;
}
