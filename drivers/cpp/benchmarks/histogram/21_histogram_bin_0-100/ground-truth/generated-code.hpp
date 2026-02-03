
// Vector x contains values between 0 and 100, inclusive. Count the number of
// values in [0,10), [10,20), ..., [90,100] and store counts in `bins`.
void binsBy10Count(std::vector<double> const& x, std::array<std::size_t, 10> &bins) {
    bins.fill(0);
    const std::size_t n = x.size();
    if (n == 0) return;

    // Choose concurrency (number of tasks). Use hardware concurrency but not more than n.
    std::size_t concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0) concurrency = 1;
    concurrency = std::min(concurrency, n);

    const std::size_t chunk = (n + concurrency - 1) / concurrency;

    // Launch tasks; each returns a local histogram array
    std::vector<hpx::future<std::array<std::size_t, 10>>> futures;
    futures.reserve(concurrency);

    for (std::size_t t = 0; t < concurrency; ++t) {
        const std::size_t start = t * chunk;
        const std::size_t end = std::min(start + chunk, n);
        if (start >= end) break;

        futures.push_back(hpx::async([start, end, &x]() -> std::array<std::size_t, 10> {
            std::array<std::size_t, 10> local{};
            local.fill(0);
            for (std::size_t i = start; i < end; ++i) {
                double v = x[i];
                // skip out-of-range / NaN values (assumption: values are [0,100])
                if (!(v >= 0.0) || !(v <= 100.0)) continue;
                // compute bin index: 0..9, with 100 mapped to bin 9
                std::size_t idx = static_cast<std::size_t>(v / 10.0);
                if (idx >= 10) idx = 9;
                ++local[idx];
            }
            return local;
        }));
    }

    // Wait and reduce
    hpx::wait_all(futures);
    for (auto &f : futures) {
        const auto local = f.get();
        for (std::size_t b = 0; b < bins.size(); ++b) {
            bins[b] += local[b];
        }
    }
}