void countQuadrants(std::vector<Point> const& points, std::array<std::size_t, 4> &bins) {
    bins.fill(0);
    const std::size_t n = points.size();
    if (n == 0) return;

    // Determine concurrency
    std::size_t concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0) concurrency = 1;
    concurrency = std::min(concurrency, n);

    const std::size_t chunk = (n + concurrency - 1) / concurrency;

    // Launch tasks that compute local quadrant counts
    std::vector<hpx::future<std::array<std::size_t, 4>>> futures;
    futures.reserve(concurrency);

    for (std::size_t t = 0; t < concurrency; ++t) {
        const std::size_t start = t * chunk;
        const std::size_t end = std::min(start + chunk, n);
        if (start >= end) break;

        futures.push_back(hpx::async([start, end, &points]() -> std::array<std::size_t, 4> {
            std::array<std::size_t, 4> local{};
            local.fill(0);
            for (std::size_t i = start; i < end; ++i) {
                double x = points[i].x;
                double y = points[i].y;
                if (x > 0.0) {
                    if (y > 0.0) {
                        ++local[0]; // Q1
                    } else if (y < 0.0) {
                        ++local[3]; // Q4
                    }
                } else if (x < 0.0) {
                    if (y > 0.0) {
                        ++local[1]; // Q2
                    } else if (y < 0.0) {
                        ++local[2]; // Q3
                    }
                }
                // Points with x==0 or y==0 are ignored
            }
            return local;
        }));
    }

    // Wait for all tasks and reduce into bins
    hpx::wait_all(futures);
    for (auto &f : futures) {
        auto local = f.get();
        for (std::size_t q = 0; q < bins.size(); ++q) {
            bins[q] += local[q];
        }
    }
}