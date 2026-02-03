// Apply isPowerOfTwo to every value in x and store results in mask.
// Note: vector<bool> is bit-packed and not safe for concurrent bit writes,
// so we write into a byte-backed temporary in parallel and populate mask
// sequentially at the end.
void mapPowersOfTwo(std::vector<int> const& x, std::vector<bool> &mask) {
    const std::size_t n = x.size();
    mask.assign(n, false);
    if (n == 0) return;

    // Determine concurrency
    std::size_t concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0) concurrency = 1;
    concurrency = std::min<std::size_t>(concurrency, n);

    const std::size_t chunk = (n + concurrency - 1) / concurrency;

    // temporary byte-backed results to avoid races on vector<bool>
    std::vector<unsigned char> tmp(n, 0);

    std::vector<hpx::future<void>> futs;
    futs.reserve(concurrency);

    for (std::size_t t = 0; t < concurrency; ++t) {
        const std::size_t start = t * chunk;
        const std::size_t end = std::min(start + chunk, n);
        if (start >= end) break;

        futs.push_back(hpx::async([start, end, &x, &tmp]() {
            for (std::size_t i = start; i < end; ++i) {
                tmp[i] = isPowerOfTwo(x[i]) ? 1u : 0u;
            }
        }));
    }

    hpx::wait_all(futs);

    // Write back into vector<bool> (sequential to avoid bit-write races)
    for (std::size_t i = 0; i < n; ++i) {
        mask[i] = (tmp[i] != 0);
    }
}