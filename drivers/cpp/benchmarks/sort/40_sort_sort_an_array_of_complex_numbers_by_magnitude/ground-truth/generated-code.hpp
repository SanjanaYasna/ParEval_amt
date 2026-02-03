
// Sort the vector x of complex numbers by their magnitude in ascending order.
// Uses a parallel sort-by-chunks + parallel k-way (pairwise) merge approach.
// Magnitudes compared using std::norm (squared magnitude) to avoid sqrt.
void sortComplexByMagnitude(std::vector<std::complex<double>> &x) {
    const std::size_t n = x.size();
    if (n <= 1) return;

    // Comparator based on squared magnitude
    auto cmp = [](const std::complex<double> &a, const std::complex<double> &b) {
        return std::norm(a) < std::norm(b);
    };

    // Determine concurrency: at most n chunks
    std::size_t concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0) concurrency = 1;
    std::size_t num_chunks = std::min<std::size_t>(concurrency, n);

    const std::size_t chunk = (n + num_chunks - 1) / num_chunks;

    // Phase 1: sort each chunk in parallel
    std::vector<hpx::future<void>> sort_futs;
    sort_futs.reserve(num_chunks);
    for (std::size_t t = 0; t < num_chunks; ++t) {
        const std::size_t start = t * chunk;
        const std::size_t end = std::min(start + chunk, n);
        if (start >= end) break;
        sort_futs.push_back(hpx::async([start, end, &x, &cmp]() {
            std::sort(x.begin() + start, x.begin() + end, cmp);
        }));
    }
    hpx::wait_all(sort_futs);

    // Phase 2: iterative pairwise merge passes (parallel merges per pass)
    std::vector<std::complex<double>> buffer(n);
    std::vector<std::complex<double>> *src = &x;
    std::vector<std::complex<double>> *dst = &buffer;

    for (std::size_t width = chunk; width < n; width *= 2) {
        std::vector<hpx::future<void>> merge_futs;
        // For each pair of runs [start, mid) and [mid, end)
        for (std::size_t start = 0; start < n; start += 2 * width) {
            const std::size_t mid = std::min(start + width, n);
            const std::size_t end = std::min(start + 2 * width, n);

            if (mid >= end) {
                // single run remains, just copy it
                merge_futs.push_back(hpx::async([start, end, src, dst]() {
                    std::copy(src->begin() + start, src->begin() + end, dst->begin() + start);
                }));
            } else {
                // merge two sorted subranges into dst
                merge_futs.push_back(hpx::async([start, mid, end, src, dst, &cmp]() {
                    std::merge(
                        src->begin() + start, src->begin() + mid,
                        src->begin() + mid,   src->begin() + end,
                        dst->begin() + start,
                        cmp
                    );
                }));
            }
        }
        hpx::wait_all(merge_futs);
        // swap source and destination for next pass
        std::swap(src, dst);
    }

    // If final result is in buffer (src points to buffer), move it back to x
    if (src != &x) {
        x.swap(*src); // efficient swap; x now contains sorted data
    }
}