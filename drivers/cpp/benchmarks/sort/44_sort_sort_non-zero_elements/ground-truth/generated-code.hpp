
// Sort the vector x in ascending order ignoring elements with value 0.
// Zero elements remain at their original indices.
void sortIgnoreZero(std::vector<int> &x) {
    const std::size_t n = x.size();
    if (n <= 1) return;

    // Gather indices and values of non-zero elements
    std::vector<std::size_t> idx;
    std::vector<int> vals;
    idx.reserve(n);
    vals.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (x[i] != 0) {
            idx.push_back(i);
            vals.push_back(x[i]);
        }
    }

    const std::size_t m = vals.size();
    if (m <= 1) return; // nothing to sort or only one non-zero

    // Determine concurrency (at most m chunks)
    std::size_t concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0) concurrency = 1;
    std::size_t num_chunks = std::min<std::size_t>(concurrency, m);

    const std::size_t chunk = (m + num_chunks - 1) / num_chunks;

    // Phase 1: sort each chunk in parallel
    std::vector<hpx::future<void>> sort_futs;
    sort_futs.reserve(num_chunks);
    for (std::size_t t = 0; t < num_chunks; ++t) {
        const std::size_t start = t * chunk;
        const std::size_t end = std::min(start + chunk, m);
        if (start >= end) break;
        sort_futs.push_back(hpx::async([start, end, &vals]() {
            std::sort(vals.begin() + start, vals.begin() + end);
        }));
    }
    hpx::wait_all(sort_futs);

    // Phase 2: iterative pairwise merge passes (parallel merges per pass)
    std::vector<int> buffer(m);
    std::vector<int>* src = &vals;
    std::vector<int>* dst = &buffer;

    for (std::size_t width = chunk; width < m; width *= 2) {
        std::vector<hpx::future<void>> merge_futs;
        for (std::size_t start = 0; start < m; start += 2 * width) {
            const std::size_t mid = std::min(start + width, m);
            const std::size_t end = std::min(start + 2 * width, m);

            if (mid >= end) {
                // single run remains, copy it
                merge_futs.push_back(hpx::async([start, end, src, dst]() {
                    std::copy(src->begin() + start, src->begin() + end, dst->begin() + start);
                }));
            } else {
                // merge two sorted subranges into dst
                merge_futs.push_back(hpx::async([start, mid, end, src, dst]() {
                    std::merge(
                        src->begin() + start, src->begin() + mid,
                        src->begin() + mid,   src->begin() + end,
                        dst->begin() + start
                    );
                }));
            }
        }
        hpx::wait_all(merge_futs);
        std::swap(src, dst);
    }

    // src now points to the sorted values (either vals or buffer)
    // Phase 3: write sorted values back into original vector at the recorded indices
    std::vector<hpx::future<void>> write_futs;
    write_futs.reserve(num_chunks);
    // reuse chunk size based on m
    const std::size_t write_chunk = (m + num_chunks - 1) / num_chunks;
    for (std::size_t t = 0; t < num_chunks; ++t) {
        const std::size_t start = t * write_chunk;
        const std::size_t end = std::min(start + write_chunk, m);
        if (start >= end) break;
        write_futs.push_back(hpx::async([start, end, &idx, src, &x]() {
            for (std::size_t i = start; i < end; ++i) {
                x[idx[i]] = (*src)[i];
            }
        }));
    }
    hpx::wait_all(write_futs);
}