
// Compute the prefix sum array of x and return the sum of that prefix-sum array.
// Example:
// input: [-7, 2, 1, 9, 4, 8]
// prefix: [-7, -5, -4, 5, 9, 17]
// output: 15
double sumOfPrefixSum(std::vector<double> const& x) {
    const std::size_t n = x.size();
    if (n == 0) return 0.0;

    // Determine concurrency (at most n tasks)
    std::size_t concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0) concurrency = 1;
    concurrency = std::min(concurrency, n);

    const std::size_t chunk = (n + concurrency - 1) / concurrency;
    const std::size_t num_blocks = (n + chunk - 1) / chunk;

    // Output prefix array
    std::vector<double> prefix(n);

    // Phase 1: compute local prefix sums per block and return block sums
    std::vector<hpx::future<double>> block_futures;
    block_futures.reserve(num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        const std::size_t start = b * chunk;
        const std::size_t end = std::min(start + chunk, n);
        // Each task writes only into prefix[start..end-1] -> no races between tasks
        block_futures.push_back(hpx::async([start, end, &x, &prefix]() -> double {
            double local_sum = 0.0;
            for (std::size_t i = start; i < end; ++i) {
                local_sum += x[i];
                prefix[i] = local_sum; // local prefix within block
            }
            return local_sum; // block total
        }));
    }

    // Wait for all blocks to finish computing local prefixes and gather block sums
    hpx::wait_all(block_futures);
    std::vector<double> block_sums(num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        block_sums[b] = block_futures[b].get();
    }

    // Phase 2: compute offsets (exclusive prefix of block sums)
    std::vector<double> offsets(num_blocks, 0.0);
    for (std::size_t b = 1; b < num_blocks; ++b) {
        offsets[b] = offsets[b - 1] + block_sums[b - 1];
    }

    // Phase 3: add offsets to each block's local prefixes (skip offset 0)
    std::vector<hpx::future<void>> adjust_futures;
    adjust_futures.reserve(num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        const double off = offsets[b];
        if (off == 0.0) continue; // nothing to do for the first block (or zero offset)
        const std::size_t start = b * chunk;
        const std::size_t end = std::min(start + chunk, n);
        adjust_futures.push_back(hpx::async([start, end, off, &prefix]() {
            for (std::size_t i = start; i < end; ++i) {
                prefix[i] += off;
            }
        }));
    }

    hpx::wait_all(adjust_futures);

    // Finally, compute the sum of the prefix array
    double total = std::accumulate(prefix.begin(), prefix.end(), 0.0);
    return total;
}