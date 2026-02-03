
// Simulate one generation of Game of Life on `input`. Store results in `output`.
// input and output are flattened N x N grids in row-major order (values 0 or 1).
void gameOfLife(std::vector<int> const& input, std::vector<int> &output, std::size_t N) {
    if (N == 0) {
        output.clear();
        return;
    }
    if (input.size() != N * N) {
        throw std::invalid_argument("gameOfLife: input size does not match N*N");
    }

    output.assign(N * N, 0); // ensure correct size and initialize

    // Choose concurrency: at most N tasks (one per row) and limited by hardware
    std::size_t concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0) concurrency = 1;
    concurrency = std::min<std::size_t>(concurrency, N);

    const std::size_t rows_per_task = (N + concurrency - 1) / concurrency;

    std::vector<hpx::future<void>> futures;
    futures.reserve(concurrency);

    for (std::size_t t = 0; t < concurrency; ++t) {
        const std::size_t row_start = t * rows_per_task;
        const std::size_t row_end = std::min(row_start + rows_per_task, N);
        if (row_start >= row_end) break;

        futures.push_back(hpx::async([row_start, row_end, N, &input, &output]() {
            for (std::size_t r = row_start; r < row_end; ++r) {
                for (std::size_t c = 0; c < N; ++c) {
                    int live_neighbors = 0;
                    // examine 8 neighbors
                    for (int dr = -1; dr <= 1; ++dr) {
                        std::ptrdiff_t nr = static_cast<std::ptrdiff_t>(r) + dr;
                        if (nr < 0 || nr >= static_cast<std::ptrdiff_t>(N)) continue;
                        for (int dc = -1; dc <= 1; ++dc) {
                            std::ptrdiff_t nc = static_cast<std::ptrdiff_t>(c) + dc;
                            if (nc < 0 || nc >= static_cast<std::ptrdiff_t>(N)) continue;
                            if (dr == 0 && dc == 0) continue; // skip self
                            live_neighbors += input[static_cast<std::size_t>(nr) * N + static_cast<std::size_t>(nc)];
                        }
                    }

                    int cur = input[r * N + c];
                    int next = 0;
                    if (cur == 1) {
                        if (live_neighbors < 2) next = 0;
                        else if (live_neighbors == 2 || live_neighbors == 3) next = 1;
                        else next = 0; // >3
                    } else { // cur == 0
                        if (live_neighbors == 3) next = 1;
                        else next = 0;
                    }
                    output[r * N + c] = next;
                }
            }
        }));
    }

    hpx::wait_all(futures);
}