int maxDegree(std::vector<int> const& A, std::size_t N)
{
    if (A.size() != N * N)
    {
        throw std::invalid_argument("A must be an N x N adjacency matrix (row-major).");
    }
    if (N == 0)
    {
        return 0;
    }

    using counting_iterator = hpx::util::counting_iterator<std::size_t>;

    return hpx::transform_reduce(
        hpx::execution::par,
        counting_iterator(0),
        counting_iterator(N),
        0,
        [](int a, int b) { return std::max(a, b); },
        [&](std::size_t row) {
            auto first = A.begin() + row * N;
            auto last  = first + N;
            return std::accumulate(first, last, 0);
        });
}