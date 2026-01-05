int smallestOdd(std::vector<int> const& x)
{
    if (x.empty())
        throw std::invalid_argument("smallestOdd: input vector is empty");

    constexpr int INF = std::numeric_limits<int>::max();

    int result = hpx::transform_reduce(
        hpx::execution::par,
        x.begin(), x.end(),
        INF,
        [](int a, int b) { return std::min(a, b); },
        [](int v) { return (v & 1) ? v : INF; });

    if (result == INF)
        throw std::invalid_argument("smallestOdd: no odd numbers found");

    return result;
}