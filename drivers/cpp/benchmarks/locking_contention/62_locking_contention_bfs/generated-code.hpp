// Parallel (HPX) version
std::vector<std::size_t> bfs_next_level_counts(Graph const& g, int source)
{
    std::size_t n = g.size();
    if (source < 0 || static_cast<std::size_t>(source) >= n)
        return {};

    std::vector<std::atomic<bool>> visited(n);
    for (auto& flag : visited)
        flag.store(false, std::memory_order_relaxed);
    visited[source].store(true, std::memory_order_relaxed);

    std::vector<int> frontier{source};
    std::vector<std::size_t> level_sizes;
    std::vector<int> scratch(n, -1);

    while (!frontier.empty())
    {
        level_sizes.push_back(frontier.size());

        std::atomic<std::size_t> next_count{0};

        hpx::for_loop(hpx::execution::par, std::size_t(0), frontier.size(),
            [&](std::size_t idx)
            {
                int u = frontier[idx];
                for (int v : g.adjacency[u])
                {
                    bool expected = false;
                    if (visited[v].compare_exchange_strong(
                            expected, true, std::memory_order_acq_rel))
                    {
                        std::size_t slot =
                            next_count.fetch_add(1, std::memory_order_relaxed);
                        scratch[slot] = v;
                    }
                }
            });

        std::size_t count = next_count.load(std::memory_order_relaxed);
        frontier.assign(scratch.begin(), scratch.begin() + count);
    }

    return level_sizes;
}