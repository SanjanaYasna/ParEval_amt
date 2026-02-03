#include <atomic>
#include <cstddef>
#include <queue>
#include <vector>

// Serial reference that still uses atomics/locking semantics for visited flags.
std::vector<std::size_t> correct_bfs_next_level_counts(Graph const& g, int source)
{
    std::size_t n = g.size();
    if (source < 0 || static_cast<std::size_t>(source) >= n)
        return {};

    std::vector<std::atomic<bool>> visited(n);
    for (auto& flag : visited)
        flag.store(false, std::memory_order_relaxed);

    if (visited[source].exchange(true, std::memory_order_acq_rel))
        return {};    // already visited? (should not happen)

    std::queue<int> frontier;
    frontier.push(source);

    std::vector<std::size_t> level_sizes;

    while (!frontier.empty())
    {
        std::size_t level_count = frontier.size();
        level_sizes.push_back(level_count);

        for (std::size_t i = 0; i < level_count; ++i)
        {
            int u = frontier.front();
            frontier.pop();

            for (int v : g.adjacency[u])
            {
                bool expected = false;
                if (visited[v].compare_exchange_strong(
                        expected, true, std::memory_order_acq_rel))
                {
                    frontier.push(v);
                }
            }
        }
    }

    return level_sizes;
}