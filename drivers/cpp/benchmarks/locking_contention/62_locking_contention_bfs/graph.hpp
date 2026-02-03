#pragma once
#include <vector>

struct Graph
{
    std::vector<std::vector<int>> adjacency;

    explicit Graph(std::size_t n = 0)
      : adjacency(n)
    {}

    void add_edge(int u, int v)
    {
        adjacency[u].push_back(v);
        adjacency[v].push_back(u);
    }

    std::size_t size() const noexcept
    {
        return adjacency.size();
    }
};
