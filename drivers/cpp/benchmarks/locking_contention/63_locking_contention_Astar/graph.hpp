#pragma once

#include <cstddef>
#include <limits>
#include <vector>

struct Edge
{
    std::size_t target;
    double cost;
};

using Graph = std::vector<std::vector<Edge>>;

struct AStarResult
{
    std::vector<std::size_t> path;
    double total_cost = std::numeric_limits<double>::infinity();
    bool found = false;
};