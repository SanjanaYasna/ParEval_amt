// reference: https://en.wikipedia.org/wiki/A*_search_algorithm

// Driver for 63_parallel_astar for HPX
// #include <hpx/hpx_main.hpp>
// struct Edge
// {
//     std::size_t target;
//     double cost;
// };

// using Graph = std::vector<std::vector<Edge>>;

// struct AStarResult
// {
//     std::vector<std::size_t> path;
//     double total_cost = std::numeric_limits<double>::infinity();
//     bool found = false;
// };
//    Run an A* search on `graph` from `start` to `goal` using `heuristic`.
//    Return an AStarResult containing the path (sequence of vertex ids),
//    its total cost, and a flag indicating success.
//    The graph is represented as an adjacency list (`Graph`) of undirected edges.
// Use HPX to compute in parallel.  Assume HPX has already been initialized. 
//
//    Example:
//       graph: 0 --(1.0)-- 1 --(1.0)-- 2
//              |                        |
//             (2.5)                    (1.2)
//              |                        |
//              3 --(0.8)-- 4 --(0.4)-- 5
//       start = 0
//       goal  = 5
//       heuristic[i] = straight-line distance from i to goal
//
//    output:
//       path       = [0, 1, 2, 5]
//       total_cost = 3.2
//       found      = true
// 
// AStarResult parallel_astar(
//     Graph const& graph,
//     std::size_t start,
//     std::size_t goal,
//     std::vector<double> const& heuristic){
#include "graph.hpp"
#include <hpx/config/version.hpp>
#if HPX_VERSION_MAJOR > 1 || (HPX_VERSION_MAJOR == 1 && HPX_VERSION_MINOR >= 10)
#  include "1_10_hpx.hpp"
#else
#  include "hpx-includes.hpp"
#endif
#include "utilities_old.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"

struct Context
{
    Graph graph;
    std::vector<double> heuristic;
    std::size_t start;
    std::size_t goal;
    AStarResult result;
    std::size_t rows;
    std::size_t cols;
};

static Graph build_weighted_grid(std::size_t rows,
    std::size_t cols,
    std::vector<double>& heuristic,
    std::size_t start,
    std::size_t goal)
{
    const std::size_t n = rows * cols;
    Graph graph(n);

    const std::size_t edge_count =
        rows * (cols - 1) + cols * (rows - 1);

    std::vector<double> weights(edge_count);
    fillRand(weights, 0.5, 5.0);
    BCAST(weights, DOUBLE);

    auto index = [cols](std::size_t r, std::size_t c) {
        return r * cols + c;
    };

    std::size_t widx = 0;
    for (std::size_t r = 0; r < rows; ++r)
    {
        for (std::size_t c = 0; c < cols; ++c)
        {
            const std::size_t u = index(r, c);
            if (c + 1 < cols)
            {
                double cost = weights[widx++];
                std::size_t v = index(r, c + 1);
                graph[u].push_back({v, cost});
                graph[v].push_back({u, cost});
            }
            if (r + 1 < rows)
            {
                double cost = weights[widx++];
                std::size_t v = index(r + 1, c);
                graph[u].push_back({v, cost});
                graph[v].push_back({u, cost});
            }
        }
    }

    heuristic.resize(n);
    auto coord = [cols](std::size_t node) {
        return std::pair<std::size_t, std::size_t>{ node / cols, node % cols };
    };

    auto [goal_r, goal_c] = coord(goal);
    for (std::size_t node = 0; node < n; ++node)
    {
        auto [r, c] = coord(node);
        double dr = static_cast<double>(goal_r > r ? goal_r - r : r - goal_r);
        double dc = static_cast<double>(goal_c > c ? goal_c - c : c - goal_c);
        heuristic[node] = 1.0 * (dr + dc);    // Manhattan distance
    }

    return graph;
}

void reset(Context* ctx)
{
    ctx->start = 0;
    ctx->goal = ctx->rows * ctx->cols - 1;
    ctx->graph = build_weighted_grid(ctx->rows, ctx->cols,
        ctx->heuristic, ctx->start, ctx->goal);
    ctx->result = AStarResult{};
}

Context* init()
{
    Context* ctx = new Context();

    std::size_t dim = std::max<std::size_t>(
        4, static_cast<std::size_t>( //go for 4k + 
               std::sqrt(static_cast<double>(DRIVER_PROBLEM_SIZE))));
    ctx->rows = dim;
    ctx->cols = dim;

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context* ctx)
{
    ctx->result = parallel_astar(
        ctx->graph, ctx->start, ctx->goal, ctx->heuristic);
}

void NO_OPTIMIZE best(Context* ctx)
{
    ctx->result = correctAStar(
        ctx->graph, ctx->start, ctx->goal, ctx->heuristic);
}

static double path_cost(Graph const& graph,
    std::vector<std::size_t> const& path)
{
    if (path.size() < 2)
        return 0.0;

    double total = 0.0;
    for (std::size_t i = 0; i + 1 < path.size(); ++i)
    {
        std::size_t u = path[i];
        std::size_t v = path[i + 1];
        bool found = false;
        for (Edge const& e : graph[u])
        {
            if (e.target == v)
            {
                total += e.cost;
                found = true;
                break;
            }
        }
        if (!found)
            return std::numeric_limits<double>::infinity();
    }
    return total;
}

bool validate(Context* ctx)
{
    int rank;
    GET_RANK(rank);

    constexpr std::size_t base_dim = 8;
    const std::size_t trials = MAX_VALIDATION_ATTEMPTS;

    for (std::size_t attempt = 0; attempt < trials; ++attempt)
    {
        std::size_t dim = base_dim + attempt;
        std::size_t start = 0;
        std::size_t goal = dim * dim - 1;

        std::vector<double> heuristic;
        Graph graph = build_weighted_grid(
            dim, dim, heuristic, start, goal);

        AStarResult reference =
            correctAStar(graph, start, goal, heuristic);
        AStarResult trial_result =
            parallel_astar(graph, start, goal, heuristic);
        SYNC();

        bool isCorrect = true;
        if (IS_ROOT(rank))
        {
            if (reference.found != trial_result.found)
            {
                isCorrect = false;
            }
            else if (reference.found)
            {
                double ref_cost = path_cost(graph, reference.path);
                double trial_cost = path_cost(graph, trial_result.path);

                if (std::abs(ref_cost - reference.total_cost) > 1e-8 ||
                    std::abs(trial_cost - trial_result.total_cost) > 1e-8 ||
                    std::abs(ref_cost - trial_cost) > 1e-5)
                {
                    isCorrect = false;
                }
            }
        }

        BCAST_PTR(&isCorrect, 1, CXX_BOOL);
        if (!isCorrect)
            return false;
    }

    return true;
}

void destroy(Context* ctx)
{
    delete ctx;
}