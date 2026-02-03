// reference: https://hpx-docs.stellar-group.org/latest/html/examples/breadth_first_search.html

// // Driver for 62_bfs_next_level for HPX
// #include <hpx/hpx_main.hpp>
// struct Graph
// {
//     std::vector<std::vector<int>> adjacency;

//     explicit Graph(std::size_t n = 0)
//       : adjacency(n)
//     {}

//     void add_edge(int u, int v)
//     {
//         adjacency[u].push_back(v);
//         adjacency[v].push_back(u);
//     }

//     std::size_t size() const noexcept
//     {
//         return adjacency.size();
//     }
// };
//    Perform a level-synchronous breadth-first search starting from `source`
//    and return a vector whose ith entry is the number of vertices in level i.
//    The graph is represented as an adjacency list (`Graph`) of undirected edges.
//    Use HPX to compute in parallel.  Assume HPX has already been initialized.
//
//    Example:
//
//    input:
//       Graph: 0--1--2
//              |  |  |
//              3--4--5
//       source = 0
//
//    output:
//       level counts = [1, 2, 3]
// std::vector<std::size_t> bfs_next_level_counts(Graph const& g, int source){

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
    int source;
    std::vector<std::size_t> levelCounts; //what's being tested and stored into 
    std::size_t rows;
    std::size_t cols;
};

static Graph build_grid_graph(std::size_t rows, std::size_t cols)
{
    Graph g(rows * cols);

    auto index = [cols](std::size_t r, std::size_t c) {
        return static_cast<int>(r * cols + c);
    };

    for (std::size_t r = 0; r < rows; ++r)
    {
        for (std::size_t c = 0; c < cols; ++c)
        {
            int u = index(r, c);
            if (c + 1 < cols)
                g.add_edge(u, index(r, c + 1));
            if (r + 1 < rows)
                g.add_edge(u, index(r + 1, c));
        }
    }
    return g;
}

void reset(Context* ctx)
{
    ctx->graph  = build_grid_graph(ctx->rows, ctx->cols);
    ctx->source = 0;
    ctx->levelCounts.clear();
}

Context* init()
{
    Context* ctx = new Context();
//stick to no more than 2^8 
    std::size_t dim = std::max<std::size_t>(4,
        static_cast<std::size_t>(std::sqrt(static_cast<double>(DRIVER_PROBLEM_SIZE))));
    ctx->rows = dim;
    ctx->cols = dim;

    reset(ctx);
    return ctx;
}
//not really used...
void NO_OPTIMIZE compute(Context* ctx)
{
    ctx->levelCounts = bfs_next_level_counts(ctx->graph, ctx->source);
}

void NO_OPTIMIZE best(Context* ctx)
{
    ctx->levelCounts = correct_bfs_next_level_counts(ctx->graph, ctx->source);
}

bool validate(Context* ctx)
{
    int rank;
    GET_RANK(rank);

    constexpr std::size_t baseDim = 256;
    const std::size_t trials      = MAX_VALIDATION_ATTEMPTS;

    for (std::size_t attempt = 0; attempt < trials; ++attempt)
    {
        std::size_t dim = baseDim + attempt * 2;
        Graph g = build_grid_graph(dim, dim);

        int source = static_cast<int>((attempt * 7) % (dim * dim));

        auto reference = correct_bfs_next_level_counts(g, source);
        auto trial     = bfs_next_level_counts(g, source);
        SYNC();

        bool isCorrect = true;
        if (IS_ROOT(rank) &&
            (reference.size() != trial.size() ||
             !std::equal(reference.begin(), reference.end(), trial.begin())))
        {
            isCorrect = false;
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