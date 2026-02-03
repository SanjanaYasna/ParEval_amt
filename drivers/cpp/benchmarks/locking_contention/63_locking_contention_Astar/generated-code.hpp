

// Parallel (HPX) implementation ------------------------------------------------
AStarResult parallel_astar(
    Graph const& graph,
    std::size_t start,
    std::size_t goal,
    std::vector<double> const& heuristic)
{
    std::size_t const n = graph.size();
    if (start >= n || goal >= n)
        throw std::runtime_error("Start/goal out of range");
    if (heuristic.size() != n)
        throw std::runtime_error("Heuristic size mismatch");

    struct Node
    {
        std::size_t id;
        double f;
    };
    auto cmp = [](Node const& a, Node const& b) { return a.f > b.f; };

    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> open(cmp);
    hpx::mutex open_mutex;

    std::vector<std::atomic<double>> g_score(n);
    for (auto& g : g_score)
        g.store(std::numeric_limits<double>::infinity());

    std::vector<std::atomic<std::size_t>> parent(n);
    for (auto& p : parent)
        p.store(std::numeric_limits<std::size_t>::max());

    g_score[start].store(0.0, std::memory_order_relaxed);
    parent[start].store(start, std::memory_order_relaxed);

    {
        hpx::lock_guard<hpx::mutex> lk(open_mutex);
        open.push({start, heuristic[start]});
    }

    std::atomic<bool> goal_found{false};
    std::atomic<double> best_goal_cost{std::numeric_limits<double>::infinity()};

    std::size_t const worker_count =
        std::max<std::size_t>(hpx::get_os_thread_count(), 1);
    std::vector<hpx::future<void>> workers;
    workers.reserve(worker_count);

    for (std::size_t w = 0; w < worker_count; ++w)
    {
        workers.emplace_back(hpx::async([&, w]() {
            while (!goal_found.load(std::memory_order_acquire))
            {
                Node current;
                {
                    hpx::lock_guard<hpx::mutex> lk(open_mutex);
                    if (open.empty())
                        break;
                    current = open.top();
                    open.pop();
                }

                std::size_t u = current.id;
                double g_u = g_score[u].load(std::memory_order_acquire);

                if (current.f > g_u + heuristic[u] + 1e-12)
                    continue;

                if (u == goal)
                {
                    double candidate = g_u;
                    double prev =
                        best_goal_cost.load(std::memory_order_acquire);
                    while (candidate < prev &&
                           !best_goal_cost.compare_exchange_weak(
                               prev, candidate, std::memory_order_acq_rel))
                    {
                        /* retry */
                    }
                    goal_found.store(true, std::memory_order_release);
                    break;
                }

                for (Edge const& edge : graph[u])
                {
                    std::size_t v = edge.target;
                    double tentative_g = g_u + edge.cost;
                    double old =
                        g_score[v].load(std::memory_order_acquire);

                    while (tentative_g < old &&
                           !g_score[v].compare_exchange_weak(
                               old, tentative_g,
                               std::memory_order_acq_rel))
                    {
                        /* retry while old > tentative */
                    }

                    if (tentative_g < old)
                    {
                        parent[v].store(u, std::memory_order_release);
                        double f = tentative_g + heuristic[v];
                        hpx::lock_guard<hpx::mutex> lk(open_mutex);
                        open.push({v, f});
                    }
                }
            }
        }));
    }

    for (auto& f : workers)
        f.wait();

    AStarResult result;
    if (!goal_found.load(std::memory_order_acquire))
        return result;

    result.found = true;
    result.total_cost =
        best_goal_cost.load(std::memory_order_acquire);

    std::vector<std::size_t> reverse_path;
    for (std::size_t node = goal; node != start;
         node = parent[node].load(std::memory_order_acquire))
    {
        if (node >= n)
            throw std::runtime_error("Broken parent chain");
        reverse_path.push_back(node);
    }
    reverse_path.push_back(start);

    std::reverse(reverse_path.begin(), reverse_path.end());
    result.path = std::move(reverse_path);
    return result;
}