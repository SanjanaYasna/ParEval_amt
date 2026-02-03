// Serial reference implementation ---------------------------------------------
AStarResult correctAStar(
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

    std::vector<double> g_score(
        n, std::numeric_limits<double>::infinity());
    std::vector<char> closed(n, 0);
    std::vector<std::size_t> parent(
        n, std::numeric_limits<std::size_t>::max());

    g_score[start] = 0.0;
    parent[start] = start;
    open.push({start, heuristic[start]});

    while (!open.empty())
    {
        Node current = open.top();
        open.pop();

        std::size_t u = current.id;
        if (closed[u])
            continue;
        closed[u] = 1;

        if (u == goal)
            break;

        double g_u = g_score[u];
        for (Edge const& edge : graph[u])
        {
            std::size_t v = edge.target;
            if (closed[v])
                continue;

            double tentative_g = g_u + edge.cost;
            if (tentative_g + 1e-12 < g_score[v])
            {
                g_score[v] = tentative_g;
                parent[v] = u;
                open.push({v, tentative_g + heuristic[v]});
            }
        }
    }

    AStarResult result;
    if (!closed[goal])
        return result;

    result.found = true;
    result.total_cost = g_score[goal];

    std::vector<std::size_t> path;
    for (std::size_t node = goal; node != start; node = parent[node])
    {
        if (node >= n)
            throw std::runtime_error("Broken parent chain");
        path.push_back(node);
    }
    path.push_back(start);
    std::reverse(path.begin(), path.end());
    result.path = std::move(path);
    return result;
}