inline double distance(Point const& p1, Point const& p2)
{
    return std::hypot(p2.x - p1.x, p2.y - p1.y);
}

static inline double cross(Point const& o, Point const& a, Point const& b)
{
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

/**
 * Return the perimeter of the smallest convex polygon that contains all the
 * points in the vector `points`. Uses HPX parallel algorithms where possible.
 *
 *   Example:
 *     input:  {{0, 3}, {1, 1}, {2, 2}, {4, 4}, {0, 0}, {1, 2}, {3, 1}, {3, 3}}
 *     output: 13.4477
 */
double convexHullPerimeter(std::vector<Point> const& points)
{
    if (points.size() < 2)
        return 0.0;

    // Work on a copy so that the original data remains untouched.
    std::vector<Point> pts(points.begin(), points.end());

    // Parallel sort by x, then by y.
    hpx::parallel::rangev1::sort(hpx::execution::par, pts,
        [](Point const& lhs, Point const& rhs) {
            if (lhs.x == rhs.x)
                return lhs.y < rhs.y;
            return lhs.x < rhs.x;
        });

    // Remove duplicates (sequential; negligible cost compared to sort).
    pts.erase(std::unique(pts.begin(), pts.end(),
                  [](Point const& lhs, Point const& rhs) {
                      return lhs.x == rhs.x && lhs.y == rhs.y;
                  }),
        pts.end());

    if (pts.size() < 2)
        return 0.0;

    // Monotonic chain construction of the convex hull.
    std::vector<Point> hull;
    hull.reserve(2 * pts.size());

    // Lower hull.
    for (auto const& p : pts)
    {
        while (hull.size() >= 2 &&
            cross(hull[hull.size() - 2], hull.back(), p) <= 0.0)
        {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    // Upper hull.
    std::size_t lower_size = hull.size();
    for (std::size_t i = pts.size(); i > 0; --i)
    {
        Point const& p = pts[i - 1];
        while (hull.size() > lower_size &&
            cross(hull[hull.size() - 2], hull.back(), p) <= 0.0)
        {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    // The first point is now duplicated at the end; remove it.
    if (!hull.empty())
        hull.pop_back();

    if (hull.size() < 2)
        return 0.0;

    // Parallel transform-reduce to accumulate perimeter distances.
    std::vector<std::size_t> indices(hull.size());
    std::iota(indices.begin(), indices.end(), 0);

    double perimeter = hpx::transform_reduce(
        hpx::execution::par, indices.begin(), indices.end(), 0.0,
        std::plus<double>(),
        [&](std::size_t idx) {
            std::size_t next = (idx + 1) % hull.size();
            return distance(hull[idx], hull[next]);
        });

    return perimeter;
}