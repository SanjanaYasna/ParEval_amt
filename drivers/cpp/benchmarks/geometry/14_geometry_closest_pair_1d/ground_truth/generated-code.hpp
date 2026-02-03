
/**
 * Return the distance between the closest two elements in the vector `x`.
 *
 * Example:
 *   input:  {7, 3, 9, 12, 31, 1}
 *   output: 2
 */
double closestPair(std::vector<double> const& x) {
    if (x.size() < 2) {
        return std::numeric_limits<double>::infinity();
    }

    // Sort the vector in parallel to ensure closest pairs are adjacent
    std::vector<double> sorted_x = x;
    hpx::sort(hpx::execution::par, sorted_x.begin(), sorted_x.end());

    // Compute distances between adjacent elements in parallel
    std::vector<double> distances(sorted_x.size() - 1);
    hpx::transform(
        hpx::execution::par,
        sorted_x.begin(), sorted_x.end() - 1,
        sorted_x.begin() + 1,
        distances.begin(),
        [](double a, double b) { return std::abs(b - a); }
    );

    // Find and return the minimum distance in parallel
    auto min_it = hpx::min_element(
        hpx::execution::par,
        distances.begin(), distances.end()
    );

    return *min_it;
}