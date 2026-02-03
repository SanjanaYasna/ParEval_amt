bool xorContains(std::vector<int> const& x, std::vector<int> const& y, int val) {
    // Launch parallel searches for val in x and y.
    auto fx = hpx::async([&x, val]() {
        return std::any_of(x.begin(), x.end(), [val](int a) { return a == val; });
    });
    auto fy = hpx::async([&y, val]() {
        return std::any_of(y.begin(), y.end(), [val](int a) { return a == val; });
    });

    // Wait and combine results: true if exactly one contains val.
    bool in_x = fx.get();
    bool in_y = fy.get();
    return in_x != in_y;
}