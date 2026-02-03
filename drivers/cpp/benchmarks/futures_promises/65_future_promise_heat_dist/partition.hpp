
// Global parameters (defined elsewhere, e.g. driver)
extern double k;
extern double dt;
extern double dx;

// Global coefficients (used by parallel_heat implementation)
double k  = 0.5;
double dt = 3.0;
double dx = 1.0;

inline std::size_t idx(std::size_t i, int dir, std::size_t size)
{
    if (i == 0 && dir == -1)
        return size - 1;
    if (i == size - 1 && dir == +1)
        return 0;

    HPX_ASSERT((i + dir) < size);
    return i + dir;
}

// ---------------------------------------------------------------------------
// Minimal partition wrapper
// ---------------------------------------------------------------------------
struct partition
{
    using tile_type   = std::vector<double>;
    using future_type = hpx::shared_future<tile_type>;

    partition() = default;
    explicit partition(future_type f)
      : data_(std::move(f))
    {}

    partition& operator=(future_type f)
    {
        data_ = std::move(f);
        return *this;
    }

    future_type const& get_future() const
    {
        return data_;
    }

private:
    future_type data_;
};
