// -----------------------------------------------------------------------------
// Tile abstraction used by the wave solver
// -----------------------------------------------------------------------------
struct wave_tile
{
    using buffer = std::vector<double>;

    wave_tile* left_neighbor  = nullptr;
    wave_tile* right_neighbor = nullptr;

    buffer prev;
    buffer curr;
    buffer next;   // reusable scratch buffer

    hpx::shared_future<double> left_in_;
    hpx::shared_future<double> right_in_;

    hpx::lcos::local::promise<double> left_out_;
    hpx::lcos::local::promise<double> right_out_;

    hpx::shared_future<double> left_future_;
    hpx::shared_future<double> right_future_;

    wave_tile() = default;

    explicit wave_tile(std::size_t n)
      : prev(n, 0.0)
      , curr(n, 0.0)
      , next(n, 0.0)
    {}

    void set_initial_halos(double left_value, double right_value)
    {
        left_in_  = hpx::make_ready_future(left_value).share();
        right_in_ = hpx::make_ready_future(right_value).share();
    }

    void reset_promises()
    {
        left_out_  = hpx::lcos::local::promise<double>();
        right_out_ = hpx::lcos::local::promise<double>();

        left_future_  = left_out_.get_future().share();
        right_future_ = right_out_.get_future().share();
    }
};