
// -----------------------------------------------------------------------------
// Tile abstraction for the Jacobi sweep
// -----------------------------------------------------------------------------
struct jacobi_tile
{
    using buffer   = std::vector<double>;
    using boundary = std::array<double, 2>;   // (ghost_left, ghost_right)

    jacobi_tile* left_neighbor  = nullptr;
    jacobi_tile* right_neighbor = nullptr;

    buffer u;
    buffer rhs;

    double left_bc_  = 0.0;
    double right_bc_ = 0.0;

    hpx::shared_future<boundary> left_in_;
    hpx::shared_future<boundary> right_in_;

    hpx::lcos::local::promise<boundary> left_out_;
    hpx::lcos::local::promise<boundary> right_out_;

    hpx::shared_future<boundary> left_future_;
    hpx::shared_future<boundary> right_future_;

    jacobi_tile() = default;

    explicit jacobi_tile(std::size_t n, double init = 0.0)
      : u(n, init)
      , rhs(n, 0.0)
    {
        reset_promises();
    }

    void reset_promises()
    {
        left_out_  = hpx::lcos::local::promise<boundary>();
        right_out_ = hpx::lcos::local::promise<boundary>();

        left_future_  = left_out_.get_future().share();
        right_future_ = right_out_.get_future().share();
    }
};
