// simple 1D data tile for an asynchronous Jacobi sweep
/*
#include <hpx/hpx_main.hpp>
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
Implement a 1‑D Jacobi relaxation using HPX where the domain is partitioned 
into tiles and the start of each iteration has the update rule:
u_new[i] = 0.5 * (u_old[i-1] + u_old[i+1] - rhs[i])
Use HPX to compute in parallel.  Assume HPX has already been initialized.

Example: 
    input:
    tiles = 2
    points_per_tile = 4    # interior points per tile
    iterations = 3
    left_boundary = 1.0
    right_boundary = 0.0
    rhs = all zeros
    initial interior = all zeros
    Output: 
    iteration 1: [0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    iteration 2: [0.5000, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    iteration 3: [0.6250, 0.2500, 0.1250, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

std::vector<double> jacobi_parallel(
    std::size_t tiles,
    std::size_t points_per_tile,
    std::size_t iterations,
    double left_boundary,
    double right_boundary,
    std::vector<double> const& initial,
    std::vector<double> const& rhs) 
{
*/
#pragma once


#include <hpx/config/version.hpp>
#if HPX_VERSION_MAJOR > 1 || (HPX_VERSION_MAJOR == 1 && HPX_VERSION_MINOR >= 10)
#  include "1_10_hpx.hpp"
#else
#  include "hpx-includes.hpp"
#endif
#include "jacobi.hpp"
#include "utilities_old.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"    // <--- include the simplified solver above

#include <hpx/assert.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/pack_traversal/unwrap.hpp> 


// -----------------------------------------------------------------------------
// Driver context, lifecycle, validation
// -----------------------------------------------------------------------------
struct Context
{
    std::size_t tiles            = 2;
    std::size_t points_per_tile  = DRIVER_PROBLEM_SIZE / 128;
    std::size_t iterations       = 1;
    double left_boundary         = 12.0;
    double right_boundary        = 30.0;

    std::vector<double> initial;
    std::vector<double> rhs;
    std::vector<double> result;
    std::vector<double> reference;
};

std::size_t total_points(Context const* ctx)
{
    return ctx->tiles * ctx->points_per_tile;
}

void reset(Context* ctx)
{
    const std::size_t total = total_points(ctx);

    ctx->initial.resize(total);
    ctx->rhs.assign(total, 0.0);

    for (std::size_t i = 0; i < total; ++i)
        ctx->initial[i] = static_cast<double>(i % ctx->points_per_tile);

    ctx->result.assign(total, 0.0);
    ctx->reference.assign(total, 0.0);
}


Context* init()
{
    Context* ctx = new Context;

    // ctx->tiles           = 3;
    // ctx->points_per_tile = DRIVER_PROBLEM_SIZE / 2;
    // ctx->iterations      = 3;
    // ctx->left_boundary   = 12.0;
    // ctx->right_boundary  = 50.0;

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context* ctx)
{
    ctx->result = jacobi_parallel(
        ctx->tiles, ctx->points_per_tile, ctx->iterations,
        ctx->left_boundary, ctx->right_boundary,
        ctx->initial, ctx->rhs);
}

void NO_OPTIMIZE best(Context* ctx)
{
    ctx->reference = jacobi_serial(
        ctx->tiles, ctx->points_per_tile, ctx->iterations,
        ctx->left_boundary, ctx->right_boundary,
        ctx->initial, ctx->rhs);
}

bool validate(Context* ctx)
{
    constexpr std::size_t attempts  = 4;
    constexpr double tolerance      = 1e-9;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    const std::size_t total = total_points(ctx);
    std::vector<double> random_initial(total);
    std::vector<double> random_rhs(total);

    for (std::size_t attempt = 0; attempt < attempts; ++attempt)
    {
        for (std::size_t i = 0; i < total; ++i)
        {
            random_initial[i] = dist(rng);
            random_rhs[i]     = 0.5 * dist(rng);
        }

        auto ref = jacobi_serial(
            ctx->tiles, ctx->points_per_tile, ctx->iterations,
            ctx->left_boundary, ctx->right_boundary,
            random_initial, random_rhs);

        auto trial = jacobi_parallel(
            ctx->tiles, ctx->points_per_tile, ctx->iterations,
            ctx->left_boundary, ctx->right_boundary,
            random_initial, random_rhs);

        if (ref.size() != trial.size())
            return false;

        for (std::size_t i = 0; i < ref.size(); ++i)
        {
            if (std::abs(ref[i] - trial[i]) > tolerance)
                return false;
        }
    }

    return true;
}

void destroy(Context* ctx)
{
    delete ctx;
}
