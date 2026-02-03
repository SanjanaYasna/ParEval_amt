/*
Driver for 1D wave eq
#include <hpx/hpx_main.hpp>
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

Stimulaate a 1-D wave equation given the domain is partitioned into tiles arranged on a periodic ring.
The update u_i^{n+1} = 2u_i^n -u_i^{n-1}+ lambda(u_{i-1}^n -2u_i^n+ u_{i+1}^n), where lambda = (c * t / dx)
Use HPX to compute in parallel.  Assume HPX has already been initialized.

Example: 
    input: np=3, nx=4, nt=3, c=1.0, dt =0.1, dx = 0.25, 
    intital condition is 
    for (std::size_t i = 0; i < total; ++i)
    {
        double x = static_cast<double>(i) * dx;
        initial[i] = std::sin(2.0 * M_PI * x);
    }
    That means lambda = (c * t / dx) = 0.4^2 =0.16
    Output: [0.000000, -0.440768, -0.000000, 0.440768, 0.000000, -0.440768, -0.000000, 0.440768, 0.000000, -0.440768, -0.000000, 0.440768]


std::vector<double> wave_parallel(
    std::vector<double> const& initial,
    std::size_t np, std::size_t nx, std::size_t nt,
    double c, double dt, double dx)
{
*/

#pragma once

#include <hpx/config/version.hpp>
#if HPX_VERSION_MAJOR > 1 || (HPX_VERSION_MAJOR == 1 && HPX_VERSION_MINOR >= 10)
#  include "1_10_hpx.hpp"
#else
#  include "hpx-includes.hpp"
#endif
#include "eq.hpp"
#include "generated-code.hpp"
#include "baseline.hpp"

// -----------------------------------------------------------------------------
// Driver context and lifecycle
// -----------------------------------------------------------------------------
struct Context
{
    std::size_t np            = 8;
    std::size_t nx            = 32;
    std::size_t nt            = 3;
    double c                  = 1.0;
    double dt                 = 0.01;
    double dx                 = 0.05;

    std::vector<double> initial;
    std::vector<double> result;
    std::vector<double> reference;
};

 std::size_t total_points(Context const* ctx)
{
    return ctx->np * ctx->nx;
}

 void reset(Context* ctx)
{
    std::size_t const total = total_points(ctx);
    ctx->initial.resize(total);

    // simple standing-wave initial condition
    for (std::size_t i = 0; i < total; ++i)
    {
        double x = static_cast<double>(i) * ctx->dx;
        ctx->initial[i] = std::sin(2.0 * M_PI * x);
    }

    ctx->result.assign(total, 0.0);
    ctx->reference.assign(total, 0.0);
}

 void run_example()
{
    Context ctx;
    ctx.np = 3;
    ctx.nx = 4;
    ctx.nt = 3;
    ctx.c  = 1.0;
    ctx.dt = 0.1;
    ctx.dx = 0.25;

    reset(&ctx);

    auto parallel = wave_parallel(ctx.initial, ctx.np, ctx.nx, ctx.nt,
                                  ctx.c, ctx.dt, ctx.dx);

    auto serial = wave_serial(ctx.initial, ctx.np, ctx.nx, ctx.nt,
                              ctx.c, ctx.dt, ctx.dx);

    auto print = [](char const* label, std::vector<double> const& v)
    {
        std::cout << label << " = [";
        for (std::size_t i = 0; i < v.size(); ++i)
        {
            if (i != 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(6) << v[i];
        }
        std::cout << "]\n";
    };

    std::cout << "Wave-equation example (np=" << ctx.np
              << ", nx=" << ctx.nx << ", nt=" << ctx.nt << ")\n";
    print("parallel", parallel);
    print("serial  ", serial);
    std::cout << '\n';
}

 Context* init()
{
    Context* ctx = new Context;
    reset(ctx);
   // run_example();
    return ctx;
}

 void compute(Context* ctx)
{
    ctx->result = wave_parallel(
        ctx->initial, ctx->np, ctx->nx, ctx->nt, ctx->c, ctx->dt, ctx->dx);
}

 void best(Context* ctx)
{
    ctx->reference = wave_serial(
        ctx->initial, ctx->np, ctx->nx, ctx->nt, ctx->c, ctx->dt, ctx->dx);
}

 bool validate(Context* ctx)
{
    constexpr std::size_t attempts = 3;
    constexpr double tolerance = 1e-9;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, +1.0);

    std::size_t const total = total_points(ctx);
    std::vector<double> random_init(total);

    for (std::size_t a = 0; a < attempts; ++a)
    {
        for (auto& value : random_init)
            value = dist(rng);

        auto serial = wave_serial(
            random_init, ctx->np, ctx->nx, ctx->nt, ctx->c, ctx->dt, ctx->dx);

        auto parallel = wave_parallel(
            random_init, ctx->np, ctx->nx, ctx->nt, ctx->c, ctx->dt, ctx->dx);

        if (serial.size() != parallel.size())
            return false;

        for (std::size_t i = 0; i < serial.size(); ++i)
        {
            if (std::abs(serial[i] - parallel[i]) > tolerance)
                return false;
        }
    }
    return true;
}

 void destroy(Context* ctx)
{
    delete ctx;
}