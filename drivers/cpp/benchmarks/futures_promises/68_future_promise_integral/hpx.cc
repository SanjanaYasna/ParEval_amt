
/*
#include <hpx/hpx_main.hpp>
struct integral_result
{
    double        accumulator = 0.0;
    std::uint64_t samples     = 0;

    double estimate(double a, double b) const
    {
        return (samples == 0)
            ? 0.0
            : (b - a) * (accumulator / static_cast<double>(samples));
    }
};

std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= (x >> 31);
    return x;
}

double sample_unit_interval(std::uint64_t index, std::uint64_t seed)
{
    std::uint64_t value = splitmix64(index + seed);
    return static_cast<double>(value >> 11) *
        (1.0 / static_cast<double>(1ull << 53));
}

Approximate the definite integral of a user-supplied function f over the interval [a, b] using Monte Carlo sampling: draw total_samples uniformly distributed points in [a, b], evaluate f = e^{-x^2} at each point, 
and accumulate the results to estimate the integral. Split the total number of samples into chunk_size blocks. 
Return an integral_result that holds the sample count and accumulated function values for the entire run.
Use HPX to compute in parallel.  Assume HPX has already been initialized.

Example: 
    input: total_samples: 5,000,000, chunk_size: 100,000, a: 0.0, b: 1.0, seed: 2024
    output: Integral estimate returned: 0.74673

template <typename F>
integral_result monte_carlo_integral_parallel(
    std::uint64_t total_samples,
    std::uint64_t chunk_size,
    double a, 
    double b,
    std::uint64_t seed,
    F&& f)
{

*/


// -----------------------------------------------------------------------------
// Monte Carlo integral utilities
// -----------------------------------------------------------------------------
#pragma once
#include <hpx/config/version.hpp>
#if HPX_VERSION_MAJOR > 1 || (HPX_VERSION_MAJOR == 1 && HPX_VERSION_MINOR >= 10)
// adjust include if you have a wrapper for HPX 1.10
#  include "1_10_hpx.hpp"
#else
#  include "hpx-includes.hpp"
#endif
#include "integral.hpp"
#include "generated-code.hpp"
#include "baseline.hpp"

// -----------------------------------------------------------------------------
// Driver context and lifecycle helpers
// -----------------------------------------------------------------------------
struct Context
{
    std::uint64_t total_samples = 20'000'000;
    std::uint64_t chunk_size    = 250'000;
    std::uint64_t seed          = 42;

    double a = 0.0;
    double b = M_PI;

    std::function<double(double)> integrand =
        [](double x) { return std::sin(x); };

    integral_result result{};
    integral_result reference{};
};

void reset(Context* ctx)
{
    ctx->result    = integral_result{};
    ctx->reference = integral_result{};
}

// -----------------------------------------------------------------------------
// Example demonstration
// -----------------------------------------------------------------------------
void run_example()
{
    Context ctx;
    ctx.total_samples = 5'000'000;
    ctx.chunk_size    = 100'000;
    ctx.seed          = 2024;
    ctx.a             = 0.0;
    ctx.b             = M_PI;
    ctx.integrand     = [](double x) { return std::sin(x); };

    ctx.result = monte_carlo_integral_parallel(
        ctx.total_samples,
        ctx.chunk_size,
        ctx.a,
        ctx.b,
        ctx.seed,
        ctx.integrand);

    ctx.reference = monte_carlo_integral_serial(
        ctx.total_samples,
        ctx.a,
        ctx.b,
        ctx.seed,
        ctx.integrand);

    std::cout << "Monte Carlo integral example (" << ctx.total_samples
              << " samples)\n";

    auto print_result = [&](char const* label, integral_result const& r)
    {
        std::cout << "  " << std::setw(10) << std::left << label
                  << "samples=" << r.samples
                  << ", estimate ≈ "
                  << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << r.estimate(ctx.a, ctx.b) << '\n';
    };

    print_result("parallel", ctx.result);
    print_result("serial",   ctx.reference);

    std::cout << "  expected  = 2.0\n";
    std::cout << "  difference= "
              << std::setprecision(6)
              << std::abs(ctx.result.estimate(ctx.a, ctx.b) -
                          ctx.reference.estimate(ctx.a, ctx.b))
              << "\n\n";
}

Context* init()
{
    Context* ctx = new Context;
    reset(ctx);
    return ctx;
}

void compute(Context* ctx)
{
    ctx->result = monte_carlo_integral_parallel(
        ctx->total_samples,
        ctx->chunk_size,
        ctx->a,
        ctx->b,
        ctx->seed,
        ctx->integrand);
}

void best(Context* ctx)
{
    ctx->reference = monte_carlo_integral_serial(
        ctx->total_samples,
        ctx->a,
        ctx->b,
        ctx->seed,
        ctx->integrand);
}

bool validate(Context* ctx)
{
    constexpr std::size_t attempts  = 4;
    constexpr double      tolerance = 1e-10;

    for (std::size_t trial = 0; trial < attempts; ++trial)
    {
        Context probe = *ctx;
        probe.seed += trial * 1021;
        probe.total_samples = std::max<std::uint64_t>(1, ctx->total_samples / (trial + 1));

        integral_result serial = monte_carlo_integral_serial(
            probe.total_samples, probe.a, probe.b, probe.seed, probe.integrand);

        integral_result parallel = monte_carlo_integral_parallel(
            probe.total_samples, ctx->chunk_size, probe.a, probe.b, probe.seed, probe.integrand);

        if (serial.samples != parallel.samples)
            return false;

        double diff = std::abs(serial.estimate(probe.a, probe.b) -
                               parallel.estimate(probe.a, probe.b));
        if (diff > tolerance)
            return false;
    }

    return true;
}

void destroy(Context* ctx)
{
    delete ctx;
}