// simple 1D data tile for an asynchronous Jacobi sweep
/*
#include <hpx/hpx_main.hpp>
struct monte_carlo_result
{
    std::uint64_t hits    = 0;
    std::uint64_t samples = 0;

    double pi_estimate() const
    {
        return (samples == 0)
            ? 0.0
            : 4.0 * static_cast<double>(hits) / static_cast<double>(samples);
    }
};

// stateless SplitMix64 hash to make RNG reproducible across runs
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
    return static_cast<double>(value >> 11) * (1.0 / static_cast<double>(1ull << 53));
}

Approximate the value of pi by counting how many randomly generated points on the unit square lie in the unit circle. Split the total number of points (num_samples) into chunk_size blocks.
Use HPX to compute in parallel.  Assume HPX has already been initialized.

monte_carlo_result monte_carlo_parallel(
    std::uint64_t total_samples,
    std::uint64_t chunk_size,
    std::uint64_t seed)
{
*/


// -----------------------------------------------------------------------------
// Monte Carlo utilities
// -----------------------------------------------------------------------------

#pragma once

#include <hpx/config/version.hpp>
#if HPX_VERSION_MAJOR > 1 || (HPX_VERSION_MAJOR == 1 && HPX_VERSION_MINOR >= 10)
#  include "1_10_hpx.hpp"
#else
#  include "hpx-includes.hpp"
#endif
#include "montecarlo.hpp"
#include "utilities_old.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"    // <--- include the simplified solver above



// -----------------------------------------------------------------------------
// Driver context and lifecycle helpers
// -----------------------------------------------------------------------------
struct Context
{
    std::uint64_t total_samples = 40'000'000;
    std::uint64_t chunk_size    = 250'000;
    std::uint64_t seed          = 0xBADC0FFEEULL;

    monte_carlo_result result{};
    monte_carlo_result reference{};
};

void reset(Context* ctx)
{
    ctx->result     = monte_carlo_result{};
    ctx->reference  = monte_carlo_result{};
}


// -----------------------------------------------------------------------------
// Example demonstration
// -----------------------------------------------------------------------------
void run_example()
{
    Context ctx;
    ctx.total_samples = 1'000'000;
    ctx.chunk_size    = 50'000;
    ctx.seed          = 123456789ULL;

    ctx.result = monte_carlo_parallel(ctx.total_samples, ctx.chunk_size, ctx.seed);
    ctx.reference = monte_carlo_serial(ctx.total_samples, ctx.seed);

    std::cout << "Monte Carlo π example (" << ctx.total_samples << " samples)\n";

    auto print_result = [](char const* label, monte_carlo_result const& r)
    {
        std::cout << "  " << std::setw(10) << std::left << label
                  << "hits=" << r.hits
                  << ", total=" << r.samples
                  << ", π ≈ " << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << r.pi_estimate() << '\n';
    };

    print_result("parallel", ctx.result);
    print_result("serial", ctx.reference);

    std::cout << "  difference = "
              << std::setprecision(6)
              << std::abs(ctx.result.pi_estimate() - ctx.reference.pi_estimate())
              << "\n\n";
}

Context* init()
{
    Context* ctx = new Context;
   // run_example();
    reset(ctx);
    return ctx;
}

void compute(Context* ctx)
{
    ctx->result = monte_carlo_parallel(
        ctx->total_samples,
        ctx->chunk_size,
        ctx->seed);
}

void best(Context* ctx)
{
    ctx->reference = monte_carlo_serial(
        ctx->total_samples,
        ctx->seed);
}

bool validate(Context* ctx)
{
    constexpr std::size_t attempts = 4;
    constexpr double tolerance = 1e-12;

    for (std::size_t trial = 0; trial < attempts; ++trial)
    {
        std::uint64_t seed = ctx->seed + trial * 7919;
        std::uint64_t samples = ctx->total_samples / (trial + 1);

        monte_carlo_result serial = monte_carlo_serial(samples, seed);
        monte_carlo_result parallel = monte_carlo_parallel(samples, ctx->chunk_size, seed);

        if (serial.samples != parallel.samples ||
            serial.hits != parallel.hits)
        {
            return false;
        }

        double diff = std::abs(serial.pi_estimate() - parallel.pi_estimate());
        if (diff > tolerance)
            return false;
    }

    return true;
}

void destroy(Context* ctx)
{
    delete ctx;
}
