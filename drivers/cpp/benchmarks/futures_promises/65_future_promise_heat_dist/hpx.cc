// Driver for the 1D heat-distribution benchmark (HPX version)
// #include <hpx/hpx_main.hpp>
// inline std::size_t idx(std::size_t i, int dir, std::size_t size)
// {
//     if (i == 0 && dir == -1)
//         return size - 1;
//     if (i == size - 1 && dir == +1)
//         return 0;

//     HPX_ASSERT((i + dir) < size);
//     return i + dir;
// }
// struct partition
// {
//     using tile_type   = std::vector<double>;
//     using future_type = hpx::shared_future<tile_type>;

//     partition() = default;
//     explicit partition(future_type f)
//       : data_(std::move(f))
//     {}

//     partition& operator=(future_type f)
//     {
//         data_ = std::move(f);
//         return *this;
//     }

//     future_type const& get_future() const
//     {
//         return data_;
//     }
// private:
//     future_type data_;
// };
// inline auto make_default_step_factory(double k, double dt, double dx)
// {
//     using future_type = partition::future_type;

//     return [coeff = k * dt / (dx * dx)](future_type const& left,
//                                         future_type const& middle,
//                                         future_type const& right) -> future_type
//     {
//         using hpx::dataflow;
//         using hpx::launch;
//         using hpx::util::unwrapping;

//         auto kernel = unwrapping([coeff](partition::tile_type const& l,
//                                          partition::tile_type const& m,
//                                          partition::tile_type const& r)
//                                         -> partition::tile_type
//         {
//             std::size_t const N = m.size();
//             HPX_ASSERT(N >= 2);

//             partition::tile_type next(N);

//             for (std::size_t j = 1; j < N - 1; ++j)
//                 next[j] = m[j] + coeff * (m[j - 1] - 2.0 * m[j] + m[j + 1]);

//             next[0] = m[0] + coeff * (l.back() - 2.0 * m[0] + m[1]);
//             next[N - 1] = m[N - 1] +
//                 coeff * (m[N - 2] - 2.0 * m[N - 1] + r.front());

//             return next;
//         });

//         return dataflow(launch::async, kernel, left, middle, right).share();
//     };
// }
// The following inputs are defined as:
// - np: number of partitions (tiles), arranged in a ring (periodic neighbors)
// - nx: number of grid points per partition (nx >= 2)
// - nt: number of time steps to advance
// - k: thermal diffusivity coefficient
// - dt: time step size
// - dx: spatial grid spacing

// Compute the 1D heat distribution equation, using the provided make_default_step_factory(k, dt, dx) to find the per-partition results after nt steps. 
// The make_default_step_factory splits the computations into three sets of tiles and returns a per-tile update function for the 1D heat distribution equation.
// Use HPX to compute in parallel.  Assume HPX has already been initialized.

// Example
//
//   input:
//     initial = [0, 1, 2, 3, 4, 5]
//     np = 2, nx = 3, nt = 1
//     k = 0.5, dt = 1.0, dx = 1.0
//
//   Explanation: layer 0 has tiles
//       tile0 = [0,1,2], tile1 = [3,4,5]
//     After one step (periodic boundaries):
//       tile0'[0] = 0 + coeff*(2 - 0 + 1) = 3.0
//       tile0'[1] = 1 + coeff*(0 - 2 + 2) = 1.0
//       tile0'[2] = 2 + coeff*(1 - 4 + 3) = 2.0
//       tile1'[0] = 3 + coeff*(1 - 6 + 4) = 3.0
//       tile1'[1] = 4 + coeff*(3 - 8 + 5) = 4.0
//       tile1'[2] = 5 + coeff*(4 -10 + 2) = 2.0
//
//   output:
//     parallel_heat(...) should return [3.0, 1.0, 2.0, 3.0, 4.0, 2.0]

// auto make_step = make_default_step_factory();
// template <typename DataflowFactory>
// std::vector<double> parallel_heat(
//     std::vector<double> const& initial,
//     std::size_t np, std::size_t nx, std::size_t nt,
//     double k_in, double dt_in, double dx_in,
//     DataflowFactory make_step)
// {


#include <hpx/config/version.hpp>
#if HPX_VERSION_MAJOR > 1 || (HPX_VERSION_MAJOR == 1 && HPX_VERSION_MINOR >= 10)
#  include "1_10_hpx.hpp"
#else
#  include "hpx-includes.hpp"
#endif
#include "partition.hpp"
#include "utilities_old.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"    // <--- include the simplified solver above
#pragma once

#include <hpx/include/dataflow.hpp>
#include <hpx/include/lcos.hpp>

struct Context
{
    std::size_t np;
    std::size_t nx;
    std::size_t nt;

    double k;
    double dt;
    double dx;

    std::vector<double> initial;
    std::vector<double> result;
    std::vector<double> reference;
};

static std::size_t total_points(Context const* ctx)
{
    return ctx->np * ctx->nx;
}

inline auto make_default_step_factory(double k, double dt, double dx)
{
    using future_type = partition::future_type;

    return [coeff = k * dt / (dx * dx)](future_type const& left,
                                        future_type const& middle,
                                        future_type const& right) -> future_type
    {
        using hpx::dataflow;
        using hpx::launch;
        using hpx::util::unwrapping;

        auto kernel = unwrapping([coeff](partition::tile_type const& l,
                                         partition::tile_type const& m,
                                         partition::tile_type const& r)
                                        -> partition::tile_type
        {
            std::size_t const N = m.size();
            HPX_ASSERT(N >= 2);

            partition::tile_type next(N);

            for (std::size_t j = 1; j < N - 1; ++j)
                next[j] = m[j] + coeff * (m[j - 1] - 2.0 * m[j] + m[j + 1]);

            next[0] = m[0] + coeff * (l.back() - 2.0 * m[0] + m[1]);
            next[N - 1] = m[N - 1] +
                coeff * (m[N - 2] - 2.0 * m[N - 1] + r.front());

            return next;
        });

        return dataflow(launch::async, kernel, left, middle, right).share();
    };
}


void run_example()
{
    // Example from the prompt: two partitions, three points each, one time step
    std::vector<double> initial = {0, 1, 2, 3, 4, 5};
    std::size_t np = 2, nx = 3, nt = 1;
    double k_val = 0.5, dt_val = 1.0, dx_val = 1.0;

    auto factory = make_default_step_factory(k_val, dt_val, dx_val);
    std::vector<double> async_res = parallel_heat(
        initial, np, nx, nt, k_val, dt_val, dx_val, factory);
    std::vector<double> serial_res = correct_parallel_heat(
        initial, np, nx, nt, k_val, dt_val, dx_val);

    int rank = 0;
    GET_RANK(rank);
    if (IS_ROOT(rank))
    {
        auto print_vec = [](char const* label, std::vector<double> const& v) {
            std::cout << label << " = [";
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (i != 0)
                    std::cout << ", ";
                std::cout << v[i];
            }
            std::cout << "]\n";
        };

        std::cout << "Example (np=2, nx=3, nt=1)\n";
        print_vec("parallel_heat", async_res);
        print_vec("correct_parallel_heat", serial_res);
        std::cout << std::endl;
    }
    SYNC();
}

void reset(Context* ctx)
{
    const std::size_t points = total_points(ctx);
    ctx->initial.resize(points);

    for (std::size_t p = 0; p < ctx->np; ++p)
    {
        const double base = static_cast<double>(p * ctx->nx);
        for (std::size_t i = 0; i < ctx->nx; ++i)
            ctx->initial[p * ctx->nx + i] = base + static_cast<double>(i);
    }

    BCAST(ctx->initial, DOUBLE);
    ctx->result.assign(points, 0.0);
    ctx->reference.assign(points, 0.0);
}

Context* init()
{
    Context* ctx = new Context;

    ctx->np = 32;
    ctx->nx = std::max<std::size_t>(1, DRIVER_PROBLEM_SIZE / ctx->np);
    ctx->nt = 64;

    ctx->k  = k;
    ctx->dt = dt;
    ctx->dx = dx;

    reset(ctx);
    //run_example();
    return ctx;
}

void NO_OPTIMIZE compute(Context* ctx)
{
    auto factory = make_default_step_factory(ctx->k, ctx->dt, ctx->dx);
    ctx->result = parallel_heat(
        ctx->initial, ctx->np, ctx->nx, ctx->nt,
        ctx->k, ctx->dt, ctx->dx,
        factory);
}

void NO_OPTIMIZE best(Context* ctx)
{
    ctx->reference = correct_parallel_heat(
        ctx->initial, ctx->np, ctx->nx, ctx->nt,
        ctx->k, ctx->dt, ctx->dx);
}

bool validate(Context* ctx)
{
    int rank = 0;
    GET_RANK(rank);

    const std::size_t points = total_points(ctx);
    std::vector<double> initial(points);
    std::vector<double> reference;
    std::vector<double> trial;

    constexpr std::size_t trials = MAX_VALIDATION_ATTEMPTS;
    constexpr double tol = 1e-6;

    for (std::size_t attempt = 0; attempt < trials; ++attempt)
    {
        for (std::size_t i = 0; i < points; ++i)
            initial[i] = static_cast<double>(attempt * points + i);

        BCAST(initial, DOUBLE);

        reference = correct_parallel_heat(
            initial, ctx->np, ctx->nx, ctx->nt,
            ctx->k, ctx->dt, ctx->dx);

        auto factory = make_default_step_factory(ctx->k, ctx->dt, ctx->dx);
        trial = parallel_heat(
            initial, ctx->np, ctx->nx, ctx->nt,
            ctx->k, ctx->dt, ctx->dx,
            factory);

        SYNC();

        bool ok = true;
        if (IS_ROOT(rank))
        {
            if (reference.size() != trial.size())
            {
                ok = false;
            }
            else
            {
                for (std::size_t i = 0; i < trial.size(); ++i)
                {
                    if (std::abs(reference[i] - trial[i]) > tol)
                    {
                        ok = false;
                        break;
                    }
                }
            }
        }

        BCAST_PTR(&ok, 1, CXX_BOOL);
        if (!ok)
            return false;
    }

    return true;
}

void destroy(Context* ctx)
{
    delete ctx;
}