// reference: https://ajdillhoff.github.io/notes/gpu_pattern_parallel_histogram/

// Driver for 61_parallel_histogram for HPX
// #include <hpx/hpx_main.hpp>
//    Build a histogram with `bins` buckets from the sample values in `data`.
//    Use HPX to compute in parallel.  Assume HPX has already been initialized.
//    Example:
//
//    input:
//       data = [0.12, 0.76, 0.54, 0.03, 0.33, 0.90, 0.61, 0.28]
//       bins = 4
//
//    output:
//       histogram = [2, 3, 1, 2]
//
// std::vector<std::uint64_t> parallel_histogram(
//     std::vector<double> const& data,
//     std::size_t bins){

#include <hpx/config/version.hpp>
#if HPX_VERSION_MAJOR > 1 || (HPX_VERSION_MAJOR == 1 && HPX_VERSION_MINOR >= 10)
#  include "1_10_hpx.hpp"
#else
#  include "hpx-includes.hpp"
#endif
#include "utilities_old.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"

struct Context
{
    std::vector<double> data; //assigned to bins
    std::vector<std::uint64_t> histogram; //histogram itself, to be checked
    std::size_t bins;   // number of histogram buckets
};

void reset(Context* ctx)
{
    fillRand(ctx->data, 0.0, 1.0);
    BCAST(ctx->data, DOUBLE);
    ctx->histogram.assign(ctx->bins, 0);
}

Context* init()
{
    Context* ctx = new Context();
//4k is ballpark 
    ctx->bins = 64;
    ctx->data.resize(DRIVER_PROBLEM_SIZE);
    ctx->histogram.assign(ctx->bins, 0);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context* ctx)
{
    ctx->histogram = parallel_histogram(ctx->data, ctx->bins);
}

void NO_OPTIMIZE best(Context* ctx)
{
    ctx->histogram = correct_parallel_histogram(ctx->data, ctx->bins);
}

bool validate(Context* ctx)
{
    const std::size_t bins = ctx->bins;
    constexpr std::size_t TEST_SAMPLES = 1 << 14;

    std::vector<double> samples(TEST_SAMPLES);
    std::vector<std::uint64_t> reference;
    std::vector<std::uint64_t> trial;

    int rank;
    GET_RANK(rank);

    const std::size_t trials = MAX_VALIDATION_ATTEMPTS;
    for (std::size_t attempt = 0; attempt < trials; ++attempt)
    {
        fillRand(samples, 0.0, 1.0);
        BCAST(samples, DOUBLE);

        reference = correct_parallel_histogram(samples, bins);
        trial     = parallel_histogram(samples, bins);
        SYNC();

        bool isCorrect = true;
        if (IS_ROOT(rank) &&
            !std::equal(reference.begin(), reference.end(), trial.begin()))
        {
            isCorrect = false;
        }
        BCAST_PTR(&isCorrect, 1, CXX_BOOL);
        if (!isCorrect)
        {
            return false;
        }
    }

    return true;
}

void destroy(Context* ctx)
{
    delete ctx;
}