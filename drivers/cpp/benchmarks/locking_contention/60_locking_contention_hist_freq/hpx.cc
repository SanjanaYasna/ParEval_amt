// Driver for 60_parallel_hist_freq for HPX
// #include <hpx/hpx_main.hpp>
// using histogram_t = std::array<std::uint64_t, 256>;
// struct SharedHistogram
// {
//     histogram_t bins{};
//     std::array<std::mutex, 256> locks;

//     SharedHistogram()
//     {
//         bins.fill(0);
//     }

//     void reset()
//     {
//         for (std::size_t i = 0; i < bins.size(); ++i)
//         {
//             std::lock_guard<std::mutex> g(locks[i]);
//             bins[i] = 0;
//         }
//     }

//     void increment(unsigned char value)
//     {
//         std::lock_guard<std::mutex> g(locks[value]);
//         ++bins[value];
//     }

//     histogram_t snapshot()
//     {
//         histogram_t copy;
//         for (std::size_t i = 0; i < bins.size(); ++i)
//         {
//             std::lock_guard<std::mutex> g(locks[i]);
//             copy[i] = bins[i];
//         }
//         return copy;
//     }
// };
//    Build a 256-bin byte-frequency histogram from `data` (values in [0,255]) and return the resulting SharedHistogram object.
//    Use HPX to compute in parallel.  Assume HPX has already been initialized.
//    Example:
//       data = [0x20, 0x41, 0x20, 0x0A]
//       bins  = 256
//    output histogram[0x20] == 2, histogram[0x41] == 1, histogram[0x0A] == 1
// 
// histogram_t parallel_histogram(std::vector<unsigned char> const& data, SharedHistogram& shared){

#include <hpx/config/version.hpp>
#if HPX_VERSION_MAJOR > 1 || (HPX_VERSION_MAJOR == 1 && HPX_VERSION_MINOR >= 10)
#  include "1_10_hpx.hpp"
#else
#  include "hpx-includes.hpp"
#endif
#include "utilities_old.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"
#include "histogram.hpp"


struct Context
{
    std::vector<unsigned char> data;
    histogram_t histogram;
    SharedHistogram shared;
};

static void fill_random_bytes(std::vector<unsigned char>& dst, std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& byte : dst)
    {
        byte = static_cast<unsigned char>(dist(rng));
    }
}

void reset(Context* ctx)
{
    static std::uint64_t seed = 1234567;
    fill_random_bytes(ctx->data, seed++);
    ctx->histogram.fill(0);
}

Context* init()
{
    Context* ctx = new Context();
    ctx->data.resize(DRIVER_PROBLEM_SIZE);
    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context* ctx)
{
    ctx->histogram = parallel_histogram(ctx->data,  ctx->shared);
}

void NO_OPTIMIZE best(Context* ctx)
{
    ctx->histogram = correct_parallel_histogram(ctx->data, ctx->shared);
}

bool validate(Context* ctx)
{
    constexpr std::size_t TEST_SIZE = 1 << 14;
    std::vector<unsigned char> samples(TEST_SIZE);

    histogram_t reference{};
    histogram_t test{};

    SharedHistogram reference_shared;
    SharedHistogram test_shared; 

    int rank;
    GET_RANK(rank);

    for (std::size_t attempt = 0; attempt < MAX_VALIDATION_ATTEMPTS; ++attempt)
    {
        fill_random_bytes(samples, 42 + attempt);
        reference_shared.reset();
        test_shared.reset();

        reference = correct_parallel_histogram(samples, reference_shared);
        test      = parallel_histogram(samples, test_shared);
        SYNC();

        bool ok = true;
        if (IS_ROOT(rank) && reference != test)
        {
            ok = false;
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