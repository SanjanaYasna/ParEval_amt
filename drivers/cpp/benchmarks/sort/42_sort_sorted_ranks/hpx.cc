// Driver for 42_sort_sorted_ranks for HPX
// /* For each value in the vector x compute its index in the sorted vector.
//    Store the results in `ranks`.
//    Examples:
// 
//    input: [3.1, 2.8, 9.1, 0.4, 3.14]
//    output: [2, 1, 4, 0, 3]
//  
//    input: [100, 7.6, 16.1, 18, 7.6]
//    output: [4, 0, 1, 2, 3]
// */
// void ranks(std::vector<float> const& x, std::vector<size_t> &ranks) {
#include <hpx/config/version.hpp>
#if HPX_VERSION_MAJOR > 1 || (HPX_VERSION_MAJOR == 1 && HPX_VERSION_MINOR >= 10)
#  include "1_10_hpx.hpp"
#else
#  include "hpx-includes.hpp"
#endif
#include "utilities.hpp"
#include "baseline.hpp" 
#include "generated-code.hpp"

struct Context {
    std::vector<float> x;
    std::vector<size_t> ranks;
};

void reset(Context *ctx) {
    fillRand(ctx->x, -100.0, 100.0);
    BCAST(ctx->x, FLOAT);
}

Context *init() {
    Context *ctx = new Context();

    ctx->x.resize(DRIVER_PROBLEM_SIZE);
    ctx->ranks.resize(DRIVER_PROBLEM_SIZE);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    ranks(ctx->x, ctx->ranks);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctRanks(ctx->x, ctx->ranks);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<float> x(TEST_SIZE);
    std::vector<size_t> correct(TEST_SIZE), test(TEST_SIZE);

    int rank;
    GET_RANK(rank);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(x, -100.0, 100.0);
        BCAST(x, FLOAT);

        // compute correct result
        correctRanks(x, correct);

        // compute test result
        ranks(x, test);
        SYNC();
        
        if (IS_ROOT(rank) && !std::equal(correct.begin(), correct.end(), test.begin())) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
