// Driver for 44_sort_non-zero for HPX
// /* Sort the vector x in ascending order ignoring elements with value 0.
//    Leave zero valued elements in-place.
//    Example:
// 
// 	  input: [8, 4, 0, 9, 8, 0, 1, -1, 7]
//    output: [-1, 1, 0, 4, 7, 0, 8, 8, 9]
// */
// void sortIgnoreZero(std::vector<int> &x) {
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
    std::vector<int> x;
};

void fillRandWithZeroes(std::vector<int> &x) {
    // fill x with random values, but set some to zero
    for (int i = 0; i < x.size(); i += 1) {
        x[i] = rand();
        if (rand() % 5) {
            x[i] = 0;
        }
    }
}

void reset(Context *ctx) {
    fillRandWithZeroes(ctx->x);
    BCAST(ctx->x, INT);
}

Context *init() {
    Context *ctx = new Context();
    ctx->x.resize(DRIVER_PROBLEM_SIZE);
    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    sortIgnoreZero(ctx->x);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctSortIgnoreZero(ctx->x);
}

bool validate(Context *ctx) {

    int rank;
    GET_RANK(rank);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int i = 0; i < numTries; i += 1) {
        std::vector<int> input(1024);
        fillRandWithZeroes(input);
        BCAST(input, INT);

        // compute correct result
        std::vector<int> correctResult = input;
        correctSortIgnoreZero(correctResult);

        // compute test result
        std::vector<int> testResult = input;
        sortIgnoreZero(testResult);
        SYNC();
        
        bool isCorrect = true;
        if (IS_ROOT(rank) && !std::equal(correctResult.begin(), correctResult.end(), testResult.begin())) {
            isCorrect = false;
        }
        BCAST_PTR(&isCorrect, 1, CXX_BOOL);
        if (!isCorrect) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}


