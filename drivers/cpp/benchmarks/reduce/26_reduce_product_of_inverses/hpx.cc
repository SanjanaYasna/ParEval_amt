// Driver for 26_reduce_product_of_inverses for HPX
// /* Return the product of the vector x with every odd indexed element inverted.
//    i.e. x_0 * 1/x_1 * x_2 * 1/x_3 * x_4 ...
//    Example:
//
//    input: [4, 2, 10, 4, 5]
//    output: 25
// */
// double productWithInverses(std::vector<double> const& x) {

#include "hpx-includes.hpp"
#include "utilities_old.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"

struct Context {
    std::vector<double> x;
};

void reset(Context *ctx) {
    fillRand(ctx->x, 1.0, 100.0);
    BCAST(ctx->x, DOUBLE);
}

Context *init() {
    Context *ctx = new Context();

    ctx->x.resize(DRIVER_PROBLEM_SIZE);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    double val = productWithInverses(ctx->x);
    (void)val;
}

void NO_OPTIMIZE best(Context *ctx) {
    double val = correctProductWithInverses(ctx->x);
    (void)val;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<double> x(TEST_SIZE);
    double test, correct;

    int rank;
    GET_RANK(rank);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(x, 1.0, 100.0);
        BCAST(x, DOUBLE);

        // compute correct result
        correct = correctProductWithInverses(x);

        // compute test result
        test = productWithInverses(x);
        SYNC();

        bool isCorrect = true;
        if (IS_ROOT(rank) && std::abs(correct - test) > 1e-4) {
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