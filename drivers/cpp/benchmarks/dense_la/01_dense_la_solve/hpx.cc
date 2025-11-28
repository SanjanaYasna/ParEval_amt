// Driver for 01_dense_la_solve for HPX
//#include <hpx/hpx_main.hpp>
// 
// /* Solve the linear system Ax=b for x.
//    A is an NxN matrix. x and b have N elements.
//    Use HPX to compute in parallel. Assume main() and hpx_main() functions have already been initialized and simply complete the function below.
//    Example:
//    
//    input: A=[[1,4,2], [1,2,3], [2,1,3]] b=[11, 11, 13]
//    output: x=[3, 1, 2]
// */
// void solveLinearSystem(std::vector<double> const& A, std::vector<double> const& b, std::vector<double> &x, size_t N) {
    //other option:
// void solveLinearSystem(std::vector<hpx::shared_future<std::vector<double>>> &A, std::vector<hpx::shared_future<std::vector<double>>> &b, std::vector<hpx::shared_future<std::vector<double>>> &x, size_t N) {

#include "hpx-includes.hpp"
#include "utilities_old.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "baseline.hpp"
#include "generated-code.hpp"
//TODO: GENERATED CODE HEADER TO PULL FROM


struct Context {
    std::vector<double> A, b, x;
    size_t N;
};

void createRandomLinearSystem(std::vector<double> &A, std::vector<double> &b, std::vector<double> &x, size_t N) {
    fillRand(A, -10.0, 10.0);
    fillRand(x, -10.0, 10.0);

    std::fill(b.begin(), b.end(), 0.0);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j += 1) {
            b[i] += A[i * N + j] * x[j];
        }
    }

    std::fill(x.begin(), x.end(), 0.0);
}

void reset(Context *ctx) {
    createRandomLinearSystem(ctx->A, ctx->b, ctx->x, ctx->N);

    BCAST(ctx->A, DOUBLE);
    BCAST(ctx->b, DOUBLE);
    BCAST(ctx->x, DOUBLE);
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = DRIVER_PROBLEM_SIZE;
    ctx->A.resize(ctx->N * ctx->N);
    ctx->b.resize(ctx->N);
    ctx->x.resize(ctx->N);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    solveLinearSystem(ctx->A, ctx->b, ctx->x, ctx->N);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctSolveLinearSystem(ctx->A, ctx->b, ctx->x, ctx->N);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 512;

    std::vector<double> A(TEST_SIZE * TEST_SIZE);
    std::vector<double> b(TEST_SIZE), correct(TEST_SIZE), test(TEST_SIZE);

    int rank;
    GET_RANK(rank);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        createRandomLinearSystem(A, b, correct, TEST_SIZE);
        std::fill(correct.begin(), correct.end(), 0.0);
        std::fill(test.begin(), test.end(), 0.0);

        BCAST(A, DOUBLE);
        BCAST(b, DOUBLE);

        // compute correct result
        correctSolveLinearSystem(A, b, correct, TEST_SIZE);

        // compute test result
        solveLinearSystem(A, b, test, TEST_SIZE);
        SYNC();
        
        bool isCorrect = true;
        if (IS_ROOT(rank) && (!fequal(correct, test, 1e-4) || std::any_of(test.begin(), test.end(), [](double x) { return std::isnan(x); }))) {
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
