// Driver for 00_dense_la_lu_decomp for Kokkos
//#include <hpx/hpx_main.hpp>
// 
// /* Factorize the matrix A into A=LU where L is a lower triangular matrix and U is an upper triangular matrix.
//    Store the results for L and U into the original matrix A. 
//    A is an NxN matrix.
//    Use HPX to compute in parallel. Assume HPX has already been initialized.
//    Example:
// 
//    input: [[4, 3], [6, 3]]
//    output: [[4, 3], [1.5, -1.5]]
// */
// void luFactorize(std::vector<double> &A, std::size_t N) {

#include <algorithm> 
#include <numeric>
#include <random>
#include <vector>


#include "hpx-includes.hpp"

#include "utilities.hpp"
#include "baseline.hpp" //contains correctLuFactorize 
#include "generated-code.hpp"

struct Context {
    std::vector<double> A;
    size_t N;
};

void reset(Context *ctx) {
    fillRand(ctx->A, -10.0, 10.0);
    BCAST(ctx->A, DOUBLE);
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = DRIVER_PROBLEM_SIZE;
    ctx->A.resize(ctx->N * ctx->N);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    luFactorize(ctx->A, ctx->N);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctLuFactorize(ctx->A, ctx->N);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 512;
    //vec of test case inputs for baseline
    std::vector<double> A_correct(TEST_SIZE * TEST_SIZE), A_test(TEST_SIZE * TEST_SIZE);
    //vec of test case inputs for generated code
    std::vector<double> A(TEST_SIZE * TEST_SIZE);
 
    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < num_tries; trialIter++){
        fillRand(A_host, -10.0, 10.0);
        A_correct = A;
        correctLuFactorize(A_correct, TEST_SIZE)
        A_test = A;
        luFactorize(A_test, TEST_SIZE);
        if (!fequal(A_host, A_test, 1e-3)) {
            return false;
        }
    }
    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
