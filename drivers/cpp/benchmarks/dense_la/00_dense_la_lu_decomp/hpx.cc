// Driver for 00_dense_la_lu_decomp for Kokkos
// #include <hpx/hpx_main.hpp>
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

#include "kokkos-includes.hpp"

#include "utilities.hpp"
#include "baseline.hpp" //contains correctLuFactorize 
#include "generated-code.hpp"


