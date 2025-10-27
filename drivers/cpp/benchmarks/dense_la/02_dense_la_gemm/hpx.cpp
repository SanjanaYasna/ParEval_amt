// Driver for 02_dense_la_gemm for HPXP
// /* Multiply the matrix A by the matrix B. Store the results in the matrix C.
//    A is an MxK matrix, B is a KxN matrix, and C is a MxN matrix. The matrices are stored in row-major.
//    Use HPX to compute in parallel. Assume main() and hpx_main() functions have already been initialized and simply complete the function below.
//    Example:
// 
//    input: A=[[1, -1, 2], [0, -2, 1]] B=[[4, 1], [-1, 0], [2, 2]]
//    output: C=[[9, 5], [4, 2]]
// */
// void gemm(std::vector<double> const& A, std::vector<double> const& B, std::vector<double> &C, size_t M, size_t K, size_t N) {

//EXAMPLE: https://github.com/Neeraj8949/Matrix_multiplication/blob/main/matrix_multiplication.cpp
//ANOTHER: https://github.com/sakshamdargan/Matrix_Multiplication_HPX/blob/master/main.cpp 
//this doesn't require futures, can be done with parallel::for_loop, as well as futures...
