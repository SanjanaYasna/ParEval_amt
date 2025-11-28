// Driver for 02_dense_la_gemm for HPX
// /* Multiply the matrix A by the matrix B. Store the results in the matrix C.
//    A is an MxK matrix, B is a KxN matrix, and C is a MxN matrix. The matrices are stored in row-major.
//    Use HPX to compute in parallel. Assume main() and hpx_main() functions have already been initialized and simply complete the function below.
//    Example:
// 
//    input: A=[[1, -1, 2], [0, -2, 1]] B=[[4, 1], [-1, 0], [2, 2]]
//    output: C=[[9, 5], [4, 2]]
// */
// void gemm(std::vector<double> const& A, std::vector<double> const& B, std::vector<double> &C, size_t M, size_t K, size_t N) {

#include <iostream>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_generate.hpp>
#include <hpx/include/parallel_transform_reduce.hpp>

#include <vector>
#include <chrono>



using namespace std;

void gemm(std::vector<double> const& A, std::vector<double> const& B, std::vector<double> &C, size_t M, size_t K, size_t N) {
    hpx::for_loop(hpx::execution::par, 0, M, [&](int i){
        for (int k = 0; k < K; k++) {
            int sum = 0;
            for (int j = 0; j < N; j++) {
                C[i*N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    });
}
void gemm(std::vector<double> const& A, std::vector<double> const& B, std::vector<double> &C, size_t M, size_t K, size_t N) {
    // Your code here
}

int main() {
    // Initialize matrices A, B, and C
    std::vector<double> A = {1, -1, 2, 0, -2, 1};
    std::vector<double> B = {4, 1, -1, 0, 2, 2};
    std::vector<double> C(A.size());

    // Call gemm function
    gemm(A, B, C, 2, 3, 2);
}


int hpx_main(int argc, char** argv) {
    vector<double> A = {1, -1, 2, 0, -2, 1};
    vector<double> B = {4, 1, -1, 0, 2, 2};
    vector<double> test(4);
    std::fill(test.begin(), test.end(), 0.0);
    // for (auto i : A)
    //     cout << i << " ";
    // for (auto i : B)
    //     cout << i << " ";
    gemm(A, B, test, 2, 3, 2);
    for (auto i : test)
        cout << i << " ";
    /*
    int matA[3][2] = {{4,2}, {7,1}, {2,3}}; // No of Rows = 3
    int matB[2][3] = {{4,6,7}, {7,1,2}};    // No of Columns = 3

    hpx::cout<<"Matrix A:"<<std::endl;  // Printing value of Matrix A
    for( int i = 0 ; i< 3; i++) {
        for (int j = 0;  j<2; j++) {
            hpx::cout<<matA[i][j]<<"\t";  
        }
        hpx::cout<<std::endl;
    }


    hpx::cout<<"Matrix B:"<<std::endl;  // Printing value of Matrix B
    for( int i = 0 ; i< 2; i++) {
        for (int j = 0;  j<3; j++) {
            hpx::cout<<matB[i][j]<<"\t";
        }
        hpx::cout<<std::endl;
    }

    hpx::cout<<"Matrix Multiplication:"<<std::endl;

    hpx::for_loop(hpx::execution::par, 0, 3, [&](int i) {
        for (int j = 0;  j<3; j++) {  // Iteration upto no of columns in Matrix B
            int sum = 0; 
            for (int k = 0;  k< 2;  k++) {  // Iteration upto no of columns in Matrix A
                sum += matA[i][k] * matB[k][j];
            }
            hpx::cout << sum << "\t"; 
        }
        hpx::cout << std::endl;  
    });
    
    */
    return hpx::finalize();
}


int main(int argc, char* argv[]) {
    // Initialize HPX runtime 

    return hpx::init(argc, argv);
}
