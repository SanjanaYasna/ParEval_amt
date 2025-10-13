// file: lu_inplace_hpx.cpp
// Build with your HPX toolchain, e.g. (example):
//    g++ -std=c++17 lu_inplace_hpx.cpp -I/path/to/hpx/include -L/path/to/hpx/lib -lhpx -o lu_inplace_hpx
//
// Alternatively, use the HPX CMake integration.

#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_execution.hpp>

#include <vector>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

// access helper for row-major NxN matrix stored in vector<double>
inline double &Aref(std::vector<double> &A, std::size_t N, std::size_t i, std::size_t j) {
    return A[i * N + j];
}
inline double Aget(const std::vector<double> &A, std::size_t N, std::size_t i, std::size_t j) {
    return A[i * N + j];
}

// In-place LU (Doolittle) without pivoting.
// Stores L (unit diagonal) in strict lower triangle, U in upper triangle including diagonal.
// Throws std::runtime_error if a zero pivot is encountered.
void lu_inplace_hpx(std::vector<double> &A, std::size_t N) {
    using hpx::execution::par;
    for (std::size_t k = 0; k < N; ++k) {
        // Compute U[k][j] for j = k..N-1
        // U[k][j] = A[k][j] - sum_{p=0..k-1} L[k][p] * U[p][j]
        hpx::parallel::for_loop(par, k, N, [&](std::size_t j) {
            double sum = 0.0;
            for (std::size_t p = 0; p < k; ++p) {
                // L[k][p] is stored in A[k,p] (lower), U[p][j] in A[p,j] (upper)
                sum += Aget(A, N, k, p) * Aget(A, N, p, j);
            }
            Aref(A, N, k, j) = Aget(A, N, k, j) - sum;
        });

        // pivot check
        double pivot = Aget(A, N, k, k);
        if (std::fabs(pivot) < 1e-16) {
            throw std::runtime_error("Zero (or tiny) pivot encountered at index " + std::to_string(k) +
                                     ". Add pivoting for numerical stability.");
        }

        // Compute L[i][k] for i = k+1..N-1
        // L[i][k] = (A[i][k] - sum_{p=0..k-1} L[i][p] * U[p][k]) / U[k][k]
        hpx::parallel::for_loop(par, k + 1, N, [&](std::size_t i) {
            double sum = 0.0;
            for (std::size_t p = 0; p < k; ++p) {
                sum += Aget(A, N, i, p) * Aget(A, N, p, k);
            }
            Aref(A, N, i, k) = (Aget(A, N, i, k) - sum) / pivot;
        });

        // Note: we do not need to "update" the trailing submatrix explicitly here because
        // subsequent iterations compute U and L entries via dot-products against already-stored L/U.
        // However, if you prefer the block algorithm you could explicitly do submatrix updates.
    }
}

// Utility: print matrix (for debugging)
void print_matrix(const std::vector<double> &A, std::size_t N, const std::string &name = "A") {
    std::cout << name << " =\n";
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            std::cout << std::setw(12) << std::setprecision(6) << Aget(A, N, i, j) << " ";
        }
        std::cout << "\n";
    }
}

// Optional: verify A = L*U (reconstruct and compare to original)
bool verify_lu(const std::vector<double> &LU, const std::vector<double> &Aorig, std::size_t N, double tol=1e-9) {
    // build L and U and multiply
    std::vector<double> LUprod(N*N, 0.0);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < N; ++k) {
                double L_ik = (k < i) ? Aget(LU, N, i, k) : (k == i ? 1.0 : 0.0);
                double U_kj = (k <= j) ? Aget(LU, N, k, j) : 0.0;
                sum += L_ik * U_kj;
            }
            LUprod[i*N + j] = sum;
        }
    }

    // compare LUprod to Aorig
    for (std::size_t idx = 0;
