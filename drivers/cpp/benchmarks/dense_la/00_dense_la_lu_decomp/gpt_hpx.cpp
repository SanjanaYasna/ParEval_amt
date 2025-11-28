#include <hpx/hpx_main.hpp>
#include <hpx/async.hpp>
#include <hpx/future.hpp>

#include <vector>
#include <iostream>
#include <stdexcept>

// Factorize the matrix A in-place into A = L * U
// A : row-major std::vector<double> of size N*N
// After factorization:
//  - For i <= j: A[i*N + j] is U(i,j)
//  - For i >  j: A[i*N + j] is L(i,j) (diagonal of L is implicitly 1)
void luFactorize(std::vector<double> &A, std::size_t N) {
    if (A.size() != N * N) throw std::invalid_argument("A size mismatch");

    auto idx = [N](std::size_t i, std::size_t j) { return i * N + j; };

    for (std::size_t k = 0; k < N; ++k) {
        // --- Compute U[k, j] for j = k..N-1 ---
        std::vector<hpx::future<void>> u_futs;
        u_futs.reserve(N - k);
        for (std::size_t j = k; j < N; ++j) {
            u_futs.push_back(hpx::async([k, j, N, &A, &idx]() {
                // U[k,j] = A[k,j] - sum_{t=0..k-1} L[k,t] * U[t,j]
                double sum = 0.0;
                for (std::size_t t = 0; t < k; ++t) {
                    double L_kt = (k > t) ? A[idx(k,t)] : (t == k ? 1.0 : 0.0); // L entries for t<k are stored
                    double U_tj = A[idx(t,j)]; // U entries for t <= j are stored
                    sum += L_kt * U_tj;
                }
                A[idx(k,j)] = A[idx(k,j)] - sum; // store U[k,j]
            }));
        }
        // Wait for U row to be computed
        hpx::when_all(u_futs).get();

        // Check for zero pivot (U[k,k] == 0)
        double pivot = A[idx(k,k)];
        if (pivot == 0.0) {
            throw std::runtime_error("Zero pivot encountered; pivoting required for numerical stability.");
        }

        // --- Compute L[i, k] for i = k+1..N-1 ---
        std::vector<hpx::future<void>> l_futs;
        l_futs.reserve((N > k+1) ? (N - (k + 1)) : 0);
        for (std::size_t i = k + 1; i < N; ++i) {
            l_futs.push_back(hpx::async([k, i, N, &A, &idx, pivot]() {
                // L[i,k] = (A[i,k] - sum_{t=0..k-1} L[i,t] * U[t,k]) / U[k,k]
                double sum = 0.0;
                for (std::size_t t = 0; t < k; ++t) {
                    double L_it = (i > t) ? A[idx(i,t)] : (t == i ? 1.0 : 0.0);
                    double U_tk = A[idx(t,k)];
                    sum += L_it * U_tk;
                }
                A[idx(i,k)] = (A[idx(i,k)] - sum) / pivot; // store L[i,k]
            }));
        }
        // Wait for L column to be computed
        hpx::when_all(l_futs).get();
    }
}

// small helper to print matrix
void printMatrix(const std::vector<double>& A, std::size_t N) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            std::cout << A[i * N + j] << (j + 1 == N ? "" : "\t");
        }
        std::cout << "\n";
    }
}

int main() {
    // Example from your prompt:
    // input: [[4,3],[6,3]]
    // represented row-major: [4,3,6,3]
    std::vector<double> A = {4.0, 3.0,
                             6.0, 3.0};
    std::size_t N = 2;

    std::cout << "Before:\n";
    printMatrix(A, N);

    try {
        luFactorize(A, N);
    } catch (const std::exception &e) {
        std::cerr << "LU failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nAfter (L below diag, U on and above diag; L diag is 1 implicitly):\n";
    printMatrix(A, N);
    // Expected output:
    // [4, 3]
    // [1.5, -1.5]

    return 0;
}
