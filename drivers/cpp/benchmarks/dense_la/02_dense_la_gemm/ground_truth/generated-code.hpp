
void gemm(std::vector<double> const& A, std::vector<double> const& B, std::vector<double> &C, size_t M, size_t K, size_t N) {
    //1.10.0 : 
    hpx::experimental::for_loop(hpx::execution::par, 0, M, [&](int i){
   // hpx::parallel::for_loop(hpx::parallel::execution::par, 0, M, [&](int i){
        for (int k = 0; k < K; k++) {
            int sum = 0;
            for (int j = 0; j < N; j++) {
                C[i*N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    });
} 
//void gemm(std::vector<double> const& A, std::vector<double> const& B, std::vector<double> &C, size_t M, size_t K, size_t N) { }
