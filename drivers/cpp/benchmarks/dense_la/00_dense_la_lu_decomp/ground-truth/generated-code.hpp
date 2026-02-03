

void luFactorize(std::vector<double>& A, std::size_t N)
{
    if (A.size() != N * N)
    {
        throw std::invalid_argument("A must be an N x N matrix (row-major).");
    }

    auto index = [N](std::size_t r, std::size_t c) noexcept -> std::size_t
    {
        return r * N + c;
    };

    constexpr double eps = 1e-12;

    for (std::size_t k = 0; k < N; ++k)
    {
        // Compute the k-th row of U (columns k..N-1).
        hpx::for_loop(hpx::execution::par, k, N,
            [&](std::size_t j)
            {
                double sum = 0.0;
                for (std::size_t p = 0; p < k; ++p)
                {
                    sum += A[index(k, p)] * A[index(p, j)];
                }
                A[index(k, j)] -= sum;
            });

        double pivot = A[index(k, k)];
        if (std::abs(pivot) < eps)
        {
            throw std::runtime_error("Matrix is singular or ill-conditioned (zero pivot encountered).");
        }

        // Compute the k-th column of L (rows k+1..N-1); L has 1s on the diagonal (implicit).
        hpx::for_loop(hpx::execution::par, k + 1, N,
            [&](std::size_t i)
            {
                double sum = 0.0;
                for (std::size_t p = 0; p < k; ++p)
                {
                    sum += A[index(i, p)] * A[index(p, k)];
                }
                A[index(i, k)] = (A[index(i, k)] - sum) / pivot;
            });
    }
}