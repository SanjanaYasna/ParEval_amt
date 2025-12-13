

void solveLinearSystem(std::vector<COOElement> const& A,
                              std::vector<double> const& b,
                              std::vector<double>& x,
                              std::size_t N)
{
    if (b.size() != N)
        throw std::invalid_argument("b must have N entries");

    std::vector<std::vector<double>> matrix(N, std::vector<double>(N, 0.0));
    std::vector<double> b_copy = b;

    // Fill dense matrix from COO entries (sequential; cost is negligible compared to elimination)
    for (auto const& element : A)
    {
        if (element.row >= N || element.column >= N)
            throw std::out_of_range("COO element index out of bounds");

        matrix[element.row][element.column] = element.value;
    }

    x.assign(N, 0.0);

    constexpr double eps = 1e-12;

    for (std::size_t i = 0; i < N; ++i)
    {
        // Pivot search (serial to keep it simple and stable)
        double maxEl = std::abs(matrix[i][i]);
        std::size_t maxRow = i;
        for (std::size_t k = i + 1; k < N; ++k)
        {
            double val = std::abs(matrix[k][i]);
            if (val > maxEl)
            {
                maxEl = val;
                maxRow = k;
            }
        }

        if (maxEl < eps)
            throw std::runtime_error("Matrix is singular or ill-conditioned.");

        if (maxRow != i)
        {
            for (std::size_t k = i; k < N; ++k)
                std::swap(matrix[maxRow][k], matrix[i][k]);
            std::swap(b_copy[maxRow], b_copy[i]);
        }

        double pivot = matrix[i][i];

        // Parallel elimination for rows below the pivot
        hpx::for_loop(hpx::execution::par, i + 1, N,
            [&](std::size_t row)
            {
                double factor = -matrix[row][i] / pivot;
                if (std::abs(factor) < eps)
                {
                    matrix[row][i] = 0.0;
                    return;
                }

                matrix[row][i] = 0.0;
                for (std::size_t col = i + 1; col < N; ++col)
                {
                    matrix[row][col] += factor * matrix[i][col];
                }
                b_copy[row] += factor * b_copy[i];
            });
    }

    // Back substitution (serial; dependencies are triangular)
    for (int i = static_cast<int>(N) - 1; i >= 0; --i)
    {
        double sum = 0.0;
        for (std::size_t j = static_cast<std::size_t>(i) + 1; j < N; ++j)
        {
            sum += matrix[i][j] * x[j];
        }

        double diag = matrix[i][i];
        if (std::abs(diag) < eps)
            throw std::runtime_error("Matrix is singular or ill-conditioned.");

        x[static_cast<std::size_t>(i)] = (b_copy[static_cast<std::size_t>(i)] - sum) / diag;
    }
}