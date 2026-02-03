void fft(std::vector<std::complex<double>> const& x, std::vector<std::complex<double>>& output) {
    const size_t N = x.size();
    output.resize(N);

    // Base case
    if (N <= 1) {
        if (N == 1) output[0] = x[0];
        return;
    }

    // Divide into even and odd indexed elements
    std::vector<std::complex<double>> even(N / 2), odd(N / 2);
    for (size_t i = 0; i < N / 2; ++i) {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    // Compute FFTs recursively (parallel for large sizes)
    std::vector<std::complex<double>> fft_even, fft_odd;

    constexpr size_t PARALLEL_THRESHOLD = 64;
    if (N >= PARALLEL_THRESHOLD) {
        // Parallel execution
        auto fut_even = hpx::async([&even]() {
            std::vector<std::complex<double>> result;
            fft(even, result);
            return result;
        });

        fft(odd, fft_odd);
        fft_even = fut_even.get();
    } else {
        // Sequential for small sizes
        fft(even, fft_even);
        fft(odd, fft_odd);
    }

    // Combine using butterfly operation
    const double PI = 3.14159265358979323846;
    for (size_t k = 0; k < N / 2; ++k) {
        std::complex<double> t = std::polar(1.0, -2.0 * PI * k / N) * fft_odd[k];
        output[k] = fft_even[k] + t;
        output[k + N / 2] = fft_even[k] - t;
    }
}

// // reverse the lowest 'bits' bits of x
// static std::size_t reverse_bits(std::size_t x, unsigned bits) {
//     std::size_t y = 0;
//     for (unsigned i = 0; i < bits; ++i) {
//         y = (y << 1) | (x & 1u);
//         x >>= 1;
//     }
//     return y;
// }

// // Compute the forward FFT (out-of-place) using HPX for parallelism.
// // - x: input vector (size n, must be power of two)
// // - output: will be resized to n and filled with the DFT of x
// void fft(std::vector<std::complex<double>> const& x, std::vector<std::complex<double>> &output) {
//     const std::size_t n = x.size();
//     if (n == 0) {
//         output.clear();
//         return;
//     }
//     // must be power of two
//     if ((n & (n - 1)) != 0) {
//         throw std::invalid_argument("fft: input size must be a power of two");
//     }

//     // compute log2(n)
//     unsigned levels = 0;
//     for (std::size_t t = n; t > 1; t >>= 1) ++levels;

//     output.resize(n);

//     // Bit-reversed copy in parallel: output[rev(i)] = x[i]
//     {
//         std::vector<hpx::future<void>> futures;
//         futures.reserve(n);
//         for (std::size_t i = 0; i < n; ++i) {
//             futures.push_back(hpx::async([i, levels, &x, &output]() {
//                 output[reverse_bits(i, levels)] = x[i];
//             }));
//         }
//         hpx::wait_all(futures);
//     }

//     const double PI = std::acos(-1.0);

//     // Iterative FFT: len is current transform size (2, 4, 8, ...)
//     for (std::size_t len = 2; len <= n; len <<= 1) {
//         const double angle = -2.0 * PI / static_cast<double>(len); // forward FFT sign
//         const std::complex<double> wlen(std::cos(angle), std::sin(angle));
//         const std::size_t half = len / 2;

//         // Launch one task per block of length 'len'
//         std::vector<hpx::future<void>> tasks;
//         tasks.reserve(n / len);
//         for (std::size_t start = 0; start < n; start += len) {
//             tasks.push_back(hpx::async([start, len, half, wlen, &output]() {
//                 std::complex<double> w(1.0, 0.0);
//                 for (std::size_t j = 0; j < half; ++j) {
//                     const std::size_t idx = start + j;
//                     const std::size_t idx2 = idx + half;
//                     const std::complex<double> u = output[idx];
//                     const std::complex<double> v = w * output[idx2];
//                     output[idx] = u + v;
//                     output[idx2] = u - v;
//                     w *= wlen;
//                 }
//             }));
//         }
//         hpx::wait_all(tasks);
//     }
// }
