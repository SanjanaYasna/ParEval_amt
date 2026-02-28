void ifft(std::vector<std::complex<double>> &x) {
    const size_t N = x.size();

    // Conjugate the input
    hpx::for_each(hpx::execution::par, x.begin(), x.end(),
        [](std::complex<double> &val) {
            val = std::conj(val);
        });

    // Apply forward FFT
    fft(x);

    // Conjugate and scale by 1/N
    hpx::for_each(hpx::execution::par, x.begin(), x.end(),
        [N](std::complex<double> &val) {
            val = std::conj(val) / static_cast<double>(N);
        });
}