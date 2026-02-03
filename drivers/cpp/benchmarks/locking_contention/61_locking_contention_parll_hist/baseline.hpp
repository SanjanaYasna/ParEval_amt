#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

// // Serial reference histogram that still uses per-bin locks.
// std::vector<std::uint64_t> correct_parallel_histogram(
//     std::vector<double> const& data,
//     std::size_t bins)
// {
//     std::vector<std::uint64_t> histogram(bins, 0);
//     std::vector<std::mutex> bin_mutexes(bins);

//     for (double value : data)
//     {
//         std::size_t bin = static_cast<std::size_t>(value * bins);
//         if (bin >= bins)
//             bin = bins - 1;

//         std::lock_guard<std::mutex> guard(bin_mutexes[bin]);
//         ++histogram[bin];
//     }

//     return histogram;
// }

std::vector<std::uint64_t> correct_parallel_histogram(
    std::vector<double> const& data,
    std::size_t bins)
{
    if (bins == 0)
        return {};    
    std::vector<std::uint64_t> hist(bins, 0);
    if (data.empty())
        return hist;

    for (double value : data)
    {
        std::size_t bin = static_cast<std::size_t>(value * bins);
        if (bin >= bins)             // values >= 1.0 go into the last bin
            bin = bins - 1;
        ++hist[bin];
    }

    return hist;
    }
