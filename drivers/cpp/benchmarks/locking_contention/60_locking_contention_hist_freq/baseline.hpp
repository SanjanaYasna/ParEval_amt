
#include "histogram.hpp"

histogram_t correct_parallel_histogram(
    std::vector<unsigned char> const& data,
    SharedHistogram& shared)
{
    shared.reset();
    for (unsigned char uc : data)
        shared.increment(uc);         // still acquires the per-bin mutex
    return shared.snapshot();
}