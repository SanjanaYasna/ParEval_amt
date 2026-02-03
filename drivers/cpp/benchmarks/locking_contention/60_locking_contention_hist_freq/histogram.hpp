
// Shared histogram that enforces locking on every bin update.
#pragma once 

using histogram_t = std::array<std::uint64_t, 256>;

struct SharedHistogram
{
    histogram_t bins{};
    std::array<std::mutex, 256> locks;

    SharedHistogram()
    {
        bins.fill(0);
    }

    void reset()
    {
        for (std::size_t i = 0; i < bins.size(); ++i)
        {
            std::lock_guard<std::mutex> g(locks[i]);
            bins[i] = 0;
        }
    }

    void increment(unsigned char value)
    {
        std::lock_guard<std::mutex> g(locks[value]);
        ++bins[value];
    }

    histogram_t snapshot()
    {
        histogram_t copy;
        for (std::size_t i = 0; i < bins.size(); ++i)
        {
            std::lock_guard<std::mutex> g(locks[i]);
            copy[i] = bins[i];
        }
        return copy;
    }
};