#include "histogram.hpp"
histogram_t parallel_histogram(
    std::vector<unsigned char> const& data,
    SharedHistogram& shared)
{
    shared.reset();

    if (data.empty())
        return shared.snapshot();

    std::size_t const workers =
        std::max<std::size_t>(1, hpx::get_num_worker_threads());
    std::size_t const chunk =
        (data.size() + workers - 1) / workers;

    std::vector<hpx::future<void>> tasks;
    tasks.reserve(workers);

    for (std::size_t tid = 0; tid < workers; ++tid)
    {
        std::size_t const begin = tid * chunk;
        if (begin >= data.size())
            break;

        std::size_t const end = std::min(begin + chunk, data.size());

        tasks.emplace_back(hpx::async([begin, end, &data, &shared]() {
            for (std::size_t i = begin; i < end; ++i)
                shared.increment(data[i]);     // enforced per-bin locking
        }));
    }

    hpx::wait_all(tasks);
    return shared.snapshot();
}