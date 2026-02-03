std::vector<std::uint64_t> parallel_histogram(
    std::vector<double> const& data,
    std::size_t bins)
{
    std::vector<std::uint64_t> global_hist(bins, 0);
    //lock bin as data is accessed 
    std::vector<hpx::lcos::local::spinlock> bin_locks(bins);

    if (data.empty())
        return global_hist;

    std::size_t workers = std::max<std::size_t>(1, hpx::get_num_worker_threads());
    //split thrad workerse to roughly 4 groups 
    std::size_t chunk_size =
        std::max<std::size_t>(1, data.size() / (workers * 4));
    chunk_size = std::min<std::size_t>(chunk_size, data.size());

    auto worker = [&](std::size_t start, std::size_t count)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            double value = data[start + i];
            std::size_t bin = static_cast<std::size_t>(value * bins);
            if (bin >= bins)
                bin = bins - 1;
            //locking usage 
            std::lock_guard<hpx::lcos::local::spinlock> guard(bin_locks[bin]);
            ++global_hist[bin];
        }
    };

    std::vector<hpx::future<void>> tasks;
    tasks.reserve((data.size() + chunk_size - 1) / chunk_size);

    for (std::size_t pos = 0; pos < data.size(); pos += chunk_size)
    {
        std::size_t count = std::min(chunk_size, data.size() - pos);
        tasks.push_back(hpx::async(worker, pos, count));
    }

    hpx::wait_all(tasks);
    return global_hist;
}