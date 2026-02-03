double productWithInverses(std::vector<double> const& x)
{
    using hpx::execution::par;

    auto zipped_begin = hpx::util::make_zip_iterator(
        x.begin(),
        hpx::util::counting_iterator<std::size_t>(0));

    auto zipped_end = hpx::util::make_zip_iterator(
        x.end(),
        hpx::util::counting_iterator<std::size_t>(x.size()));

    return hpx::transform_reduce(
        par, zipped_begin, zipped_end, 1.0, std::multiplies<double>(),
        [](auto const& tup)
        {
            auto const& [value, index] = tup;
            return (index & 1) ? 1.0 / value : value;
        });
}