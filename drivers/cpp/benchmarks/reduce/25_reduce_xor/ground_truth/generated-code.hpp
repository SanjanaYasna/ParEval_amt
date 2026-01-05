bool reduceLogicalXOR(std::vector<bool> const& x)
{
    return hpx::reduce(
        hpx::execution::par,      // parallel execution policy
        x.begin(), x.end(),
        false,                    // identity element for XOR
        [](bool a, bool b) { return a != b; }   // logical XOR
    );
}