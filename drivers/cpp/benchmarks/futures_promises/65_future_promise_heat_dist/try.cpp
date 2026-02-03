// Simplified 1D heat-distribution example (HPX)
//  - no HPX components
//  - no partition_data / partition_server
//  - only a minimal partition wrapper (shared_future<vector<double>>)
//
// Compile with HPX headers and libraries. Example (adjust include/library paths):
//   c++ -std=c++17 simplified_heat.cpp -I${HPX_INCLUDE_DIR} -L${HPX_LIB_DIR} \
//       -lhpx_init -lhpx -lpthread -O2 -o simplified_heat

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/include/dataflow.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/modules/program_options.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>



///////////////////////////////////////////////////////////////////////////////
// Command-line variables
bool header = true;    // print csv heading
double k   = 0.5;      // heat transfer coefficient
double dt  = 1.0;      // time step
double dx  = 1.0;      // grid spacing

inline std::size_t idx(std::size_t i, int dir, std::size_t size)
{
    if (i == 0 && dir == -1)
        return size - 1;
    if (i == size - 1 && dir == +1)
        return 0;

    HPX_ASSERT((i + dir) < size);
    return i + dir;
}

///////////////////////////////////////////////////////////////////////////////
// Minimal partition type: wraps a shared_future<vector<double>>
struct partition
{
    using data_type   = std::vector<double>;
    using future_type = hpx::shared_future<data_type>;

    partition() = default;
    explicit partition(future_type f)
      : data_(std::move(f))
    {}

    partition& operator=(future_type f)
    {
        data_ = std::move(f);
        return *this;
    }

    future_type const& get_future() const
    {
        return data_;
    }

    bool valid() const
    {
        return data_.valid();
    }

private:
    future_type data_;
};

///////////////////////////////////////////////////////////////////////////////
struct stepper
{
    using space = std::vector<partition>;

    static double heat(double left, double middle, double right)
    {
        return middle + (k * dt / (dx * dx)) *
            (left - 2.0 * middle + right);
    }

    static partition heat_part(partition const& left,
                               partition const& middle,
                               partition const& right)
    {
        using hpx::dataflow;
        using hpx::launch;
        using hpx::util::unwrapping;

        auto kernel = unwrapping([](partition::data_type const& l,
                                    partition::data_type const& m,
                                    partition::data_type const& r)
                                      -> partition::data_type
        {
            std::size_t const N = m.size();
            HPX_ASSERT(N >= 2);

            partition::data_type next(N);
            for (std::size_t i = 1; i < N - 1; ++i)
            {
                next[i] = heat(m[i - 1], m[i], m[i + 1]);
            }

            next[0]     = heat(l.back(),        m[0],     m[1]);
            next[N - 1] = heat(m[N - 2],        m[N - 1], r.front());
            return next;
        });

        return partition(dataflow(
            launch::async, kernel,
            left.get_future(), middle.get_future(), right.get_future()).share());
    }

    space do_work(std::size_t np, std::size_t nx, std::size_t nt)
    {
        if (np == 0 || nx == 0)
            return {};

        std::vector<space> layers(2);
        for (space& s : layers)
            s.resize(np);

        for (std::size_t p = 0; p < np; ++p)
        {
            partition::data_type tile(nx);
            double const base = double(p * nx);
            for (std::size_t j = 0; j < nx; ++j)
                tile[j] = base + double(j);

            layers[0][p] =
                partition(hpx::make_ready_future(std::move(tile)).share());
        }

        for (std::size_t t = 0; t < nt; ++t)
        {
            space const& current = layers[t % 2];
            space& next          = layers[(t + 1) % 2];

            for (std::size_t i = 0; i < np; ++i)
            {
                next[i] = heat_part(
                    current[idx(i, -1, np)],
                    current[i],
                    current[idx(i, +1, np)]);
            }
        }

        return layers[nt % 2];
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    std::uint64_t np = vm["np"].as<std::uint64_t>();
    std::uint64_t nx = vm["nx"].as<std::uint64_t>();
    std::uint64_t nt = vm["nt"].as<std::uint64_t>();

    if (vm.count("no-header"))
        header = false;

    stepper step;

    std::uint64_t t0 = hpx::chrono::high_resolution_clock::now();
    stepper::space result = step.do_work(np, nx, nt);
    std::uint64_t elapsed = hpx::chrono::high_resolution_clock::now() - t0;

    if (vm.count("results"))
    {
        for (std::size_t p = 0; p < result.size(); ++p)
        {
            std::vector<double> tile = result[p].get_future().get();
            hpx::cout << "U[" << p << "] = {";
            for (std::size_t j = 0; j < tile.size(); ++j)
            {
                if (j != 0)
                    hpx::cout << ", ";
                hpx::cout << tile[j];
            }
            hpx::cout << "}\n";
        }
        hpx::cout << hpx::flush;
    }

    std::uint64_t num_threads = hpx::get_num_worker_threads();
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    options_description desc("options");
    desc.add_options()
        ("results", "print final value of each partition")
        ("nx", value<std::uint64_t>()->default_value(10),
            "grid points per partition")
        ("nt", value<std::uint64_t>()->default_value(45),
            "number of time steps")
        ("np", value<std::uint64_t>()->default_value(10),
            "number of partitions")
        ("k",  value<double>(&k )->default_value(0.5),
            "heat transfer coefficient")
        ("dt", value<double>(&dt)->default_value(1.0),
            "time-step size")
        ("dx", value<double>(&dx)->default_value(1.0),
            "grid spacing")
        ("no-header", "do not print CSV header row");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc;
    return hpx::init(argc, argv, init_args);
}