#include <iostream>
#include <hpx/hpx_main.hpp>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include <chrono>

class Context;
extern "C++" {
    /* todo -- these could all be in a class, but I'm not sure if virtual 
       overloading would incur a noticable overhead here. */
    Context *init();
    void compute(Context *ctx);
    void best(Context *ctx);
    bool validate(Context *ctx);
    void reset(Context *ctx);
    void destroy(Context *ctx);
}

int main(int argc, char **argv) {
    
}