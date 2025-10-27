// Driver for 03_dense_la_axpy for HPX
// /* Compute z = alpha*x+y where x and y are vectors. Store the result in z.
//    Use HPX to compute in parallel. Assume main() and hpx_main() functions have already been initialized and simply complete the function below.
//    Example:
//    
//    input: x=[1, -5, 2, 9] y=[0, 4, 1, -1] alpha=2
//    output: z=[2, -6, 5, 17]
// */
// void axpy(double alpha, std::vector<hpx::shared_future<std::vector<double>>> &x,  std::vector<hpx::shared_future<std::vector<double>>> &y, std::vector<hpx::shared_future<std::vector<double>>> &z) {

//can be futurized? most likely wanted answer https://github.com/mariomulansky/hpx_odeint/blob/a247d77b3149ff7dc8674ef13934570b80fe46cc/tex/futurization.pdf 
