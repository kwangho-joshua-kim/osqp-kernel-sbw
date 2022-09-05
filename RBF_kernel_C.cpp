#include <Rcpp.h>
#include <omp.h>
using namespace Rcpp;

// define both_non_NA(a, b)
inline bool both_non_NA(double a, double b) {
  return (!ISNAN(a) && !ISNAN(b));
}

// [[Rcpp::export]]
NumericMatrix RBF_kernel_C(NumericMatrix X, int c, IntegerVector set_c) {
  int n = X.nrow(), p = X.ncol();
  // default value following Hazlett (2020)
  double gamma = 1/double(2*p);
  // allocate the output matrix
  NumericMatrix out(n, c);
  
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      int cj = set_c[j]-1;
      double dist = 0;
      for (int k = 0; k < p; ++k) {
        double xi = X(i, k), xj = X(cj, k);
          dist += (xi - xj)*(xi - xj);
      }
      out(i, j) = exp(-gamma*dist);
    }
  }
  return out;
}
