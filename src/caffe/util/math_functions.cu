#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}


template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}



template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}


template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype, typename Op>
__global__ void xopy_kernel_broadcast(int n, int dima0, int dima1, int dima2, int dima3, 
                                            int dimb0, int dimb1, int dimb2, int dimb3, 
                                            const Dtype* x, const Dtype* y, Dtype* z)
{
  int w0, h0, c0, n0;
  int w1, h1, c1, n1;
  int indexa, indexb, indexy;

  CUDA_KERNEL_LOOP(index, n) {

    w0 = index % dima3;
    h0 = (index / dima3) % dima2;
    c0 = (index / dima2 / dima3) % dima1;
    n0 = index / dima3 / dima2 / dima1;
    
    w1 = (dimb3 < dima3)? 0 : w0;
    h1 = (dimb2 < dima2)? 0 : h0;
    c1 = (dimb1 < dima1)? 0 : c0;
    n1 = (dimb0 < dima0)? 0 : n0;

    indexa = index;
    indexb = w1 + (h1 + (c1 + n1 * dimb1) * dimb2) * dimb3;
      
    Op o;
    z[index] = o(x[indexa], y[indexb]);
  }
}

template <typename Dtype, typename Op>
__global__ void gpu_dimension_reduction(int n, int dima0, int dima1, int dima2, int dima3, 
                                               int dimb0, int dimb1, int dimb2, int dimb3, 
                                               Dtype init, const Dtype* a, Dtype* b)
{
  int w0, h0, c0, n0;
  int w1, h1, c1, n1;
  int indexa, indexb, indexy;
  Dtype result = init;
  int wmax, hmax, cmax, nmax;
  if (dima0 > dimb0) nmax = dima0; else nmax = 1;
  if (dima1 > dimb1) cmax = dima1; else cmax = 1;
  if (dima2 > dimb2) hmax = dima2; else hmax = 1;
  if (dima3 > dimb3) wmax = dima3; else wmax = 1;
  // a complete trash, very slow version. need optimization
  CUDA_KERNEL_LOOP(index, n) {

    w0 = index % dimb3;
    h0 = (index / dimb3) % dimb2;
    c0 = (index / dimb2 / dimb3) % dimb1;
    n0 = index / dimb3 / dimb2 / dimb1;

    Op op;
    for (int dn = 0; dn < nmax; dn++) {
    for (int dc = 0; dc < cmax; dc++) {
    for (int dh = 0; dh < hmax; dh++) {
    for (int dw = 0; dw < wmax; dw++) {
      indexa = (w0+dw) + ((h0+dh) + ((c0+dc) + (n0+dn) * dima1) * dima2) * dima3;
      result = op(result, a[indexa]);
    }}}}
      
    b[index] = result;
  }
}


template <typename Dtype> class PrivateAddOp    { public: __device__ Dtype operator()(const Dtype a, const Dtype b){ return a+b; } };
template <typename Dtype> class PrivateMulOp    { public: __device__ Dtype operator()(const Dtype a, const Dtype b){ return a*b; } };
template <typename Dtype> class PrivateSubOp    { public: __device__ Dtype operator()(const Dtype a, const Dtype b){ return a-b; } };
template <typename Dtype> class PrivateRevSubOp { public: __device__ Dtype operator()(const Dtype a, const Dtype b){ return b-a; } };
template <typename Dtype> class PrivateDivOp    { public: __device__ Dtype operator()(const Dtype a, const Dtype b){ return a/b; } };
template <typename Dtype> class PrivateRevDivOp { public: __device__ Dtype operator()(const Dtype a, const Dtype b){ return b/a; } };

static bool sould_broadcast_a(const int dima[4], const int dimb[4])
{
  bool brd_a = 0;
  bool brd_b = 0;
  for (int i=0; i<4; i++)
  {
    if (dima[i] < dimb[i])
    {
      assert(dima[i] == 1);
      brd_a |= true;
    }
    else if (dima[i] > dimb[i])
    {
      assert(dimb[i] == 1);
      brd_b |= true;
    }
  }
  assert(brd_a ^ brd_b);
  return brd_a;
}

template <>
void caffe_gpu_add_broadcast<float>(const int dima[4], const int dimb[4],
                                    const float* a, const float* b, float* y) {

  int Na = dima[0] * dima[1] * dima[2] * dima[3];
  int Nb = dimb[0] * dimb[1] * dimb[2] * dimb[3];
  int N = (Na > Nb)? Na: Nb;

  if (sould_broadcast_a(dima, dimb))
  {
    xopy_kernel_broadcast<float, PrivateAddOp<float> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dimb[0], dimb[1], dimb[2], dimb[3], dima[0], dima[1], dima[2], dima[3], b, a, y);
  }
  else
  {
    xopy_kernel_broadcast<float, PrivateAddOp<float> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dima[0], dima[1], dima[2], dima[3], dimb[0], dimb[1], dimb[2], dimb[3], a, b, y);
  }
}

template <>
void caffe_gpu_sub_broadcast<float>(const int dima[4], const int dimb[4],
                                    const float* a, const float* b, float* y) {

  int Na = dima[0] * dima[1] * dima[2] * dima[3];
  int Nb = dimb[0] * dimb[1] * dimb[2] * dimb[3];
  int N = (Na > Nb)? Na: Nb;

  if (sould_broadcast_a(dima, dimb))
  {
    xopy_kernel_broadcast<float, PrivateRevSubOp<float> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dimb[0], dimb[1], dimb[2], dimb[3], dima[0], dima[1], dima[2], dima[3], b, a, y);
  }
  else
  {
    xopy_kernel_broadcast<float, PrivateSubOp<float> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dima[0], dima[1], dima[2], dima[3], dimb[0], dimb[1], dimb[2], dimb[3], a, b, y);
  }
}

template <>
void caffe_gpu_mul_broadcast<float>(const int dima[4], const int dimb[4],
                                    const float* a, const float* b, float* y) {

  int Na = dima[0] * dima[1] * dima[2] * dima[3];
  int Nb = dimb[0] * dimb[1] * dimb[2] * dimb[3];
  int N = (Na > Nb)? Na: Nb;

  if (sould_broadcast_a(dima, dimb))
  {
    xopy_kernel_broadcast<float, PrivateMulOp<float> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dimb[0], dimb[1], dimb[2], dimb[3], dima[0], dima[1], dima[2], dima[3], b, a, y);
  }
  else
  {
    xopy_kernel_broadcast<float, PrivateMulOp<float> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dima[0], dima[1], dima[2], dima[3], dimb[0], dimb[1], dimb[2], dimb[3], a, b, y);
  }
}

template <>
void caffe_gpu_div_broadcast<float>(const int dima[4], const int dimb[4],
                                    const float* a, const float* b, float* y) {

  int Na = dima[0] * dima[1] * dima[2] * dima[3];
  int Nb = dimb[0] * dimb[1] * dimb[2] * dimb[3];
  int N = (Na > Nb)? Na: Nb;

  if (sould_broadcast_a(dima, dimb))
  {
    xopy_kernel_broadcast<float, PrivateRevDivOp<float> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dimb[0], dimb[1], dimb[2], dimb[3], dima[0], dima[1], dima[2], dima[3], b, a, y);
  }
  else
  {
    xopy_kernel_broadcast<float, PrivateDivOp<float> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dima[0], dima[1], dima[2], dima[3], dimb[0], dimb[1], dimb[2], dimb[3], a, b, y);
  }
}

template <>
void caffe_gpu_add_broadcast<double>(const int dima[4], const int dimb[4],
                                    const double* a, const double* b, double* y) {

  int Na = dima[0] * dima[1] * dima[2] * dima[3];
  int Nb = dimb[0] * dimb[1] * dimb[2] * dimb[3];
  int N = (Na > Nb)? Na: Nb;

  if (sould_broadcast_a(dima, dimb))
  {
    xopy_kernel_broadcast<double, PrivateAddOp<double> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dimb[0], dimb[1], dimb[2], dimb[3], dima[0], dima[1], dima[2], dima[3], b, a, y);
  }
  else
  {
    xopy_kernel_broadcast<double, PrivateAddOp<double> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dima[0], dima[1], dima[2], dima[3], dimb[0], dimb[1], dimb[2], dimb[3], a, b, y);
  }
}

template <>
void caffe_gpu_sub_broadcast<double>(const int dima[4], const int dimb[4],
                                    const double* a, const double* b, double* y) {

  int Na = dima[0] * dima[1] * dima[2] * dima[3];
  int Nb = dimb[0] * dimb[1] * dimb[2] * dimb[3];
  int N = (Na > Nb)? Na: Nb;

  if (sould_broadcast_a(dima, dimb))
  {
    xopy_kernel_broadcast<double, PrivateRevSubOp<double> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dimb[0], dimb[1], dimb[2], dimb[3], dima[0], dima[1], dima[2], dima[3], b, a, y);
  }
  else
  {
    xopy_kernel_broadcast<double, PrivateSubOp<double> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dima[0], dima[1], dima[2], dima[3], dimb[0], dimb[1], dimb[2], dimb[3], a, b, y);
  }
}

template <>
void caffe_gpu_mul_broadcast<double>(const int dima[4], const int dimb[4],
                                    const double* a, const double* b, double* y) {

  int Na = dima[0] * dima[1] * dima[2] * dima[3];
  int Nb = dimb[0] * dimb[1] * dimb[2] * dimb[3];
  int N = (Na > Nb)? Na: Nb;

  if (sould_broadcast_a(dima, dimb))
  {
    xopy_kernel_broadcast<double, PrivateMulOp<double> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dimb[0], dimb[1], dimb[2], dimb[3], dima[0], dima[1], dima[2], dima[3], b, a, y);
  }
  else
  {
    xopy_kernel_broadcast<double, PrivateMulOp<double> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dima[0], dima[1], dima[2], dima[3], dimb[0], dimb[1], dimb[2], dimb[3], a, b, y);
  }
}

template <>
void caffe_gpu_div_broadcast<double>(const int dima[4], const int dimb[4],
                                    const double* a, const double* b, double* y) {
  int Na = dima[0] * dima[1] * dima[2] * dima[3];
  int Nb = dimb[0] * dimb[1] * dimb[2] * dimb[3];
  int N = (Na > Nb)? Na: Nb;
  if (sould_broadcast_a(dima, dimb))
  {
    xopy_kernel_broadcast<double, PrivateRevDivOp<double> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dimb[0], dimb[1], dimb[2], dimb[3], dima[0], dima[1], dima[2], dima[3], b, a, y);
  }
  else
  {
    xopy_kernel_broadcast<double, PrivateDivOp<double> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dima[0], dima[1], dima[2], dima[3], dimb[0], dimb[1], dimb[2], dimb[3], a, b, y);
  }
}

template <>
void caffe_gpu_sum_reduce<float>(const int dima[4], const int dimb[4],
                             const float* a, float* b) {

  int Na = dima[0] * dima[1] * dima[2] * dima[3];
  int Nb = dimb[0] * dimb[1] * dimb[2] * dimb[3];
  int N = (Na < Nb)? Na: Nb;

  if (sould_broadcast_a(dima, dimb))
  {
    assert(1==0);
  }
  else
  {
    gpu_dimension_reduction <float, PrivateAddOp<float> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dima[0], dima[1], dima[2], dima[3], dimb[0], dimb[1], dimb[2], dimb[3], 0, a, b);
  }
}

template <>
void caffe_gpu_sum_reduce<double>(const int dima[4], const int dimb[4],
                             const double* a, double* b) {

  int Na = dima[0] * dima[1] * dima[2] * dima[3];
  int Nb = dimb[0] * dimb[1] * dimb[2] * dimb[3];
  int N = (Na < Nb)? Na: Nb;

  if (sould_broadcast_a(dima, dimb))
  {
    assert(1==0);
  }
  else
  {
    gpu_dimension_reduction <double, PrivateAddOp<double> >
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dima[0], dima[1], dima[2], dima[3], dimb[0], dimb[1], dimb[2], dimb[3], 0, a, b);
  }
}


template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}


template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}


}  // namespace caffe
