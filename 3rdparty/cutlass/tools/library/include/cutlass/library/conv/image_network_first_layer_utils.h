#pragma once

#include "cutlass/conv/convolution.h"
#include "cutlass/array.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/layout/tensor.h"

#include "cutlass_common.h"

namespace cutlass_build_env {
namespace library {

using namespace cutlass;

////////////////////////////////////////////////////////////////////////////////////////////////////

static const uint64_t AMPERE_MEM_DESC_DEFAULT = uint64_t(0x1000000000000000ul);
static const uint64_t MEM_DESC_DEFAULT = AMPERE_MEM_DESC_DEFAULT;

static CUTLASS_DEVICE uint32_t half_to_half2(half_t x) {
  uint32_t res = 0;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  __half2 tmp = __half2half2(reinterpret_cast<__half&>(x));
  res = reinterpret_cast<int&>(tmp);
#else
    assert(0);
#endif

  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static CUTLASS_DEVICE uint32_t float2_to_half2(float x, float y) {
  uint32_t res = 0;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  asm volatile(
      "{ .reg.f16 lo, hi; \n"
      "   cvt.rn.f16.f32 lo, %1;\n"
      "   cvt.rn.f16.f32 hi, %2;\n"
      "   mov.b32 %0, {lo, hi};\n"
      "}\n"
      : "=r"(res)
      : "f"(x), "f"(y));
#else
    assert(0);
#endif

  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static CUTLASS_DEVICE uint32_t fma_fp16x2(uint32_t a, uint32_t x, uint32_t b) {
    uint32_t res = 0;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                    : "=r"(res)
                    : "r"(a), "r"(x), "r"(b));
#else
    assert(0);
#endif

    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static CUTLASS_DEVICE uint32_t relu_fp16x2(uint32_t x, uint32_t lb = 0u) {
    uint32_t res = 0;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile( "max.NaN.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(lb));
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // To support values like NAN and INF, the hmax2 defined in this header file can't be used
    __half2 half2_lb = *reinterpret_cast<__half2*>(&lb);
    __half half_lb_low = __low2half(half2_lb);
    __half half_lb_high = __high2half(half2_lb);
    __half2 half2_x = *reinterpret_cast<__half2*>(&x);
    __half x_low = __low2half(half2_x);
    __half x_high = __high2half(half2_x);
    __half res_low, res_high;
    if ( x_low < half_lb_low ) {
        res_low = half_lb_low;
    }
    else {
        res_low = x_low;
    }
    if ( x_high < half_lb_high ) {
        res_high = half_lb_high;
    }
    else {
        res_high = x_high;
    }
    __half2 half2_res = __halves2half2(res_low, res_high);
    res = *reinterpret_cast<uint32_t*>(&half2_res);
#else
    assert(0);
#endif

    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static CUTLASS_DEVICE uint32_t relu_ub_fp16x2(uint32_t x, uint32_t ub) {
    uint32_t res = 0;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile( "min.NaN.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(ub));
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#if ((__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2) || __CUDACC_VER_MAJOR__ >= 11)
    // The logic of lop3 is (x&sela)|(ub&~sela)
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\n" \
        "\t set.leu.u32.f16x2 sela, %1, %2;\n" \
        "\t lop3.b32 %0, %1, %2, sela, 0xe4;\n"
        "}\n" : "=r"(res) : "r"(x), "r"(ub));
#else
    // For compiler version <=10.1
    uint32_t tmp = 0;
    asm volatile("set.leu.f16x2.f16x2 %0, %1, %2;\n" : "=r"(tmp) : "r"(x), "r"(ub));
    if( tmp == 0x3c003c00 ) { tmp = 0xffffffff; }
    else if( tmp == 0x3c000000 ) { tmp = 0xffff0000; }
    else if( tmp == 0x00003c00 ) { tmp = 0x0000ffff; }
    // The logic of lop3 is (x&sela)|(ub&~sela)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xe4;\n" : "=r"(res) : "r"(x), "r"(ub), "r"(tmp));
#endif
#else
    assert(0);
#endif

    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static CUTLASS_DEVICE uint32_t clippedRelu(uint32_t in, uint32_t lb, uint32_t ub)  {
    uint32_t res = 0;
    res = relu_fp16x2(in, lb);
    res = relu_ub_fp16x2(res, ub);
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static CUTLASS_DEVICE uint32_t swish( uint32_t in ) {
    uint32_t res = 0;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    __half2 tmp = __float2half2_rn(0.500000f);
    uint32_t literal0 = reinterpret_cast<uint32_t&>(tmp);
    // v0 = 0.5 * in
    uint32_t v0;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(v0) : "r"(in), "r"(literal0));
    // v1 = tanh(v0)
    uint32_t v1;
    asm volatile ("tanh.approx.f16x2 %0, %1;" : "=r"(v1) : "r"(v0));
    // res = v0 * v1 + v0
    res = fma_fp16x2(v0, v1, v0);
#else
    assert(0);
#endif

    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kLdgWidth>
CUTLASS_DEVICE void ldg(int *dst, char const *ptr);

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specilizations for LDG inline ptxas
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void ldg<32>(
    int *dst,
    char const *ptr) {

    asm volatile(
        "{\n" \
        "\tld.global.b32 %0, [%1];\n" \
        "}\n"
        : "=r"(dst[0])
        : "l"(ptr));
}

template <>
CUTLASS_DEVICE void ldg<64>(
    int *dst,
    char const *ptr) {

    asm volatile(
        "{\n" \
        "\tld.global.v2.b32 {%0, %1}, [%2];\n" \
        "}\n"
        : "=r"(dst[0]), "=r"(dst[1])
        : "l"(ptr));
}

template <>
CUTLASS_DEVICE void ldg<128>(
    int *dst,
    char const *ptr) {

    asm volatile(
        "{\n" \
        "\tld.global.v4.b32 {%0, %1, %2, %3}, [%4];\n" \
        "}\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "l"(ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kStsWidth>
CUTLASS_DEVICE void sts(int const *src, char *ptr);

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specilizations for STS inline ptxas
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void sts<32>(
    int const *src,
    char *ptr) {

  reinterpret_cast<Array<int, 1> *>(ptr)[0] = reinterpret_cast<Array<int, 1> const *>(src)[0];
}

template <>
CUTLASS_DEVICE void sts<64>(
    int const *src,
    char *ptr) {

  reinterpret_cast<Array<int, 2> *>(ptr)[0] = reinterpret_cast<Array<int, 2> const *>(src)[0];
}

template <>
CUTLASS_DEVICE void sts<128>(
    int const *src,
    char *ptr) {

  reinterpret_cast<Array<int, 4> *>(ptr)[0] = reinterpret_cast<Array<int, 4> const *>(src)[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kFltR,
         int kFltS>
CUTLASS_DEVICE void wgrad_ldg_a(
    int const kThreadsPerRow,
    int const kTidxModThreadsPerRow,
    int const kH,
    int const kW,
    int const kHEnd,
    int const kWEnd,
    uint16_t const *gmem_img,
    int *fetch_img) {

  // 1xLDG.64
  using LongIndex = cutlass::layout::TensorNHWC::LongIndex;
  // The offset in one row where the threads starts loading.
  LongIndex const kGmemImgOffset = (kW + kTidxModThreadsPerRow) * 4;

  if ((unsigned)kH < kHEnd && (unsigned)(kW + kTidxModThreadsPerRow) < kWEnd) {
    char const *ptr = reinterpret_cast<char const *>(&gmem_img[kGmemImgOffset]);
    int *dst = fetch_img;
    ldg<64>(dst, ptr);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specilizations for LDG.A
// For r3s3 case, we need one LDG.64 and one LDG.32 to load matrix A
// For other cases, we only need one LDG.64 to load matrix A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void wgrad_ldg_a<3,3>(
    int const kThreadsPerRow,
    int const kTidxModThreadsPerRow,
    int const kH,
    int const kW,
    int const kHEnd,
    int const kWEnd,
    uint16_t const* gmem_img,
    int *fetch_img) {

  // 1xLDG.64 + 1xLDG.32
  using LongIndex = cutlass::layout::TensorNHWC::LongIndex;
  int w = kW + kTidxModThreadsPerRow;
  LongIndex gmem_img_offset = w * 4;

  if ((unsigned)kH < kHEnd && (unsigned)(w) < kWEnd) {
    char const *ptr = reinterpret_cast<char const *>(&gmem_img[gmem_img_offset]);
    int *dst = fetch_img;
    ldg<64>(dst, ptr);
  }

  w = kW + kThreadsPerRow + kTidxModThreadsPerRow / 2;
  int const kC = kTidxModThreadsPerRow % 2 * 2;
  gmem_img_offset = w * 4 + kC;

  if ((unsigned)kH < kHEnd && (unsigned)(w) < kWEnd) {
    char const *ptr = reinterpret_cast<char const *>(&gmem_img[gmem_img_offset]);
    int *dst = &fetch_img[2];
    ldg<32>(dst, ptr);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kFltR,
         int kFltS>
CUTLASS_DEVICE void wgrad_sts_a(
    int const kThreadsPerRow,
    int const kTidxModThreadsPerRow,
    int const kImgW,
    uint16_t *smem_img,
    int *fetch_img) {

  // 1xSTS.64
  using LongIndex = cutlass::layout::TensorNHWC::LongIndex;
  int const kStoreW = kTidxModThreadsPerRow;
  LongIndex const kOffset = kStoreW * 4;

  if (kStoreW < kImgW) {
    char *ptr = reinterpret_cast<char *>(&smem_img[kOffset]);
    int const *src = fetch_img;
    sts<64>(src, ptr);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specilizations for STS.A (matches LDG.A)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void wgrad_sts_a<3,3>(
    int const kThreadsPerRow,
    int const kTidxModThreadsPerRow,
    int const kImgW,
    uint16_t *smem_img,
    int *fetch_img) {

  // 1xSTS.64 + 1xSTS.32
  using LongIndex = cutlass::layout::TensorNHWC::LongIndex;
  int store_w = kTidxModThreadsPerRow;
  LongIndex offset = store_w * 4;

  if (store_w < kImgW) {
    char *ptr = reinterpret_cast<char *>(&smem_img[offset]);
    int const *src = fetch_img;
    sts<64>(src, ptr);
  }

  int const kStoreC = kTidxModThreadsPerRow % 2 * 2;
  store_w = kThreadsPerRow + kTidxModThreadsPerRow / 2;
  offset = store_w * 4 + kStoreC;

  if (store_w < kImgW) {
    char *ptr = reinterpret_cast<char *>(&smem_img[offset]);
    int const *src = &fetch_img[2];
    sts<32>(src, ptr);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kFltK,
         int kLdgWidth>
CUTLASS_DEVICE void wgrad_ldg_b_(
    int const kTidX,
    int const kP,
    int const kPEnd,
    int const kQBeg,
    int const kQEnd,
    int const kCtaK,
    int const kFltKPerCTA,
    uint16_t const *gmem_err,
    int *fetch_err) {

  using LongIndex = cutlass::layout::TensorNHWC::LongIndex;
  int const kThreadsPerErrPixel = kFltK / (kLdgWidth / 16);
  // The q coefficient loaded by that thread.
  int const kErrLoadQ = kQBeg + kTidX / kThreadsPerErrPixel;
  // The k coefficient loaded by that thread.
  int const kErrLoadK = kCtaK * kFltKPerCTA + (kTidX % kThreadsPerErrPixel) * (kLdgWidth / 16);
  // The offset in one row of global memory for the error tensor.
  LongIndex const kGmemErrOffset = kErrLoadQ * kFltK + kErrLoadK;
  char const *ptr = reinterpret_cast<char const *>(&gmem_err[kGmemErrOffset]);
  int *dst = fetch_err;
  if (kP < kPEnd && kErrLoadQ < kQEnd && kErrLoadK < kFltK) {
    ldg<kLdgWidth>(dst, ptr);
  }
}
template <int kFltK,
         int kFltR,
         int kFltS>
CUTLASS_DEVICE void wgrad_ldg_b(
    int const kTidX,
    int const kP,
    int const kPEnd,
    int const kQBeg,
    int const kQEnd,
    int const kCtaK,
    int const kFltKPerCTA,
    uint16_t const *gmem_err,
    int *fetch_err);

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specilizations for LDG.B
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void wgrad_ldg_b<16,3,3>(
    int const kTidX,
    int const kP,
    int const kPEnd,
    int const kQBeg,
    int const kQEnd,
    int const kCtaK,
    int const kFltKPerCTA,
    uint16_t const *gmem_err,
    int *fetch_err){

  // 1xLDG.64 + 1xLDG.32
  wgrad_ldg_b_<16, 64>(kTidX, kP, kPEnd, kQBeg, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err);

  int const kQBegNext = kQBeg + 3 * 32 / (16 / 4);
  int *fetch_err_next = &fetch_err[2];
  wgrad_ldg_b_<16, 32>(kTidX, kP, kPEnd, kQBegNext, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err_next);
}

template <>
CUTLASS_DEVICE void wgrad_ldg_b<16,5,5>(
    int const kTidX,
    int const kP,
    int const kPEnd,
    int const kQBeg,
    int const kQEnd,
    int const kCtaK,
    int const kFltKPerCTA,
    uint16_t const *gmem_err,
    int *fetch_err){

  // 1xLDG.64
  wgrad_ldg_b_<16, 64>(kTidX, kP, kPEnd, kQBeg, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err);
}

template <>
CUTLASS_DEVICE void wgrad_ldg_b<16,7,7>(
    int const kTidX,
    int const kP,
    int const kPEnd,
    int const kQBeg,
    int const kQEnd,
    int const kCtaK,
    int const kFltKPerCTA,
    uint16_t const *gmem_err,
    int *fetch_err){

  // 1xLDG.64
  wgrad_ldg_b_<16, 64>(kTidX, kP, kPEnd, kQBeg, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err);
}

template <>
CUTLASS_DEVICE void wgrad_ldg_b<32,3,3>(
    int const kTidX,
    int const kP,
    int const kPEnd,
    int const kQBeg,
    int const kQEnd,
    int const kCtaK,
    int const kFltKPerCTA,
    uint16_t const *gmem_err,
    int *fetch_err){

  // 1xLDG.128 + 1xLDG.64
  wgrad_ldg_b_<32, 128>(kTidX, kP, kPEnd, kQBeg, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err);

  int const kQBegNext = kQBeg + 3 * 32 / (32 / 8);
  int *fetch_err_next = &fetch_err[4];
  wgrad_ldg_b_<32, 64>(kTidX, kP, kPEnd, kQBegNext, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err_next);
}

template <>
CUTLASS_DEVICE void wgrad_ldg_b<32,5,5>(
    int const kTidX,
    int const kP,
    int const kPEnd,
    int const kQBeg,
    int const kQEnd,
    int const kCtaK,
    int const kFltKPerCTA,
    uint16_t const *gmem_err,
    int *fetch_err){

  // 1xLDG.128
  wgrad_ldg_b_<32, 128>(kTidX, kP, kPEnd, kQBeg, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err);
}

template <>
CUTLASS_DEVICE void wgrad_ldg_b<32,7,7>(
    int const kTidX,
    int const kP,
    int const kPEnd,
    int const kQBeg,
    int const kQEnd,
    int const kCtaK,
    int const kFltKPerCTA,
    uint16_t const *gmem_err,
    int *fetch_err){

  // 1xLDG.64 + 1xLDG.32
  wgrad_ldg_b_<32, 64>(kTidX, kP, kPEnd, kQBeg, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err);

  int const kQBegNext = kQBeg + 7 * 32 / (32 / 4);
  int *fetch_err_next = &fetch_err[2];
  wgrad_ldg_b_<32, 32>(kTidX, kP, kPEnd, kQBegNext, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err_next);
}

template <>
CUTLASS_DEVICE void wgrad_ldg_b<64,3,3>(
    int const kTidX,
    int const kP,
    int const kPEnd,
    int const kQBeg,
    int const kQEnd,
    int const kCtaK,
    int const kFltKPerCTA,
    uint16_t const *gmem_err,
    int *fetch_err){

  // 2xLDG.128 + 1xLDG.64 + 1xLDG.32
  wgrad_ldg_b_<64, 128>(kTidX, kP, kPEnd, kQBeg, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err);

  int q_beg_next = kQBeg + 3 * 32 / (64 / 8);
  int *fetch_err_next = &fetch_err[4];
  wgrad_ldg_b_<64, 128>(kTidX, kP, kPEnd, q_beg_next, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err_next);

  q_beg_next += 3 * 32 / (64 / 8);
  fetch_err_next = &fetch_err[8];
  wgrad_ldg_b_<64, 64>(kTidX, kP, kPEnd, q_beg_next, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err_next);

  q_beg_next += 3 * 32 / (64 / 4);
  fetch_err_next = &fetch_err[10];
  wgrad_ldg_b_<64, 32>(kTidX, kP, kPEnd, q_beg_next, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err_next);
}

template <>
CUTLASS_DEVICE void wgrad_ldg_b<64,5,5>(
    int const kTidX,
    int const kP,
    int const kPEnd,
    int const kQBeg,
    int const kQEnd,
    int const kCtaK,
    int const kFltKPerCTA,
    uint16_t const *gmem_err,
    int *fetch_err){

  // 1xLDG.128 + 1xLDG.64 + 1xLDG.32
  wgrad_ldg_b_<64, 128>(kTidX, kP, kPEnd, kQBeg, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err);

  int q_beg_next = kQBeg + 5 * 32 / (64 / 8);
  int *fetch_err_next = &fetch_err[4];
  wgrad_ldg_b_<64, 64>(kTidX, kP, kPEnd, q_beg_next, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err_next);

  q_beg_next += 5 * 32 / (64 / 4);
  fetch_err_next = &fetch_err[6];
  wgrad_ldg_b_<64, 32>(kTidX, kP, kPEnd, q_beg_next, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err_next);
}

template <>
CUTLASS_DEVICE void wgrad_ldg_b<64,7,7>(
    int const kTidX,
    int const kP,
    int const kPEnd,
    int const kQBeg,
    int const kQEnd,
    int const kCtaK,
    int const kFltKPerCTA,
    uint16_t const *gmem_err,
    int *fetch_err){

  // 1xLDG.128 + 1xLDG.32
  wgrad_ldg_b_<64, 128>(kTidX, kP, kPEnd, kQBeg, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err);

  int const kQBegNext = kQBeg + 7 * 32 / (64 / 8);
  int *fetch_err_next = &fetch_err[4];
  wgrad_ldg_b_<64, 32>(kTidX, kP, kPEnd, kQBegNext, kQEnd, kCtaK, kFltKPerCTA, gmem_err, fetch_err_next);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kFltK,
         int kStsWidth>
CUTLASS_DEVICE void wgrad_sts_b_(
    int const kTidX,
    int const kErrQ,
    int const kSmemErrFp16sPerSlice,
    int const kKPerSlice,
    int const kStorePixelBeg,
    uint16_t *smem_err,
    int const *fetch_err) {

  using LongIndex = cutlass::layout::TensorNHWC::LongIndex;
  int const kThreadsPerErrPixel = kFltK / (kStsWidth / 16);
  // Compute the slice of shared memory written by this thread.
  int const kSmemErrStoreSlice = kTidX % kThreadsPerErrPixel / (kKPerSlice / (kStsWidth / 16));
  int const kSmemErrStoreK = kTidX % kThreadsPerErrPixel % (kKPerSlice / (kStsWidth / 16)) * (kStsWidth / 16);
  // Determine the pixel.
  int const kSmemErrStorePixel = kStorePixelBeg + kTidX / kThreadsPerErrPixel;
  int const kOffset = kSmemErrStoreSlice * kSmemErrFp16sPerSlice + kSmemErrStorePixel * kKPerSlice + kSmemErrStoreK;

  // Store the data to shared memory.
  char *ptr = reinterpret_cast<char *>(&smem_err[kOffset]);
  int const *src = fetch_err;

  if (kSmemErrStorePixel < kErrQ) {
    sts<kStsWidth>(src, ptr);
  }
}

template <int kFltK,
         int kFltR,
         int kFltS>
CUTLASS_DEVICE void wgrad_sts_b(
    int const kTidX,
    int const kErrQ,
    int const kSmemErrFp16sPerSlice,
    int const kKPerSlice,
    int const kStorePixelBeg,
    uint16_t *smem_err,
    int const *fetch_err);

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specilizations for STS.B (matches LDG.B)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void wgrad_sts_b<16,3,3>(
    int const kTidX,
    int const kErrQ,
    int const kSmemErrFp16sPerSlice,
    int const kKPerSlice,
    int const kStorePixelBeg,
    uint16_t *smem_err,
    int const *fetch_err) {

  // 1xSTS.64 + 1xSTS.32
  wgrad_sts_b_<16,64>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, 0, smem_err, fetch_err);

  int kStorePixelNext = 3 * 32 / (16 / 4);
  int const *fetch_err_next = &fetch_err[2];
  wgrad_sts_b_<16,32>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, kStorePixelNext, smem_err, fetch_err_next);
}

template <>
CUTLASS_DEVICE void wgrad_sts_b<16,5,5>(
    int const kTidX,
    int const kErrQ,
    int const kSmemErrFp16sPerSlice,
    int const kKPerSlice,
    int const kStorePixelBeg,
    uint16_t *smem_err,
    int const *fetch_err) {

  // 1xSTS.64
  wgrad_sts_b_<16,64>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, 0, smem_err, fetch_err);
}

template <>
CUTLASS_DEVICE void wgrad_sts_b<16,7,7>(
    int const kTidX,
    int const kErrQ,
    int const kSmemErrFp16sPerSlice,
    int const kKPerSlice,
    int const kStorePixelBeg,
    uint16_t *smem_err,
    int const *fetch_err) {

  // 1xSTS.64
  wgrad_sts_b_<16,64>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, 0, smem_err, fetch_err);
}

template <>
CUTLASS_DEVICE void wgrad_sts_b<32,3,3>(
    int const kTidX,
    int const kErrQ,
    int const kSmemErrFp16sPerSlice,
    int const kKPerSlice,
    int const kStorePixelBeg,
    uint16_t *smem_err,
    int const *fetch_err) {

  // 1xSTS.128 + 1xSTS.64
  wgrad_sts_b_<32,128>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, 0, smem_err, fetch_err);

  int kStorePixelNext = 3 * 32 / (32 / 8);
  int const *fetch_err_next = &fetch_err[4];
  wgrad_sts_b_<32,64>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, kStorePixelNext, smem_err, fetch_err_next);
}

template <>
CUTLASS_DEVICE void wgrad_sts_b<32,5,5>(
    int const kTidX,
    int const kErrQ,
    int const kSmemErrFp16sPerSlice,
    int const kKPerSlice,
    int const kStorePixelBeg,
    uint16_t *smem_err,
    int const *fetch_err) {

  // 1xSTS.128
  wgrad_sts_b_<32,128>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, 0, smem_err, fetch_err);
}

template <>
CUTLASS_DEVICE void wgrad_sts_b<32,7,7>(
    int const kTidX,
    int const kErrQ,
    int const kSmemErrFp16sPerSlice,
    int const kKPerSlice,
    int const kStorePixelBeg,
    uint16_t *smem_err,
    int const *fetch_err) {

  // 1xSTS.64 + 1xSTS.32
  wgrad_sts_b_<32,64>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, 0, smem_err, fetch_err);

  int kStorePixelNext = 7 * 32 / (32 / 4);
  int const *fetch_err_next = &fetch_err[2];
  wgrad_sts_b_<32,32>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, kStorePixelNext, smem_err, fetch_err_next);
}

template <>
CUTLASS_DEVICE void wgrad_sts_b<64,3,3>(
    int const kTidX,
    int const kErrQ,
    int const kSmemErrFp16sPerSlice,
    int const kKPerSlice,
    int const kStorePixelBeg,
    uint16_t *smem_err,
    int const *fetch_err) {

  // 2xSTS.128 + 1xSTS.64 + 1xSTS.32
  wgrad_sts_b_<64,128>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, 0, smem_err, fetch_err);

  int store_pixel_next = 3 * 32 / (64 / 8);
  int const *fetch_err_next = &fetch_err[4];
  wgrad_sts_b_<64,128>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, store_pixel_next, smem_err, fetch_err_next);

  store_pixel_next += 3 * 32 / (64 / 8);
  fetch_err_next = &fetch_err[8];
  wgrad_sts_b_<64,64>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, store_pixel_next, smem_err, fetch_err_next);

  store_pixel_next += 3 * 32 / (64 / 4);
  fetch_err_next = &fetch_err[10];
  wgrad_sts_b_<64,32>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, store_pixel_next, smem_err, fetch_err_next);
}

template <>
CUTLASS_DEVICE void wgrad_sts_b<64,5,5>(
    int const kTidX,
    int const kErrQ,
    int const kSmemErrFp16sPerSlice,
    int const kKPerSlice,
    int const kStorePixelBeg,
    uint16_t *smem_err,
    int const *fetch_err) {

  // 1xSTS.128 + 1xSTS.64 + 1xSTS.32
  wgrad_sts_b_<64,128>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, 0, smem_err, fetch_err);

  int store_pixel_next = 5 * 32 / (64 / 8);
  int const *fetch_err_next = &fetch_err[4];
  wgrad_sts_b_<64,64>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, store_pixel_next, smem_err, fetch_err_next);

  store_pixel_next += 5 * 32 / (64 / 4);
  fetch_err_next = &fetch_err[6];
  wgrad_sts_b_<64,32>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, store_pixel_next, smem_err, fetch_err_next);
}

template <>
CUTLASS_DEVICE void wgrad_sts_b<64,7,7>(
    int const kTidX,
    int const kErrQ,
    int const kSmemErrFp16sPerSlice,
    int const kKPerSlice,
    int const kStorePixelBeg,
    uint16_t *smem_err,
    int const *fetch_err) {

  // 1xSTS.128 + 1xSTS.32
  wgrad_sts_b_<64,128>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, 0, smem_err, fetch_err);

  int kStorePixelNext = 7 * 32 / (64 / 8);
  int const *fetch_err_next = &fetch_err[4];
  wgrad_sts_b_<64,32>(kTidX, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, kStorePixelNext, smem_err, fetch_err_next);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int kFltR,
  int kFltS
>
struct LdsmA;

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specilizations for LDSM.A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct LdsmA<3, 3> {
  using LdsmOperand = Array<unsigned, 2>;

  CUTLASS_DEVICE
  void operator()(
    LdsmOperand *D_ptr,
    uint16_t const* smem_ptr,
    int const kSmemImgCol
  ) const {

    // 1xLDSM.2
    cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 2>(D_ptr[0], &smem_ptr[kSmemImgCol * 4]);
  }
};

template <>
struct LdsmA<5, 5> {
  using LdsmOperand = Array<unsigned, 3>;

  CUTLASS_DEVICE
  void operator()(
    LdsmOperand *D_ptr,
    uint16_t const* smem_ptr,
    int const kSmemImgCol
  ) const {

    // 1xLDSM.2 + 1xLDSM.1
    cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 2>(reinterpret_cast<Array<unsigned, 2> *>(D_ptr)[0], &smem_ptr[kSmemImgCol * 4]);
    cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 1>(reinterpret_cast<Array<unsigned, 1> *>(D_ptr)[2], &smem_ptr[(kSmemImgCol + 4) * 4]);
  }
};

template <>
struct LdsmA<7, 7> {
  using LdsmOperand = Array<unsigned, 4>;

  CUTLASS_DEVICE
  void operator()(
    LdsmOperand *D_ptr,
    uint16_t const* smem_ptr,
    int const kSmemImgCol
  ) const {

    // 1xLDSM.4
    cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 4>(*D_ptr, &smem_ptr[kSmemImgCol * 4]);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kFltK> struct LdsmB;

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specilizations for LDSM.B
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct LdsmB<16> {
  using LdsmOperand = Array<unsigned, 2>;

  CUTLASS_DEVICE
  void operator()(
    LdsmOperand *D_ptr,
    uint16_t const* smem_ptr,
    int const kSmemErrFp16sPerSlice
  ) const {

    // 1xLDSM.2
    cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 2>(D_ptr[0], smem_ptr);
  }
};

template <>
struct LdsmB<32> {
  using LdsmOperand = Array<unsigned, 4>;

  CUTLASS_DEVICE
  void operator()(
    LdsmOperand *D_ptr,
    uint16_t const* smem_ptr,
    int const kSmemErrFp16sPerSlice
  ) const {

    // 1xLDSM.4
    cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 4>(D_ptr[0], smem_ptr);
  }
};

template <>
struct LdsmB<64> {
  using LdsmOperand = Array<unsigned, 8>;

  CUTLASS_DEVICE
  void operator()(
    LdsmOperand *D_ptr,
    uint16_t const* smem_ptr,
    int const kSmemErrFp16sPerSlice
  ) const {

    // 2xLDSM.4
    cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 4>(reinterpret_cast<Array<unsigned, 4> *>(D_ptr)[0], smem_ptr);
    cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 4>(reinterpret_cast<Array<unsigned, 4> *>(D_ptr)[1], &smem_ptr[4 * kSmemErrFp16sPerSlice]);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // End namespace library
}  // End namespace cutlass_build_env
