#ifndef _KERNEL_UTILS_H_
#define _KERNEL_UTILS_H_

#include <cuda_fp16.h>
#include <cassert>
#include <cfloat>
#include <cstdint>

template <typename TYPE>
__device__ __forceinline__ TYPE relu(TYPE in)
{
    return in < TYPE(0) ? TYPE(0) : in;
}
template <typename T>
__device__ __forceinline__ float toFloat(T x)
{
    return float(x);
}

// Declared, but not defined, since a general definition might round in an undesired way.
// Instead, we use explicit specializations to get the semantics we want for each T.
template <typename T>
__device__ T fromFloat(float x);
template <typename T>
__device__ T fromHalf(half x);
template <typename T>
__device__ half toHalf(T x)
{
    return x;
}

// Round float to an int8.
//
// Special cases:
//      x >  127 -> 127
//      x < -128 -> -128
//      x is NaN -> -128
// The behavior for NaN matches the unit test framework.

template <typename T>
__device__ __forceinline__ T fromFloatReLU(float x, float relu)
{
    return fromFloat<T>(fmaxf(x, relu));
}
template <>
__device__ __forceinline__ int8_t fromFloatReLU<int8_t>(float x, float relu)
{
    // The order of the next two statements matters when x is a NaN,
    // because IEEE max/min return the non-NaN operand when one operand
    // is a NaN and the other is not.
    x = fmaxf(x, fmaxf(INT8_MIN, relu));
    x = fminf(x, INT8_MAX);
    return __float2int_rn(x);
}
template <>
__device__ __forceinline__ int8_t fromFloat<int8_t>(float x)
{
    // The order of the next two statements matters when x is a NaN,
    // because IEEE max/min return the non-NaN operand when one operand
    // is a NaN and the other is not.
    x = fmaxf(x, INT8_MIN);
    x = fminf(x, INT8_MAX);
    return __float2int_rn(x);
}

template <>
__device__ __forceinline__ float fromFloat<float>(float x)
{
    return x;
}

template <>
__device__ __forceinline__ half fromFloat<half>(float x)
{
    return __float2half(x);
}
template <>
__device__ __forceinline__ half fromHalf<half>(const half x)
{
    return x;
}

template <>
__device__ __forceinline__ half toHalf<float>(const float x)
{
    return fromFloat<half>(x);
}
template <>
__device__ __forceinline__ float toFloat<half>(half x)
{
    return __half2float(x);
}
template <>
__device__ __forceinline__ float fromHalf<float>(const half x)
{
    return toFloat(x);
}
template <typename TYPE>
__device__ __forceinline__ half2 toHalf2(const TYPE& a, const TYPE& b)
{
    return __floats2half2_rn(toFloat(a), toFloat(b));
}

template <>
__device__ __forceinline__ half2 toHalf2<float>(const float& a, const float& b)
{
    return __floats2half2_rn(a, b);
}

template <>
__device__ __forceinline__ half2 toHalf2<half>(const half& a, const half& b)
{
    return __halves2half2(a, b);
}

template <typename TYPE>
__device__ __forceinline__ void fromHalf2(const half2& h2, TYPE& a, TYPE& b)
{
    float2 f2 = __half22float2(h2);
    a = fromFloat<TYPE>(f2.x);
    b = fromFloat<TYPE>(f2.y);
}

template <>
__device__ __forceinline__ void fromHalf2<float>(const half2& h2, float& a, float& b)
{
    float2 f2 = __half22float2(h2);
    a = f2.x;
    b = f2.y;
}

template <>
__device__ __forceinline__ void fromHalf2<half>(const half2& h2, half& a, half& b)
{
    a = __low2half(h2);
    b = __high2half(h2);
}

template <>
__device__ __forceinline__ half relu<half>(half in)
{
    return fromFloat<half>(relu(toFloat(in)));
}

template <typename TYPE>
__device__ __forceinline__ TYPE sigmoid(TYPE in)
{
    return TYPE(1.0f) / (TYPE(1.0f) + TYPE(__expf(-in)));
}
template <>
__device__ __forceinline__ half sigmoid<half>(half in)
{
    return fromFloat<half>(sigmoid(toFloat(in)));
}

template <typename TYPE>
__device__ __forceinline__ TYPE tanh_(TYPE in)
{
    return tanh(in);
}
template <>
__device__ __forceinline__ half tanh_<half>(half in)
{
    return fromFloat<half>(tanh(toFloat(in)));
}


// Integer division rounding up
inline __host__ __device__ int divUp(int x, int n)
{
    return (x + n - 1) / n;
}

inline __host__ __device__ int roundUp(int m, int n) { return divUp(m, n) * n; }
// Exact integer division
inline __host__ int divExact(int m, int n)
{
    assert(m % n == 0);
    return m / n;
}

template <typename To, typename From>
__device__ __forceinline__ To convertTo(From);

template <>
__device__ __forceinline__ half convertTo(float x)
{
    return __float2half(x);
}

template <>
__device__ __forceinline__ half convertTo(half x)
{
    return x;
}

template <>
__device__ __forceinline__ float convertTo(float x)
{
    return x;
}

template <>
__device__ __forceinline__ float convertTo(half x)
{
    return __half2float(x);
}

template <typename To, typename From>
__device__ __forceinline__ To convertTo(From a, From b);

template <>
__device__ __forceinline__ half2 convertTo(float a, float b)
{
    return __floats2half2_rn(a, b);
}

template <>
__device__ __forceinline__ half2 convertTo(half a, half b)
{
    return __halves2half2(a, b);
}

#endif // _KERNEL_UTILS_H_
