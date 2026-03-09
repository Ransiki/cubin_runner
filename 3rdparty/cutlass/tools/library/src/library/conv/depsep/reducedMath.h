#ifndef _REDUCED_MATH_H
#define _REDUCED_MATH_H
//Dynamically strength-reduced div and mod
//
//Ideas taken from Sean Baxter's MGPU library.
//These classes provide for reduced complexity division and modulus
//on integers, for the case where the same divisor or modulus will
//be used repeatedly.  

namespace cutlassRM {
namespace rt {

namespace detail {

// Count leading zeros - start from most significant bit.
int clz(int x) {
    for(int i = 31; i >= 0; --i)
        if((1<< i) & x) return 31 - i;
    return 32;
}

#define CUDNN_IS_POW_2(x) (0 == ((x) & ((x) - 1)))

int find_log_2(int x, bool round_up = false) {
    int a = 31 - clz(x);
    if (round_up) a += !CUDNN_IS_POW_2(x);
    return a;
}

void find_divisor(int denom,
                  unsigned int& mul_coeff, unsigned int& shift_coeff) {
    if (denom == 0) {
        return;       
    }
    if (denom == 1) {
        // if dividing by 1, reduced math doesn't work because mul_coeff would
        // need to be 2^32, which doesn't fit into unsigned int.  the div()
        // routine handles this special case separately.
        mul_coeff = 0;
        shift_coeff = 0;
        return;
    }
    // To express the division N/D in terms of a multiplication, what we first
    // imagine is simply N*(1/D).  However, 1/D will always evaluate to 0 (for D>1),
    // so we need another way.  There's nothing that says we have to use exactly
    // the fraction 1/D; instead it could be any X/Y that reduces to 1/D (i.e.,
    // Y=X*D), or at least to "close enough" to it.  If we pick Y that is a power
    // of two, then the N*(X/Y) can be N*X followed by a right-shift by some amount.
    // The power of two we should pick should be at least 2^32, because in the
    // div() routine we'll use umulhi(), which returns only the upper 32 bits --
    // this being equivalent to a right-shift by 32.  But we might want a higher
    // power of two for better accuracy depending on the magnitude of the denominator.
    // Once we've picked Y, then X [our mul_coeff value] is simply Y/D, rounding up,
    // and we save shift_coeff as whatever further shift we have to do beyond
    // what the umulhi() implies.
    unsigned int p = 31 + find_log_2(denom, true);
    unsigned int m = (unsigned int)(((1ull << p) + (unsigned int)denom - 1)/(unsigned int)denom);
    mul_coeff = m;
    shift_coeff = p - 32;
}

__host__ __device__ __forceinline__
unsigned int umulhi(unsigned int x, unsigned int y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 100
    return __umulhi(x, y);
#else
    unsigned long long z = (unsigned long long)x * (unsigned long long)y;
    return (unsigned int)(z >> 32);
#endif  
}

// This is a weird implementation that returns div_up(0,1)=0 but 
// div_up(0,2)=1 (wrong) -- just do not use it with a=0.
__host__ __device__
inline int div_up(int a, int b) {
    return (a - 1) / b + 1;
}

} //end namespace detail

class reduced_divisor {
  public:
    reduced_divisor() {}
    __host__ __forceinline__
    reduced_divisor(int _y) : y(_y) {
        detail::find_divisor(y, mul_coeff, shift_coeff);
    }
    __host__ __device__ __forceinline__
    reduced_divisor(unsigned _mul_coeff, unsigned _shift_coeff, int _y) 
        : mul_coeff(_mul_coeff), shift_coeff(_shift_coeff), y(_y) {
    }
    __host__ __device__ __forceinline__
    int div(int x) const {
        // if dividing by 1, then find_divisor wouldn't have worked because
        // mul_coeff would have had to be 2^32, which can't be represented,
        // so we have to special case that one. 
        return (y!=1) ? detail::umulhi((unsigned int)x, mul_coeff) >> shift_coeff : x;
    }
    __host__ __device__ __forceinline__
    int mod(int x) const {
        return x - (div(x) * y);
    }
    __host__ __device__ __forceinline__
    void divmod(int x, int& q, int& mod) const {
        q = div(x);
        mod = x - (q * y);
    }   
    __host__ __device__ __forceinline__
    int get() const {
        return y;
    }
    inline __host__ 
    void get_mul_shift(unsigned &mul, unsigned &shift) {
        mul = mul_coeff;
        shift = shift_coeff;
    }
  protected:
    unsigned int mul_coeff;
    unsigned int shift_coeff;
    int y;
};

} //end namespace rt
}
#endif
