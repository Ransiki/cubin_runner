#pragma once
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <cstdio>

// Dummy stream that swallows all << operator calls
struct _NullStream {
    template<typename T> _NullStream& operator<<(T const&) { return *this; }
};

// TVM_FFI_ICHECK: returns a stream-like object so `ICHECK(cond) << "msg"` compiles
struct _CheckHelper {
    bool ok;
    const char* file;
    int line;
    _CheckHelper(bool cond, const char* f, int l) : ok(cond), file(f), line(l) {}
    ~_CheckHelper() noexcept(false) {
        if (!ok) throw std::runtime_error(std::string("CHECK failed at ") + file + ":" + std::to_string(line));
    }
    template<typename T> _CheckHelper& operator<<(T const&) { return *this; }
};

#define TVM_FFI_ICHECK(cond) _CheckHelper((cond), __FILE__, __LINE__)
#define TVM_FFI_ICHECK_EQ(a, b) TVM_FFI_ICHECK((a) == (b))
#define TVM_FFI_ICHECK_NE(a, b) TVM_FFI_ICHECK((a) != (b))
#define TVM_FFI_ICHECK_LE(a, b) TVM_FFI_ICHECK((a) <= (b))
#define TVM_FFI_ICHECK_LT(a, b) TVM_FFI_ICHECK((a) < (b))
#define TVM_FFI_ICHECK_GE(a, b) TVM_FFI_ICHECK((a) >= (b))
#define TVM_FFI_ICHECK_GT(a, b) TVM_FFI_ICHECK((a) > (b))

#define TVM_FFI_LOG_AND_THROW(type) \
    throw std::runtime_error("TVM_FFI error"); _NullStream()
