#pragma once
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <cstdio>

#define FLASHINFER_CHECK(cond, ...) \
    do { if (!(cond)) { throw std::runtime_error("FLASHINFER_CHECK failed"); } } while(0)

#define FLASHINFER_WARN(...) do {} while(0)
