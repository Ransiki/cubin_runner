#pragma once
#include <cstdio>

namespace tensorrt_llm::common {
class Logger {
public:
    static Logger& getLogger() { static Logger l; return l; }
    template<typename... Args> void log(Args&&...) {}
};
} // namespace tensorrt_llm::common

#define TLLM_LOG_WARNING(...)
#define TLLM_LOG_ERROR(...)
#define TLLM_LOG_DEBUG(...)
