#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <chrono>

namespace rtp_llm {

// ----------------------------- MemoryAsyncMatchContext ---------------------------------

void MemoryAsyncMatchContext::waitDone() {
    return;
}

bool MemoryAsyncMatchContext::done() const {
    return true;
}

bool MemoryAsyncMatchContext::success() const {
    return true;
}

size_t MemoryAsyncMatchContext::matchedBlockCount() const {
    return matched_block_count_;
}

// ----------------------------- MemoryAsyncContext ---------------------------------

bool MemoryAsyncContext::done() const {
    return already_done_.load();
}

bool MemoryAsyncContext::success() const {
    if (!broadcast_result_ || !broadcast_result_->success()) {
        return false;
    }
    const auto& responses = broadcast_result_->responses();
    for (const auto& response : responses) {
        if (!response.has_mem_response() || !response.mem_response().success()) {
            return false;
        }
    }
    return true;
}

void MemoryAsyncContext::waitDone() {
    if (done()) {
        return;
    }
    if (broadcast_result_) {
        auto t0 = std::chrono::steady_clock::now();
        int  warn_count = 0;
        while (!broadcast_result_->done()) {
            if (!broadcast_result_->waitDone(10000)) {
                ++warn_count;
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t0).count();
                RTP_LLM_LOG_WARNING("[MemoryAsyncContext::waitDone] broadcast STILL WAITING elapsed=%lld ms",
                                    (long long)elapsed_ms);
            }
        }
    }
    if (done_callback_) {
        done_callback_(success());
    }
    already_done_.store(true);
}

void MemoryAsyncContext::setBroadcastResult(
    const std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>& result) {
    broadcast_result_ = result;
}

}  // namespace rtp_llm