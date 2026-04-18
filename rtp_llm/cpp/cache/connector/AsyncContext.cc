#include "rtp_llm/cpp/cache/connector/AsyncContext.h"

#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <chrono>

namespace rtp_llm {

// --------------------------------- FusedAsyncContext ---------------------------------

FusedAsyncContext::FusedAsyncContext(const std::vector<std::shared_ptr<AsyncContext>>& contexts): contexts_(contexts) {}

void FusedAsyncContext::waitDone() {
    RTP_LLM_PROFILE_FUNCTION();
    for (size_t i = 0; i < contexts_.size(); i++) {
        if (contexts_[i]) {
            RTP_LLM_PROFILE_SCOPE_DYNAMIC("wait_sub_context[%zu]", i);
            contexts_[i]->waitDone();
        }
    }
}

bool FusedAsyncContext::done() const {
    for (const auto& context : contexts_) {
        if (context && !context->done()) {
            return false;
        }
    }
    return true;
}

bool FusedAsyncContext::success() const {
    for (const auto& context : contexts_) {
        if (context && !context->success()) {
            return false;
        }
    }
    return true;
}

// --------------------------------- FusedAsyncReadContext ---------------------------------

FusedAsyncReadContext::FusedAsyncReadContext(const std::shared_ptr<FusedAsyncContext>& fused_match_context,
                                             const std::shared_ptr<KVCacheResource>&   resource,
                                             const std::shared_ptr<Meta>&              meta):
    fused_match_context_(fused_match_context), resource_(resource), meta_(meta) {}

void FusedAsyncReadContext::waitDone() {
    RTP_LLM_PROFILE_FUNCTION();
    const auto  trace_id = meta_ ? meta_->trace_id() : std::string("N/A");
    const auto  t0       = std::chrono::steady_clock::now();
    int         warn_count = 0;
    std::unique_lock<std::mutex> lock(done_mutex_);
    while (!done()) {
        if (done_cv_.wait_for(lock, std::chrono::seconds(10)) == std::cv_status::timeout) {
            ++warn_count;
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            bool match_done = fused_match_context_ ? fused_match_context_->done() : true;
            bool match_ok   = fused_match_context_ ? fused_match_context_->success() : true;
            bool read_set   = read_ctx_set_.load();
            bool read_done  = false;
            {
                std::lock_guard<std::mutex> lk(read_ctx_mutex_);
                read_done = !fused_read_context_ || fused_read_context_->done();
            }
            RTP_LLM_LOG_WARNING("[FusedAsyncReadContext::waitDone] WAITING trace_id=%s elapsed=%lld ms "
                                "match_done=%d match_ok=%d read_ctx_set=%d read_done=%d",
                                trace_id.c_str(), (long long)elapsed_ms,
                                match_done, match_ok, read_set, read_done);
        }
    }
}

void FusedAsyncReadContext::notifyDone() {
    std::lock_guard<std::mutex> lock(done_mutex_);
    done_cv_.notify_all();
}

bool FusedAsyncReadContext::done() const {
    if (!fused_match_context_) {
        return true;
    }
    if (!fused_match_context_->done()) {
        return false;
    }
    if (!fused_match_context_->success()) {
        return true;
    }
    std::lock_guard<std::mutex> lock(read_ctx_mutex_);
    if (!read_ctx_set_.load()) {
        return false;
    }
    return !fused_read_context_ || fused_read_context_->done();
}

bool FusedAsyncReadContext::success() const {
    if (done() && (fused_match_context_ && fused_match_context_->success())) {
        std::lock_guard<std::mutex> lk(read_ctx_mutex_);
        return !fused_read_context_ || fused_read_context_->success();
    }
    return false;
}

void FusedAsyncReadContext::setFusedReadContext(const std::shared_ptr<FusedAsyncContext>& fused_read_context) {
    std::lock_guard<std::mutex> lk(read_ctx_mutex_);
    fused_read_context_ = fused_read_context;
    read_ctx_set_.store(true);
}

const std::shared_ptr<FusedAsyncContext> FusedAsyncReadContext::fusedReadContext() const {
    std::lock_guard<std::mutex> lk(read_ctx_mutex_);
    return fused_read_context_;
}

const std::shared_ptr<FusedAsyncContext>& FusedAsyncReadContext::fusedMatchContext() const {
    return fused_match_context_;
}

const std::shared_ptr<KVCacheResource>& FusedAsyncReadContext::resource() const {
    return resource_;
}

const std::shared_ptr<Meta>& FusedAsyncReadContext::meta() const {
    return meta_;
}

}  // namespace rtp_llm