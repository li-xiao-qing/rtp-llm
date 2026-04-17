#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/cuda/SplitKvCacheCopy.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#include <thread>

namespace rtp_llm {

namespace {

at::cuda::CUDAStream& getNoBlockCopyStream() {
    static thread_local auto stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
    return stream;
}

}  // namespace

void execNoBlockCopy(const MultiCopyParams& params) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

    int copy_device = -1;
    if (!params.multi_dst.empty()) {
        if (params.multi_dst[0].is_cuda()) {
            copy_device = static_cast<int>(params.multi_dst[0].get_device());
        } else if (params.multi_src[0].is_cuda()) {
            copy_device = static_cast<int>(params.multi_src[0].get_device());
        }
        if (copy_device >= 0) {
            check_cuda_value(cudaSetDevice(copy_device));
        }
    }

    auto stream = getNoBlockCopyStream().stream();

    if (params.split_kv_layer_num > 0 && copy_device >= 0) {
        if (splitKvMultiCopy(params.multi_src,
                             params.multi_dst,
                             params.split_kv_layer_num,
                             static_cast<int64_t>(params.split_kv_cache_stride_bytes),
                             static_cast<int64_t>(params.split_kv_scale_stride_bytes),
                             stream)) {
            check_cuda_value(cudaStreamSynchronize(stream));
            check_cuda_error();
            return;
        }
    }

    size_t total_bytes = 0;
    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        total_bytes += params.multi_src[i].nbytes();
    }

    RTP_LLM_LOG_INFO("[NoBlockCopy] begin: device=%d, num_tensors=%zu, total_bytes=%zu, stream=%p",
                     copy_device, params.multi_src.size(), total_bytes, (void*)stream);

    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        auto ret = cudaMemcpyAsync(params.multi_dst[i].data_ptr(),
                                   params.multi_src[i].data_ptr(),
                                   params.multi_src[i].nbytes(),
                                   cudaMemcpyDefault,
                                   stream);
        if (ret != cudaSuccess) {
            RTP_LLM_LOG_WARNING("[NoBlockCopy] cudaMemcpyAsync[%zu] FAILED: %s", i, cudaGetErrorString(ret));
            check_cuda_value(ret);
        }
    }

    RTP_LLM_LOG_INFO("[NoBlockCopy] all cudaMemcpyAsync submitted, starting cudaStreamSynchronize");

    auto t0 = std::chrono::steady_clock::now();
    constexpr int kPollIntervalMs = 5000;
    int poll_count = 0;
    while (true) {
        auto query_ret = cudaStreamQuery(stream);
        if (query_ret == cudaSuccess) {
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            RTP_LLM_LOG_INFO("[NoBlockCopy] stream done after %lld ms (poll_count=%d)", (long long)elapsed_ms, poll_count);
            break;
        }
        if (query_ret != cudaErrorNotReady) {
            RTP_LLM_LOG_WARNING("[NoBlockCopy] cudaStreamQuery returned error: %s", cudaGetErrorString(query_ret));
            check_cuda_value(query_ret);
            break;
        }
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (elapsed_ms > (poll_count + 1) * kPollIntervalMs) {
            poll_count++;
            RTP_LLM_LOG_WARNING("[NoBlockCopy] STILL WAITING: device=%d, elapsed=%lld ms, stream=%p, total_bytes=%zu",
                                copy_device, (long long)elapsed_ms, (void*)stream, total_bytes);
        }
        std::this_thread::yield();
    }
    check_cuda_error();
}

void warmupNoBlockCopy() {
    if (!warmupSplitKvCopyKernels(at::cuda::getCurrentCUDAStream().stream())) {
        RTP_LLM_LOG_WARNING("warmupSplitKvCopyKernels failed; split-KV copy may JIT on first use");
    }
}

}  // namespace rtp_llm
