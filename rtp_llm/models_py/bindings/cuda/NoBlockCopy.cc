#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/cuda/SplitKvCacheCopy.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>

namespace rtp_llm {

namespace {

at::cuda::CUDAStream& getNoBlockCopyStream() {
    static thread_local auto stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
    return stream;
}

inline int64_t steadyNowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
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

    int actual_device = -1;
    cudaGetDevice(&actual_device);
    RTP_LLM_LOG_INFO("LXQ|execNoBlockCopy: copy_device=%d, actual_device=%d, stream=%p",
                      copy_device, actual_device, (void*)stream);

    if (params.split_kv_layer_num > 0 && copy_device >= 0) {
        const auto t0 = steadyNowUs();
        if (splitKvMultiCopy(params.multi_src,
                             params.multi_dst,
                             params.split_kv_layer_num,
                             static_cast<int64_t>(params.split_kv_cache_stride_bytes),
                             static_cast<int64_t>(params.split_kv_scale_stride_bytes),
                             stream)) {
            const auto t1 = steadyNowUs();
            check_cuda_value(cudaStreamSynchronize(stream));
            const auto t2 = steadyNowUs();
            check_cuda_error();
            RTP_LLM_LOG_INFO("LXQ|execNoBlockCopy[splitKv]: pairs=%zu, layers=%d, device=%d, "
                              "launch_us=%ld, sync_us=%ld, thread=%lu",
                              params.multi_src.size(), params.split_kv_layer_num, copy_device,
                              t1 - t0, t2 - t1, pthread_self());
            return;
        }
    }

    const auto t0 = steadyNowUs();
    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        check_cuda_value(cudaMemcpyAsync(params.multi_dst[i].data_ptr(),
                                         params.multi_src[i].data_ptr(),
                                         params.multi_src[i].nbytes(),
                                         cudaMemcpyDefault,
                                         stream));
    }
    const auto t1 = steadyNowUs();
    check_cuda_value(cudaStreamSynchronize(stream));
    const auto t2 = steadyNowUs();
    check_cuda_error();
    RTP_LLM_LOG_INFO("LXQ|execNoBlockCopy[memcpy]: pairs=%zu, device=%d, "
                      "submit_us=%ld, sync_us=%ld, thread=%lu",
                      params.multi_src.size(), copy_device,
                      t1 - t0, t2 - t1, pthread_self());
}

void warmupNoBlockCopy() {
    if (!warmupSplitKvCopyKernels(at::cuda::getCurrentCUDAStream().stream())) {
        RTP_LLM_LOG_WARNING("warmupSplitKvCopyKernels failed; split-KV copy may JIT on first use");
    }
}

}  // namespace rtp_llm
