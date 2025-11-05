#pragma once

#include "kernels/api.cuh"
#include "kernels/exception.cuh"

namespace deep_ep {

template <typename dtype_t>
dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

template <typename dtype_t>
dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;
}

struct Config {
    int num_sms;
    int num_max_nvl_chunked_send_tokens;
    int num_max_nvl_chunked_recv_tokens;
    int num_max_rdma_chunked_send_tokens;
    int num_max_rdma_chunked_recv_tokens;

    Config(int num_sms,
           int num_max_nvl_chunked_send_tokens,
           int num_max_nvl_chunked_recv_tokens,
           int num_max_rdma_chunked_send_tokens,
           int num_max_rdma_chunked_recv_tokens)
        : num_sms(num_sms),
          num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
          num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
          num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
          num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {
        EP_HOST_ASSERT(num_sms >= 0);
        EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens > 0 and num_max_nvl_chunked_recv_tokens > 0);
        EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens);
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens > 0 and num_max_rdma_chunked_recv_tokens > 0);

        // Ceil up RDMA buffer size
        this->num_max_rdma_chunked_recv_tokens = align_up<int>(num_max_rdma_chunked_recv_tokens, num_max_rdma_chunked_send_tokens);
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens < num_max_rdma_chunked_recv_tokens);
        // NOTES: this assertion is related to RDMA lazy head update, we must ensure senders always have space to push
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens <= num_max_rdma_chunked_recv_tokens / 2);
    }

    size_t get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
        // Below are some assumptions
        // TODO: add assertions
        constexpr int kNumMaxTopK = 128;
        constexpr int kNumMaxScales = 128;
        EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(num_ranks <= NUM_MAX_NVL_PEERS or num_sms % 2 == 0);
        const auto num_rdma_ranks = std::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
        const auto num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
        const int num_channels = num_sms / 2;

        size_t num_bytes = 0;
        num_bytes += num_channels * num_nvl_ranks * (2 * num_rdma_ranks + 3) * sizeof(int);
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * hidden_bytes;
#ifndef DISABLE_NVSHMEM
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * internode::get_source_meta_bytes();
#endif
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(topk_idx_t);
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(float);
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxScales * sizeof(float);
        num_bytes = ((num_bytes + 127) / 128) * 128;
        return num_bytes;
    }

    size_t get_rdma_buffer_size_hint(int64_t hidden_bytes, int num_ranks) const {
#ifndef DISABLE_NVSHMEM
        // Legacy mode
        if (num_ranks <= NUM_MAX_NVL_PEERS)
            return 0;

        // Below are some assumptions
        // TODO: add assertions
        constexpr int kNumMaxTopK = 128;
        constexpr int kNumMaxScales = 128;
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(num_sms % 2 == 0);
        const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
        const int num_channels = num_sms / 2;

        size_t num_bytes = 0;
        num_bytes += num_channels * num_rdma_ranks * (NUM_MAX_NVL_PEERS * 2 + 2) * 2 * sizeof(int);
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * hidden_bytes * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * internode::get_source_meta_bytes() * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(topk_idx_t) * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(float) * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxScales * sizeof(float) * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * sizeof(int4) * 2;
        num_bytes = ((num_bytes + 127) / 128) * 128;
        return num_bytes;
#else
        EP_HOST_ASSERT(false and "NVSHMEM is disable during compilation");
#endif
    }
};

struct LowLatencyBuffer {
    int num_clean_int = 0;

    void* dispatch_rdma_send_buffer = nullptr;
    void* dispatch_rdma_recv_data_buffer = nullptr;
    int* dispatch_rdma_recv_count_buffer = nullptr;//信号指针

    void* combine_rdma_send_buffer = nullptr;
    void* combine_rdma_recv_data_buffer = nullptr;
    int* combine_rdma_recv_flag_buffer = nullptr;//信号指针

    void* combine_rdma_send_buffer_data_start = nullptr;
    size_t num_bytes_per_combine_msg = 0;

    std::pair<int*, int> clean_meta() {
        EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer == combine_rdma_recv_flag_buffer);
        return {dispatch_rdma_recv_count_buffer, num_clean_int};
    }
};

struct LowLatencyLayout {
    //这个totoal_byte 就是 RDMA buffer的总大小(所有预申请的缓冲区总和)
    size_t total_bytes = 0;
    //shk  为什么是两个buffer? 一个buffer里面已经有dispatch和combine的send buffer和recv buffer了
    // 用了两套缓冲区, 难道是一套缓冲区在运行的时候,另外一套在准备数据
    // 还是双向的,以防发送接收, 对面也有一套发送接收
    LowLatencyBuffer buffers[2];

    //模板函数，用于计算指针偏移量
    template <typename out_ptr_t = void*, typename count_ptr_t = uint8_t*, typename in_ptr_t = void*>
    out_ptr_t advance(const in_ptr_t& ptr, size_t count) {
        return reinterpret_cast<out_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) + count);
    }

    //构造函数，用于初始化LowLatencyLayout对象
    LowLatencyLayout(void* rdma_buffer, int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
        //缩放因子的数量
        const int num_scales = hidden / 128;

        /**
         * @brief 计算并分配缓冲区内存布局
         * 
         * 该函数负责:
         * 1. 计算dispatch和combine操作所需的缓冲区大小
         * 2. 分配对称的奇偶发送/接收/信号缓冲区
         * 3. 确保内存对齐要求
         * 
         * 缓冲区布局说明:
         * - 2个对称的奇偶发送缓冲区
         * - 2个对称的奇偶接收缓冲区
         * - 2个对称的奇偶信号缓冲区
         * 
         * @note 如需进行数据转换，应为combine消息添加控制int4
         * @note num_scales * sizeof(nv_bfloat162)表示每128通道的min/max值
         * 
         * @param num_scales 缩放因子数量
         * @param hidden 隐藏层大小
         * @param num_max_dispatch_tokens_per_rank 每个rank的最大dispatch token数
         * @param num_experts 专家数量
         */
        EP_HOST_ASSERT(num_scales * sizeof(float) <= hidden);
        //dispatch阶段每个消息的大小: 一个int4 数据头 + 最大数据体(依据编码类型)
        //为什么不根据用哪种模式,来选择,而是用max?
        //因为动态复用? 分配内存是前期做的,但是后期会不同的模式?
        // FP8 一字节,所以就是 hidden + 缩放因子 * 缩放因子的类型
        size_t num_bytes_per_dispatch_msg = sizeof(int4) + std::max(hidden * sizeof(nv_bfloat16), hidden + num_scales * sizeof(float));
        //combine固定接收数据类型是nv_bfloat162 和 nv_bfloat16, 接收hidden以及对应的缩放因子
        size_t num_bytes_per_combine_msg = num_scales * sizeof(nv_bfloat162) + hidden * sizeof(nv_bfloat16);


        // 下面有三套缓冲区计算:  发送buffer  接受buffer 信号buffer
        // Send buffer
        size_t dispatch_send_buffer_bytes = num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
        size_t combine_send_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;
        size_t send_buffer_bytes = std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);
        EP_HOST_ASSERT(send_buffer_bytes % sizeof(int4) == 0);
        total_bytes += send_buffer_bytes * 2;

        // Symmetric receive buffers
        // TODO: optimize memory usages
        size_t dispatch_recv_data_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
        size_t combine_recv_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;
        size_t recv_buffer_bytes = std::max(dispatch_recv_data_buffer_bytes, combine_recv_buffer_bytes);
        EP_HOST_ASSERT(recv_buffer_bytes % sizeof(int4) == 0);
        total_bytes += recv_buffer_bytes * 2;

        // Symmetric signaling buffers
        size_t dispatch_recv_count_buffer_bytes = num_experts * sizeof(int);
        size_t combine_recv_flag_buffer_bytes = dispatch_recv_count_buffer_bytes;
        size_t signaling_buffer_bytes = std::max(dispatch_recv_count_buffer_bytes, combine_recv_flag_buffer_bytes);
        size_t signaling_buffer_bytes_aligned = align_up<size_t>(signaling_buffer_bytes, 128);
        total_bytes += signaling_buffer_bytes_aligned * 2;

        // Assign pointers
        // NOTES: we still leave some space for distinguishing dispatch/combine buffer,
        // so you may see some parameters are duplicated
        //给buffer[0]和[1] 里面的元素赋值
        for (int i = 0; i< 2; ++i) {
            buffers[i] = {static_cast<int>(signaling_buffer_bytes / sizeof(int)),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2 + recv_buffer_bytes * i),
                          advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2 + recv_buffer_bytes * i),
                          advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
                          num_bytes_per_combine_msg};
        }
    }
};

size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
    auto num_bytes = LowLatencyLayout(nullptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts).total_bytes;
    // 这里是为了cuda编程中, 128 bytes对齐
    return ((num_bytes + NUM_BUFFER_ALIGNMENT_BYTES) / NUM_BUFFER_ALIGNMENT_BYTES) * NUM_BUFFER_ALIGNMENT_BYTES;
}

}  // namespace deep_ep
