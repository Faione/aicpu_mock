// Version-specific adaptation of production ParamsInit and CheckNeedInitOutput.
#pragma once
#include "metadata_core.h"
#include <limits>
#include "attention/common/op_kernel/load_balance/section_stream_k/section_stream_k.h"
#include "attention/quant_flash_attn/op_host/qfa_adjust_sinner_souter.h"
#include "attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata.h"
namespace qfa_mock {
inline std::vector<int64_t> Cumulative(const std::vector<int64_t>& lengths) {
    auto result = lengths;
    for (size_t i = 1; i < result.size(); ++i) result[i] += result[i - 1];
    return result;
}
inline void Configure(const Target& target, const Request& request,
    const std::vector<int64_t>& qLens, const std::vector<int64_t>& kvLens,
    load_balance::DeviceInfo& device, load_balance::BaseInfo& base,
    load_balance::SectionStreamKParam& param) {
    const auto arch92 = target.arch92;
    const auto aic = target.aic, aiv = target.aiv;
    const auto batch = request.batch, nq = request.nq, nk = request.nk;
    const auto dim = request.dim, dimV = request.dimV, mask = request.mask;
    const auto maxQ = request.maxQ, maxKv = request.maxKv;
    const auto paged = request.paged;
    const auto& cuQ = request.cuQ;
    const auto& cuKv = request.cuKv;
    device.aicCoreMaxNum = device.aicCoreMinNum = aic;
    device.aivCoreMaxNum = device.aivCoreMinNum = aiv;
    base.batchSize = batch;
    base.queryHeadNum = nq;
    base.kvHeadNum = nk;
    base.querySeqSize = maxQ;
    base.kvSeqSize = maxKv;
    base.isCumulativeQuerySeq = true;
    base.isCumulativeKvSeq = !paged;
    base.actualQuerySeqSize = Cumulative(qLens);
    base.actualKvSeqSize = paged ? kvLens : Cumulative(kvLens);
    base.headDimQk = dim;
    // Current production ParamsInit uses headDim_ here (v1 restricts Dv == D).
    base.headDimV = dim;
    base.attenMaskFlag = mask != 0;
    base.sparseMode = mask;
    base.preToken = base.nextToken = std::numeric_limits<uint32_t>::max();
    base.layoutQuery = load_balance::Layout::TND;
    base.layoutKv = paged ? load_balance::Layout::PA_BNBD : load_balance::Layout::TND;
    base.queryType = base.kvType = load_balance::DataType::FP8_E4M3FN;

    namespace tile = optiling::quant_flash_attn::qfa_tiling_util;
    uint32_t outer = 0, inner = 0;
    if (arch92) {
        tile::AdjustSinnerAndSouterArch92(dimV, maxQ, maxKv, mask, -1, -1,
                                        tile::LAYOUT_BSND, 1, false, outer, inner);
    } else {
        tile::AdjustSinnerAndSouter(dim, maxQ, maxKv, mask, -1, -1,
                                  tile::LAYOUT_BSND, 1, outer, inner);
    }
    param.mBaseSize = outer * (aiv / aic);
    param.s2BaseSize = inner;
    param.l2Byte = arch92 ? 0 : 96U * 1024U * 1024U;
    param.fdTolerance = arch92 ? 300 : 10;
    // Match ParamsInit: arch92 keeps GeneralBalanceParam's default fdLeastBlock=0.
    if (!arch92) param.fdLeastBlock = 3;
    param.fdOn = !arch92;
    param.kernelSplitMode = load_balance::KernelSplitMode::BN1_S1_S2;
}
inline bool NeedInit(const Request& request, const std::vector<int64_t>& qLens,
                     const std::vector<int64_t>& kvLens) {
    const auto batch = request.batch;
    const auto mask = request.mask;
    const auto& cuQ = request.cuQ;
    const auto& cuKv = request.cuKv;

    bool needInit = false;
    for (uint32_t b = 0; b < batch; ++b) {
        const auto qAlloc = cuQ[b + 1] - cuQ[b];
        const auto kvAlloc = cuKv.empty() ? -1 : cuKv[b + 1] - cuKv[b];
        needInit |= qAlloc == 0 || qLens[b] == 0 || qLens[b] < qAlloc || kvAlloc == 0 || kvLens[b] == 0 ||
                    (mask == 3 && qLens[b] > kvLens[b]);
    }
    return needInit;
}
} // namespace qfa_mock
