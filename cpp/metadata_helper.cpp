// Host adapter for the production QFA scheduler. Compile against ops-transformer headers.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "attention/common/op_kernel/load_balance/section_stream_k/section_stream_k.h"
#include "attention/quant_flash_attn/op_host/qfa_adjust_sinner_souter.h"
#include "attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata.h"

namespace {
void Require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
int64_t ReadInt(int64_t low, int64_t high) {
    int64_t value;
    Require(bool(std::cin >> value), "truncated or invalid wire input");
    Require(value >= low && value <= high, "wire integer outside supported range");
    return value;
}
std::vector<int64_t> ReadArray(uint32_t batch, bool cumulative) {
    const auto count = ReadInt(-1, 4097);
    if (count == -1) return {};
    Require(count == batch + (cumulative ? 1U : 0U), "wire array length mismatch");
    std::vector<int64_t> values;
    for (int64_t i = 0; i < count; ++i) values.push_back(ReadInt(0, INT32_MAX));
    if (cumulative) {
        Require(values.front() == 0 && std::is_sorted(values.begin(), values.end()), "invalid cumulative array");
    }
    return values;
}
std::vector<int64_t> UsedLengths(const std::vector<int64_t>& cu, const std::vector<int64_t>& used,
                                 uint32_t batch, int64_t maximum) {
    std::vector<int64_t> result(batch, maximum);
    for (uint32_t i = 0; i < batch; ++i) {
        const auto allocated = cu.empty() ? maximum : cu[i + 1] - cu[i];
        result[i] = used.empty() ? allocated : used[i];
        Require(allocated <= maximum && result[i] <= allocated, "effective/allocated/max length mismatch");
    }
    return result;
}
std::vector<int64_t> Cumulative(const std::vector<int64_t>& lengths) {
    std::vector<int64_t> result(lengths);
    for (size_t i = 1; i < result.size(); ++i) result[i] += result[i - 1];
    return result;
}
void PrintVector(const std::vector<uint32_t>& values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
    std::cout << ']';
}
} // namespace

int main(int argc, char** argv) try {
    Require(argc == 2, "usage: metadata_helper output.bin < private-wire-request");
    std::string magic;
    Require(bool(std::cin >> magic) && magic == "QFA_HOST_V1", "invalid wire version");
    const bool arch92 = ReadInt(0, 1);
    const uint32_t aic = ReadInt(1, 1024), aiv = ReadInt(1, 2048);
    Require(aiv == aic * (arch92 ? 1U : 2U), "unsupported core ratio");
    const uint32_t batch = ReadInt(1, 4096), nq = ReadInt(1, 1024), nk = ReadInt(1, nq);
    Require(nq % nk == 0, "invalid GQA ratio");
    const uint32_t dim = ReadInt(128, 128), dimV = ReadInt(128, 128);
    const int64_t maxQ = ReadInt(1, INT32_MAX), maxKv = ReadInt(1, INT32_MAX);
    const uint32_t mask = ReadInt(0, 5);
    Require(mask == 0 || mask == 3 || mask == 5, "unsupported mask mode");
    const bool paged = ReadInt(0, 1);
    const auto cuQ = ReadArray(batch, true), cuKv = ReadArray(batch, true);
    const auto usedQ = ReadArray(batch, false), usedKv = ReadArray(batch, false);
    std::string trailing;
    Require(!(std::cin >> trailing), "trailing wire input");
    Require(!cuQ.empty(), "TND Q requires cumulative lengths");
    Require(paged ? cuKv.empty() && !usedKv.empty() : !cuKv.empty(), "KV lengths/layout mismatch");
    const auto qLens = UsedLengths(cuQ, usedQ, batch, maxQ);
    const auto kvLens = UsedLengths(cuKv, usedKv, batch, maxKv);
    const uint64_t row = ((16ULL + uint64_t(aic + aiv) * 16 * batch * nq + 4095) / 4096) * 4096;
    Require(row * 8 <= 256ULL * 1024 * 1024, "metadata capacity limit exceeded");

    load_balance::DeviceInfo device;
    device.aicCoreMaxNum = device.aicCoreMinNum = aic;
    device.aivCoreMaxNum = device.aivCoreMinNum = aiv;
    load_balance::BaseInfo base;
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
    load_balance::SectionStreamKParam param;
    param.mBaseSize = outer * (aiv / aic);
    param.s2BaseSize = inner;
    param.l2Byte = arch92 ? 0 : 96U * 1024U * 1024U;
    param.fdTolerance = arch92 ? 300 : 10;
    // Match ParamsInit: arch92 keeps GeneralBalanceParam's default fdLeastBlock=0.
    if (!arch92) param.fdLeastBlock = 3;
    param.fdOn = !arch92;
    param.kernelSplitMode = load_balance::KernelSplitMode::BN1_S1_S2;
    load_balance::SectionStreamKResult result;
    Require(load_balance::SectionStreamK::Compute(device, base, param, result) == SECTION_STREAM_K_SUCCESS,
            "SectionStreamK::Compute failed");
    Require(result.sectionNum > 0, "scheduler returned no sections");
    const uint64_t usedWords = 16 + uint64_t(result.sectionNum) * (aic + aiv) * 16;
    Require(usedWords <= row, "scheduler result exceeds row capacity");

    // Equivalent to current CheckNeedInitOutput for the supported TND Q subset.
    bool needInit = false;
    for (uint32_t b = 0; b < batch; ++b) {
        const auto qAlloc = cuQ[b + 1] - cuQ[b];
        const auto kvAlloc = cuKv.empty() ? -1 : cuKv[b + 1] - cuKv[b];
        needInit |= qAlloc == 0 || qLens[b] == 0 || qLens[b] < qAlloc || kvAlloc == 0 || kvLens[b] == 0 ||
                    (mask == 3 && qLens[b] > kvLens[b]);
    }
    std::vector<uint32_t> words(row * 2, 0);
    optiling::detail::FaMetaData metadata(aic, aiv, result.sectionNum, words.data());
    metadata.Clear();
    metadata.SetHeadMedata(optiling::HEAD_SECTION_NUM_INDEX, result.sectionNum);
    metadata.SetHeadMedata(optiling::HEAD_M_BASE_SIZE_INDEX, param.mBaseSize);
    metadata.SetHeadMedata(optiling::HEAD_S2_BASE_SIZE_INDEX, param.s2BaseSize);
    metadata.SetHeadMedata(optiling::HEAD_AIC_NUM_INDEX, aic);
    metadata.SetHeadMedata(optiling::HEAD_AIV_NUM_INDEX, aiv);
    metadata.SetHeadMedata(optiling::HEAD_NEED_INIT_OUTPUT_INDEX, needInit);
    std::vector<uint32_t> activeAic, activeAiv;
    for (uint32_t s = 0; s < result.sectionNum; ++s) {
        const auto& fa = result.sectionFaResult[s];
        const auto& fd = result.sectionFdResult[s];
        Require(fa.usedCoreNum > 0 && fa.usedCoreNum <= aic && fd.usedVecNum <= aiv, "invalid active core count");
        activeAic.push_back(fa.usedCoreNum);
        activeAiv.push_back(fd.usedVecNum);
        if (fd.usedVecNum) metadata.SetHeadMedata(optiling::HEAD_IS_FD_INDEX, 1);
        for (uint32_t c = 0; c < fa.usedCoreNum; ++c) {
            if (c || s) {
                const auto& prev = c ? fa : result.sectionFaResult[s - 1];
                const auto pc = c ? c - 1 : prev.usedCoreNum - 1;
                metadata.SetFaMetadata(s, c, optiling::FA_BN_START_INDEX, prev.bNEnd[pc]);
                metadata.SetFaMetadata(s, c, optiling::FA_M_START_INDEX, prev.mEnd[pc]);
                metadata.SetFaMetadata(s, c, optiling::FA_S2_START_INDEX, prev.s2End[pc]);
            }
            metadata.SetFaMetadata(s, c, optiling::FA_BN_END_INDEX, fa.bNEnd[c]);
            metadata.SetFaMetadata(s, c, optiling::FA_M_END_INDEX, fa.mEnd[c]);
            metadata.SetFaMetadata(s, c, optiling::FA_S2_END_INDEX, fa.s2End[c]);
            metadata.SetFaMetadata(s, c, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, fa.firstFdDataWorkspaceIdx[c]);
        }
        for (uint32_t c = 0; c < fd.usedVecNum; ++c) {
            const auto task = fd.taskIdx[c];
            Require(task < aic, "FD task index out of range");
            metadata.SetFdMetadata(s, c, optiling::FD_BN_IDX_INDEX, fd.bNIdx[task]);
            metadata.SetFdMetadata(s, c, optiling::FD_M_IDX_INDEX, fd.mIdx[task]);
            metadata.SetFdMetadata(s, c, optiling::FD_WORKSPACE_IDX_INDEX, fd.workspaceIdx[task]);
            metadata.SetFdMetadata(s, c, optiling::FD_WORKSPACE_NUM_INDEX, fd.s2SplitNum[task]);
            metadata.SetFdMetadata(s, c, optiling::FD_M_START_INDEX, fd.mStart[c]);
            metadata.SetFdMetadata(s, c, optiling::FD_M_NUM_INDEX, fd.mLen[c]);
        }
    }
    std::ofstream out(argv[1], std::ios::binary | std::ios::trunc);
    Require(bool(out), "cannot open output file");
    for (const auto word : words) {
        for (unsigned shift = 0; shift < 32; shift += 8) out.put(static_cast<char>(word >> shift));
    }
    out.close();
    Require(bool(out), "cannot write output file");
    std::cout << "{\"protocol_version\":1,\"algorithm_version\":\"qfa-host-v1\",\"shape\":[2," << row
              << "],\"used_words\":" << usedWords << ",\"header\":";
    PrintVector(std::vector<uint32_t>(words.begin(), words.begin() + 16));
    std::cout << ",\"active_aic\":";
    PrintVector(activeAic);
    std::cout << ",\"active_aiv\":";
    PrintVector(activeAiv);
    std::cout << "}\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "{\"error\":{\"code\":\"HELPER_FAILED\",\"message\":\"" << e.what() << "\"}}\n";
    return 1;
}
