#include "metadata_adapter.h"
#include <algorithm>
#include <array>
#include <climits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
namespace qfa_mock {
void Require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
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

void ValidateRequest(const Target& t, const Request& r) {
    Require(t.aic >= 1 && t.aic <= 1024 && t.aiv >= 1 && t.aiv <= 2048, "invalid target cores");
    Require(t.aiv == t.aic * (t.arch92 ? 1U : 2U), "unsupported core ratio");
    Require(r.batch >= 1 && r.batch <= 4096 && r.nq >= 1 && r.nq <= 1024 &&
            r.nk >= 1 && r.nk <= r.nq && r.nq % r.nk == 0, "invalid batch/head configuration");
    Require(r.dim == 128 && r.dimV == 128, "unsupported dimensions");
    Require(r.maxQ > 0 && r.maxQ <= INT32_MAX && r.maxKv > 0 && r.maxKv <= INT32_MAX, "invalid max lengths");
    Require(r.mask == 0 || r.mask == 3 || r.mask == 5, "unsupported mask mode");
    for (const auto* values : {&r.cuQ, &r.cuKv, &r.usedQ, &r.usedKv})
        for (auto value : *values) Require(value >= 0 && value <= INT32_MAX, "invalid length value");
    for (const auto* cu : {&r.cuQ, &r.cuKv}) if (!cu->empty())
        Require(cu->size() == r.batch + 1 && cu->front() == 0 && std::is_sorted(cu->begin(), cu->end()), "invalid cumulative array");
    for (const auto* used : {&r.usedQ, &r.usedKv})
        Require(used->empty() || used->size() == r.batch, "invalid used length array");
    Require(!r.cuQ.empty(), "TND Q requires cumulative lengths");
    Require(r.paged ? r.cuKv.empty() && !r.usedKv.empty() : !r.cuKv.empty(), "KV lengths/layout mismatch");
    UsedLengths(r.cuQ, r.usedQ, r.batch, r.maxQ);
    UsedLengths(r.cuKv, r.usedKv, r.batch, r.maxKv);
}
uint64_t RowSize(const Target& t, const Request& r) {
    ValidateRequest(t, r);
    const uint64_t row = ((16ULL + uint64_t(t.aic + t.aiv) * 16 * r.batch * r.nq + 4095) / 4096) * 4096;
    Require(row * 8 <= kMaxMetadataBytes, "metadata capacity limit exceeded");
    return row;
}
Result Compute(const Target& target, const Request& request, uint64_t rowCapacity) {
    ValidateRequest(target, request);
    const auto aic = target.aic, aiv = target.aiv;
    const uint64_t row = rowCapacity ? rowCapacity : RowSize(target, request);
    Require(row >= 16 && row <= kMaxMetadataBytes / 8, "invalid metadata row capacity");
    const auto qLens = UsedLengths(request.cuQ, request.usedQ, request.batch, request.maxQ);
    const auto kvLens = UsedLengths(request.cuKv, request.usedKv, request.batch, request.maxKv);
    load_balance::DeviceInfo device;
    load_balance::BaseInfo base;
    load_balance::SectionStreamKParam param;
    Configure(target, request, qLens, kvLens, device, base, param);
    load_balance::SectionStreamKResult result;
    Require(load_balance::SectionStreamK::Compute(device, base, param, result) == SECTION_STREAM_K_SUCCESS,
            "SectionStreamK::Compute failed");
    Require(result.sectionNum > 0, "scheduler returned no sections");
    const uint64_t usedWords = 16 + uint64_t(result.sectionNum) * (aic + aiv) * 16;
    Require(usedWords <= row, "scheduler result exceeds row capacity");

    const bool needInit = NeedInit(request, qLens, kvLens);
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

    Result output{row, usedWords, std::move(words), std::move(activeAic), std::move(activeAiv)};
    Validate(target, request, output);
    return output;
}
} // namespace qfa_mock

namespace qfa_mock {
void Validate(const Target& t, const Request& r, const Result& result) {
    ValidateRequest(t, r);
    const auto& w = result.words;
    Require(result.row >= 16 && result.row <= kMaxMetadataBytes / 8 && w.size() == 2 * result.row,
            "invalid metadata allocation");
    const uint32_t sections = w[0], m = w[2], s2base = w[3];
    Require(sections > 0 && m > 0 && s2base > 0 && w[1] <= 1 && w[4] == t.aic && w[5] == t.aiv,
            "invalid metadata header");
    for (size_t i = 6; i < 15; ++i) Require(w[i] == 0, "nonzero reserved header");
    const auto q = UsedLengths(r.cuQ, r.usedQ, r.batch, r.maxQ);
    const auto kv = UsedLengths(r.cuKv, r.usedKv, r.batch, r.maxKv);
    Require(w[15] == uint32_t(NeedInit(r, q, kv)), "incorrect output initialization flag");
    const uint64_t used = 16 + uint64_t(sections) * (t.aic + t.aiv) * 16;
    Require(used <= result.row && result.usedWords == used, "metadata capacity mismatch");
    Require(std::all_of(w.begin() + used, w.end(), [](uint32_t x) { return x == 0; }), "nonzero metadata padding/backward row");
    Require(result.activeAic.size() == sections && result.activeAiv.size() == sections, "missing active core counts");
    using Point = std::array<uint32_t, 3>;
    using Task = std::pair<uint32_t, uint32_t>;
    struct Coverage { uint32_t end = 0, pieces = 0; };
    std::map<Task, Coverage> coverage;
    const uint32_t bnTotal = r.batch * r.nq;
    auto blocks = [&](uint32_t bn) { return uint32_t((q[bn / r.nq] + m - 1) / m); };
    auto s2end = [&](uint32_t bn, uint32_t mi) -> uint32_t {
        const auto b = bn / r.nq;
        if (q[b] == 0 || kv[b] == 0) return 0;
        if (r.mask != 3) return (kv[b] + s2base - 1) / s2base;
        const int64_t last = std::min<int64_t>((uint64_t(mi) + 1) * m, q[b]) - 1 + kv[b] - q[b];
        return last < 0 ? 0 : uint32_t(std::min<int64_t>(last, kv[b] - 1) / s2base + 1);
    };
    Point previous{0, 0, 0};
    uint64_t steps = 0;
    auto tick = [&]() { Require(++steps <= 20000000, "metadata validation work limit exceeded"); };
    for (uint32_t sec = 0; sec < sections; ++sec) {
        Require(result.activeAic[sec] >= 1 && result.activeAic[sec] <= t.aic && result.activeAiv[sec] <= t.aiv,
                "invalid active core counts");
        for (uint32_t c = 0; c < t.aic; ++c) {
            const uint64_t offset = 16 + (uint64_t(sec) * t.aic + c) * 16;
            const uint32_t* record = w.data() + offset;
            for (size_t i = 7; i < 16; ++i) Require(record[i] == 0, "nonzero reserved FA field");
            if (c >= result.activeAic[sec]) {
                for (size_t i = 0; i < 16; ++i) Require(record[i] == 0, "nonzero inactive FA record");
                continue;
            }
            Point begin{record[0], record[1], record[2]}, end{record[3], record[4], record[5]};
            Require(begin == previous && begin <= end && record[6] < 2 * t.aic, "invalid FA chain/workspace");
            for (auto p : {begin, end}) {
                Require(p[0] <= bnTotal, "FA BN out of range");
                if (p[0] == bnTotal) Require(p[1] == 0 && p[2] == 0, "invalid terminal FA point");
                else Require(p[1] <= blocks(p[0]) && p[2] <= uint64_t(kv[p[0] / r.nq] + s2base - 1) / s2base,
                             "FA M/S2 out of range");
            }
            Point p = begin;
            while (p != end) {
                tick();
                Require(p < end && p[0] < bnTotal, "FA walk exceeds endpoint");
                const auto limit = p[1] < blocks(p[0]) ? s2end(p[0], p[1]) : 0;
                if (p[1] >= blocks(p[0]) || p[2] >= limit) {
                    ++p[1];
                    if (p[1] >= blocks(p[0])) { p[1] = 0; ++p[0]; }
                    p[2] = 0;
                    continue;
                }
                const auto finish = (p[0] == end[0] && p[1] == end[1]) ? std::min(end[2], limit) : limit;
                Require(finish > p[2], "empty FA segment");
                auto& span = coverage[{p[0], p[1]}];
                Require(span.end == p[2], "FA coverage gap or overlap");
                span.end = finish; ++span.pieces;
                if (finish == limit) {
                    ++p[1];
                    if (p[1] >= blocks(p[0])) { p[1] = 0; ++p[0]; }
                    p[2] = 0;
                } else p[2] = finish;
            }
            previous = end;
        }
    }
    std::set<Task> splitTasks;
    for (uint32_t bn = 0; bn < bnTotal; ++bn) for (uint32_t mi = 0; mi < blocks(bn); ++mi) {
        tick();
        const auto it = coverage.find({bn, mi});
        const auto finish = it == coverage.end() ? 0 : it->second.end;
        Require(finish == s2end(bn, mi), "incomplete FA coverage");
        if (it != coverage.end() && it->second.pieces > 1) splitTasks.insert({bn, mi});
    }
    using FdTask = std::tuple<uint32_t, uint32_t, uint32_t>;
    std::map<FdTask, std::vector<std::pair<uint32_t, uint32_t>>> intervals;
    std::map<FdTask, std::pair<uint32_t, uint32_t>> workspaces;
    std::set<Task> fdTasks;
    const uint64_t fdBase = 16 + uint64_t(sections) * t.aic * 16;
    for (uint32_t sec = 0; sec < sections; ++sec) for (uint32_t c = 0; c < t.aiv; ++c) {
        const auto* record = w.data() + fdBase + (uint64_t(sec) * t.aiv + c) * 16;
        for (size_t i = 6; i < 16; ++i) Require(record[i] == 0, "nonzero reserved FD field");
        if (c >= result.activeAiv[sec]) {
            for (size_t i = 0; i < 16; ++i) Require(record[i] == 0, "nonzero inactive FD record");
            continue;
        }
        const auto bn = record[0], mi = record[1], workspace = record[2], parts = record[3], start = record[4], count = record[5];
        Require(bn < bnTotal && mi < blocks(bn) && parts >= 2 && count > 0, "invalid FD task");
        const uint64_t rows = std::min<int64_t>(m, q[bn / r.nq] - uint64_t(mi) * m);
        Require(uint64_t(start) + count <= rows && uint64_t(workspace) + parts <= 2 * t.aic, "FD interval/workspace out of range");
        const FdTask key{sec, bn, mi};
        const std::pair<uint32_t, uint32_t> value{workspace, parts};
        Require(workspaces.count(key) == 0 || workspaces.at(key) == value, "FD workspace mismatch");
        workspaces[key] = value;
        Require(coverage.at({bn, mi}).pieces == parts, "FD split count mismatch");
        intervals[key].push_back({start, start + count}); fdTasks.insert({bn, mi});
    }
    Require(fdTasks == splitTasks && w[1] == uint32_t(!fdTasks.empty()), "FD/FA split mismatch");
    Require(!t.arch92 || fdTasks.empty(), "unsupported arch92 FD");
    std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> workspaceSpans;
    for (auto& item : intervals) {
        std::sort(item.second.begin(), item.second.end());
        uint32_t cursor = 0;
        for (const auto& span : item.second) { Require(span.first == cursor, "FD coverage gap/overlap"); cursor = span.second; }
        const auto bn = std::get<1>(item.first), mi = std::get<2>(item.first);
        Require(cursor == std::min<int64_t>(m, q[bn / r.nq] - uint64_t(mi) * m), "incomplete FD coverage");
        const auto value = workspaces.at(item.first);
        workspaceSpans[std::get<0>(item.first)].push_back({value.first, value.first + value.second});
    }
    for (auto& item : workspaceSpans) {
        std::sort(item.second.begin(), item.second.end());
        for (size_t i = 1; i < item.second.size(); ++i)
            Require(item.second[i - 1].second <= item.second[i].first, "overlapping FD workspace");
    }
}
} // namespace qfa_mock
