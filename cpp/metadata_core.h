#pragma once
#include <cstdint>
#include <vector>

namespace qfa_mock {
inline constexpr const char* kAdapterVersion = "qfa-host-v2";
inline constexpr uint64_t kMaxMetadataBytes = 256ULL * 1024 * 1024;
struct Target {
    bool arch92 = true;
    uint32_t aic = 80, aiv = 80;
};
struct Request {
    uint32_t batch = 1, nq = 1, nk = 1, dim = 128, dimV = 128, mask = 3;
    int64_t maxQ = 1, maxKv = 1;
    bool paged = true;
    std::vector<int64_t> cuQ, cuKv, usedQ, usedKv;
};
struct Result {
    uint64_t row = 0, usedWords = 0;
    std::vector<uint32_t> words, activeAic, activeAiv;
};
void Require(bool ok, const char* message);
void ValidateRequest(const Target&, const Request&);
uint64_t RowSize(const Target&, const Request&);
Result Compute(const Target&, const Request&, uint64_t rowCapacity = 0);
void Validate(const Target&, const Request&, const Result&);
} // namespace qfa_mock
