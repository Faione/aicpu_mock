// Host adapter for the production QFA scheduler. Compile against ops-transformer headers.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "metadata_core.h"
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
void PrintVector(const std::vector<uint32_t>& values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
    std::cout << ']';
}
} // namespace

int main(int argc, char** argv) try {
    Require(argc == 2 || (argc == 4 && std::string(argv[2]) == "--row-size"), "usage: metadata_helper output.bin [--row-size words] < private-wire-request");
    std::string magic;
    Require(bool(std::cin >> magic) && magic == "QFA_HOST_V1", "invalid wire version");
    const bool arch92 = ReadInt(0, 1);
    const uint32_t aic = ReadInt(1, 1024), aiv = ReadInt(1, 2048);
    const uint32_t batch = ReadInt(1, 4096), nq = ReadInt(1, 1024), nk = ReadInt(1, nq);
    const uint32_t dim = ReadInt(1, INT32_MAX), dimV = ReadInt(1, INT32_MAX);
    const int64_t maxQ = ReadInt(1, INT32_MAX), maxKv = ReadInt(1, INT32_MAX);
    const uint32_t mask = ReadInt(0, 5);
    const bool paged = ReadInt(0, 1);
    const auto cuQ = ReadArray(batch, true), cuKv = ReadArray(batch, true);
    const auto usedQ = ReadArray(batch, false), usedKv = ReadArray(batch, false);
    std::string trailing;
    Require(!(std::cin >> trailing), "trailing wire input");
    qfa_mock::Target target{arch92, aic, aiv};
    qfa_mock::Request request{batch, nq, nk, dim, dimV, mask, maxQ, maxKv, paged, cuQ, cuKv, usedQ, usedKv};
    uint64_t capacity = 0;
    if (argc == 4) {
        const std::string text = argv[3];
        Require(!text.empty() && std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }), "invalid row capacity");
        capacity = std::stoull(text);
        Require(capacity > 0, "row capacity must be positive");
    }
    const auto result = qfa_mock::Compute(target, request, capacity);
    const auto row = result.row, usedWords = result.usedWords;
    const auto& words = result.words;
    const auto& activeAic = result.activeAic;
    const auto& activeAiv = result.activeAiv;
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
