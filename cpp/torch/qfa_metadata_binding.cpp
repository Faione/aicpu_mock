#include "qfa_metadata_binding.h"
#include "metadata_core.h"
#include "qfa_mock_build.h"
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace qfa_mock {
namespace {
// PyBind keeps the GIL throughout control/compute calls. No inherited mutex on fork.
struct Context {
    pid_t pid = 0;
    bool initialized = false, mock = false;
    Target target;
    std::string identity;
    uint64_t nativeCalls = 0, mockCalls = 0;
    std::vector<uint32_t> header;
};
Context& State() {
    static Context value;
    if (value.pid != getpid()) { value = Context{}; value.pid = getpid(); }
    return value;
}
std::string Env(const char* key) { const auto* s = std::getenv(key); return s ? s : ""; }
bool RequestedMock() {
    auto mode = Env("QFA_METADATA_BACKEND");
    const bool configured = !Env("QFA_METADATA_CONFIG").empty();
    Require(mode.empty() || mode == "native" || mode == "mock_aicpu", "invalid QFA_METADATA_BACKEND");
    Require(mode != "native" || !configured, "native backend conflicts with mock configuration");
    return mode == "mock_aicpu" || (mode.empty() && configured);
}
void Configure(bool mock, bool arch92, uint32_t aic, uint32_t aiv, const std::string& identity) {
    auto& state = State();
    Require(mock == RequestedMock(), "backend/configuration mismatch");
    Require(identity == QFA_MOCK_BUILD_ID, "loaded extension build identity mismatch; rebuild/restart");
    if (state.initialized) {
        Require(state.mock == mock && state.target.arch92 == arch92 && state.target.aic == aic &&
                state.target.aiv == aiv && state.identity == identity, "worker backend/target changed; restart");
        return;
    }
    Request probe; probe.cuQ = {0, 1}; probe.usedKv = {1};
    if (mock) ValidateRequest(Target{arch92, aic, aiv}, probe);
    state.initialized = true; state.mock = mock; state.target = Target{arch92, aic, aiv}; state.identity = identity;
}
std::vector<int64_t> ReadLengths(const OptionalTensor& tensor, const at::Device& outputDevice) {
    if (!tensor.has_value() || tensor->numel() == 0) return {};
    Require(tensor->dim() == 1 && tensor->numel() <= 4097, "invalid length tensor shape");
    Require(tensor->scalar_type() == at::kInt || tensor->scalar_type() == at::kLong, "length tensor must be int32/int64");
    Require(tensor->device().is_cpu() || tensor->device() == outputDevice, "length/output device mismatch");
    auto host = tensor->to(at::kCPU).to(at::kLong).contiguous();
    const auto* ptr = host.data_ptr<int64_t>();
    return {ptr, ptr + host.numel()};
}
} // namespace

at::Tensor Dispatch(NativeMetadata native, const OptionalTensor& cuQ, const OptionalTensor& cuKv,
    const OptionalTensor& usedQ, const OptionalTensor& usedKv, int64_t batch, int64_t maxQ, int64_t maxKv,
    int64_t nq, int64_t nk, int64_t dim, int64_t dimV, int64_t quant, int64_t mask, int64_t left,
    int64_t right, std::string layoutQ, std::string layoutScale, std::string layoutKv, std::string layoutOut,
    bool grad, const at::Tensor& output) {
    auto& state = State();
    const bool requested = RequestedMock();
    if (!state.initialized) {
        Require(!requested, "mock backend not initialized before metadata execution");
        state.initialized = true; state.mock = false;
    }
    Require(state.mock == requested, "worker backend changed; restart");
    if (!state.mock) {
        ++state.nativeCalls;
        return native(cuQ, cuKv, usedQ, usedKv, batch, maxQ, maxKv, nq, nk, dim, dimV, quant, mask,
                      left, right, layoutQ, layoutScale, layoutKv, layoutOut, grad, output);
    }
    Require(quant == 1 && !grad && left == -1 && right == -1 && layoutQ == "TND" &&
            layoutScale == "TND" && layoutOut == "TND" && (layoutKv == "TND" || layoutKv == "PA_BNBD"),
            "unsupported metadata request");
    Require(batch >= 1 && batch <= 4096 && nq >= 1 && nq <= 1024 && nk >= 1 && nk <= nq &&
            dim == 128 && dimV == 128 && mask >= 0 && mask <= 5, "invalid metadata scalar arguments");
    Require(output.scalar_type() == at::kInt && output.dim() == 2 && output.size(0) == 2 &&
            output.is_contiguous(), "metadata output must be contiguous int32 (2,rowSize)");
#ifndef QFA_MOCK_CPU_ACCEPTANCE
    Require(output.device().type() == c10::DeviceType::PrivateUse1, "metadata output must be on NPU");
#endif
    Request r;
    r.batch = batch; r.nq = nq; r.nk = nk; r.dim = dim; r.dimV = dimV; r.mask = mask;
    r.maxQ = maxQ; r.maxKv = maxKv; r.paged = layoutKv == "PA_BNBD";
    r.cuQ = ReadLengths(cuQ, output.device()); r.cuKv = ReadLengths(cuKv, output.device());
    r.usedQ = ReadLengths(usedQ, output.device()); r.usedKv = ReadLengths(usedKv, output.device());
    auto result = Compute(state.target, r, output.size(1));
    auto host = at::empty({2, int64_t(result.row)}, at::TensorOptions().dtype(at::kInt).device(at::kCPU));
    std::memcpy(host.data_ptr<int32_t>(), result.words.data(), result.words.size() * sizeof(uint32_t));
    output.copy_(host, false);
    ++state.mockCalls;
    state.header.assign(result.words.begin(), result.words.begin() + 16);
    return output;
}
void RegisterMetadataControls(pybind11::module_& m) {
    m.def("_qfa_mock_configure", &Configure);
    m.def("_qfa_mock_info", []() {
        const auto& s = State(); pybind11::dict info;
        info["build_id"] = QFA_MOCK_BUILD_ID; info["adapter_version"] = kAdapterVersion;
        info["initialized"] = s.initialized; info["mock"] = s.mock;
        info["native_calls"] = s.nativeCalls; info["mock_calls"] = s.mockCalls; info["header"] = s.header;
#ifdef QFA_MOCK_CPU_ACCEPTANCE
        info["cpu_acceptance_only"] = true;
#else
        info["cpu_acceptance_only"] = false;
#endif
        return info;
    });
    m.def("_qfa_mock_row_size", [](int64_t batch, int64_t nq) {
        const auto& s = State(); Require(s.initialized && s.mock && RequestedMock(), "mock context not initialized");
        Require(batch >= 1 && batch <= 4096 && nq >= 1 && nq <= 1024, "invalid allocation shape");
        Request r; r.batch = batch; r.nq = nq;
        for (int64_t i = 0; i <= batch; ++i) r.cuQ.push_back(i);
        r.usedKv.assign(batch, 1);
        return RowSize(s.target, r);
    });
}
} // namespace qfa_mock
