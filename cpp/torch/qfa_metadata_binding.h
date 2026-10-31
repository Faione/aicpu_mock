#pragma once
#include <torch/extension.h>
#include <functional>
#include <string>
#include <type_traits>

namespace qfa_mock {
using OptionalTensor = c10::optional<at::Tensor>;
using NativeMetadata = at::Tensor (*)(const OptionalTensor&, const OptionalTensor&, const OptionalTensor&,
    const OptionalTensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, std::string, std::string, std::string, std::string, bool, const at::Tensor&);
at::Tensor Dispatch(NativeMetadata, const OptionalTensor&, const OptionalTensor&, const OptionalTensor&,
    const OptionalTensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, std::string, std::string, std::string, std::string, bool, const at::Tensor&);

template<class R, class... Args>
std::function<R(Args...)> MakeMetadataBinding(R (*native)(Args...)) {
    static_assert(std::is_same_v<decltype(native), NativeMetadata>, "metadata signature requires explicit adaptation");
    return [native](Args... args) -> R { return Dispatch(native, args...); };
}
void RegisterMetadataControls(pybind11::module_& module);
} // namespace qfa_mock
