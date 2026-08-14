#include <torch/extension.h>
#include <supa.h>

#include <cstdint>
#include <tuple>

extern "C" void launch_complex_mul_forward(
    const void* input,
    const void* weight,
    void* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int mode_count,
    suStream_t stream);

extern "C" void launch_complex_mul_backward_input(
    const void* grad_output,
    const void* weight,
    void* grad_input,
    int batch_size,
    int in_channels,
    int out_channels,
    int mode_count,
    suStream_t stream);

extern "C" void launch_complex_mul_backward_weight(
    const void* grad_output,
    const void* input,
    void* grad_weight,
    int batch_size,
    int in_channels,
    int out_channels,
    int mode_count,
    suStream_t stream);

namespace {

void check_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name,
        " must be a SUPA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.scalar_type() == at::kComplexFloat,
        name,
        " must have dtype torch.complex64");
    TORCH_CHECK(tensor.dim() == 3, name, " must be a rank-3 tensor");
}

void check_forward_inputs(
    const torch::Tensor& input,
    const torch::Tensor& weight) {
    check_tensor(input, "input");
    check_tensor(weight, "weight");
    TORCH_CHECK(
        input.device() == weight.device(),
        "input and weight must be on the same SUPA device");
    TORCH_CHECK(
        input.size(1) == weight.size(0),
        "input and weight channel dimensions do not match");
    TORCH_CHECK(
        input.size(2) == weight.size(2),
        "input and weight mode dimensions do not match");
    TORCH_CHECK(
        input.size(0) > 0 && input.size(1) > 0 &&
            weight.size(1) > 0 && input.size(2) > 0,
        "batch, channel, and mode dimensions must be positive");
}

}  // namespace

torch::Tensor complex_mul_modes_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight) {
    check_forward_inputs(input, weight);
    const auto batch_size = static_cast<int>(input.size(0));
    const auto in_channels = static_cast<int>(input.size(1));
    const auto out_channels = static_cast<int>(weight.size(1));
    const auto mode_count = static_cast<int>(input.size(2));
    auto output = torch::empty(
        {batch_size, out_channels, mode_count},
        input.options());

    launch_complex_mul_forward(
        input.data_ptr(),
        weight.data_ptr(),
        output.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        mode_count,
        nullptr);
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> complex_mul_modes_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight) {
    check_forward_inputs(input, weight);
    check_tensor(grad_output, "grad_output");
    const auto batch_size = static_cast<int>(input.size(0));
    const auto in_channels = static_cast<int>(input.size(1));
    const auto out_channels = static_cast<int>(weight.size(1));
    const auto mode_count = static_cast<int>(input.size(2));
    TORCH_CHECK(
        grad_output.device() == input.device(),
        "grad_output and input must be on the same SUPA device");
    TORCH_CHECK(
        grad_output.size(0) == batch_size &&
            grad_output.size(1) == out_channels &&
            grad_output.size(2) == mode_count,
        "grad_output shape must be [B, C_out, K]");

    auto grad_input = torch::empty_like(input);
    auto grad_weight = torch::empty_like(weight);
    launch_complex_mul_backward_input(
        grad_output.data_ptr(),
        weight.data_ptr(),
        grad_input.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        mode_count,
        nullptr);
    launch_complex_mul_backward_weight(
        grad_output.data_ptr(),
        input.data_ptr(),
        grad_weight.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        mode_count,
        nullptr);
    return std::make_tuple(grad_input, grad_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "complex_mul_modes",
        &complex_mul_modes_forward,
        "SUPA complex mode multiplication forward");
    module.def(
        "complex_mul_modes_forward",
        &complex_mul_modes_forward,
        "SUPA complex mode multiplication forward");
    module.def(
        "complex_mul_modes_backward",
        &complex_mul_modes_backward,
        "SUPA complex mode multiplication backward");
}
