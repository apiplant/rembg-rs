// ConvRot activation transform and row-wise INT8 quantization for Fibo2.
// The layout and scaling follow Comfy-Kitchen's int8_tensorwise linear path.

typedef unsigned short rmbg_u16;
typedef signed char rmbg_i8;
typedef int rmbg_i32;

__device__ __forceinline__ float abs_value(float value) {
    return value < 0.0f ? -value : value;
}

__device__ __forceinline__ float max_value(float a, float b) {
    return a > b ? a : b;
}

__device__ __forceinline__ float read_bf16(const rmbg_u16 *ptr) {
    return __uint_as_float(static_cast<unsigned int>(*ptr) << 16);
}

__device__ __forceinline__ rmbg_u16 write_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<rmbg_u16>(bits >> 16);
}

__device__ __forceinline__ float h4_value(float a, float b, float c, float d, int row) {
    if (row == 0) return 0.5f * (a + b + c - d);
    if (row == 1) return 0.5f * (a + b - c + d);
    if (row == 2) return 0.5f * (a - b + c + d);
    return 0.5f * (-a + b + c + d);
}

extern "C" __global__ void convrot_groups(
    const rmbg_u16 *__restrict__ input,
    float *__restrict__ rotated,
    float *__restrict__ group_max,
    int K,
    int groups) {
    __shared__ float values[256];
    __shared__ float reduction[256];
    const int row = static_cast<int>(blockIdx.x) / groups;
    const int group = static_cast<int>(blockIdx.x) - row * groups;
    const int lane = static_cast<int>(threadIdx.x);
    const int offset = row * K + group * 256;
    values[lane] = read_bf16(input + offset + lane);
    __syncthreads();

    for (int stride = 1; stride < 256; stride *= 4) {
        const int digit = (lane / stride) & 3;
        const int base = lane - digit * stride;
        const float a = values[base];
        const float b = values[base + stride];
        const float c = values[base + 2 * stride];
        const float d = values[base + 3 * stride];
        const float result = h4_value(a, b, c, d, digit);
        __syncthreads();
        values[lane] = result;
        __syncthreads();
    }

    const float value = values[lane];
    rotated[offset + lane] = value;
    reduction[lane] = abs_value(value);
    __syncthreads();
    for (int width = 128; width > 0; width >>= 1) {
        if (lane < width) reduction[lane] = max_value(reduction[lane], reduction[lane + width]);
        __syncthreads();
    }
    if (lane == 0) group_max[row * groups + group] = reduction[0];
}

extern "C" __global__ void quantize_rotated_rows(
    const float *__restrict__ input,
    const float *__restrict__ group_max,
    rmbg_i8 *__restrict__ output,
    float *__restrict__ scales,
    int K,
    int groups) {
    __shared__ float reduction[256];
    __shared__ float row_scale;
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    float peak = 0.0f;
    for (int g = lane; g < groups; g += 256) peak = max_value(peak, group_max[row * groups + g]);
    reduction[lane] = peak;
    __syncthreads();
    for (int width = 128; width > 0; width >>= 1) {
        if (lane < width) reduction[lane] = max_value(reduction[lane], reduction[lane + width]);
        __syncthreads();
    }
    if (lane == 0) {
        row_scale = reduction[0] > 0.0f ? reduction[0] / 127.0f : 1.0f;
        scales[row] = row_scale;
    }
    __syncthreads();
    for (int col = lane; col < K; col += 256) {
        const float value = input[row * K + col] / row_scale;
        const int quantized = __float2int_rn(value);
        output[row * K + col] = static_cast<rmbg_i8>(quantized < -127 ? -127 : (quantized > 127 ? 127 : quantized));
    }
}

extern "C" __global__ void quantize_bf16_rows(
    const rmbg_u16 *__restrict__ input,
    rmbg_i8 *__restrict__ output,
    float *__restrict__ scales,
    int K) {
    __shared__ float reduction[256];
    __shared__ float row_scale;
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    float peak = 0.0f;
    for (int col = lane; col < K; col += 256) peak = max_value(peak, abs_value(read_bf16(input + row * K + col)));
    reduction[lane] = peak;
    __syncthreads();
    for (int width = 128; width > 0; width >>= 1) {
        if (lane < width) reduction[lane] = max_value(reduction[lane], reduction[lane + width]);
        __syncthreads();
    }
    if (lane == 0) {
        row_scale = reduction[0] > 0.0f ? reduction[0] / 127.0f : 1.0f;
        scales[row] = row_scale;
    }
    __syncthreads();
    for (int col = lane; col < K; col += 256) {
        const float value = read_bf16(input + row * K + col) / row_scale;
        const int quantized = __float2int_rn(value);
        output[row * K + col] = static_cast<rmbg_i8>(quantized < -127 ? -127 : (quantized > 127 ? 127 : quantized));
    }
}

extern "C" __global__ void dequantize_int8_rows(
    const rmbg_i32 *__restrict__ accum,
    const float *__restrict__ x_scale,
    const float *__restrict__ weight_scale,
    const rmbg_u16 *__restrict__ bias,
    rmbg_u16 *__restrict__ output,
    int M,
    int N,
    int has_bias) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int total = M * N;
    if (index >= total) return;
    const int row = index / N;
    const int col = index - row * N;
    float value = static_cast<float>(accum[index]) * x_scale[row] * weight_scale[col];
    if (has_bias != 0) value += read_bf16(bias + col);
    output[index] = write_bf16(value);
}
