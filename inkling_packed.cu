#include <cuda_runtime.h>
#include <cstdint>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

// One descriptor per expert (or one for a dense matrix). Expert IDs map to
// packed slab offsets on the host so a batch can mix different codebooks.
struct Matrix {
    const uint32_t* codes;
    const float* scales;
    const float* lookup;
    int bits, group, hash_shift, scalar_levels;
};
static_assert(sizeof(Matrix) == 40, "Unexpected matrix descriptor layout");

__device__ inline uint32_t index_at(const Matrix& matrix, const uint32_t* row,
                                    int words, int width, int j) {
    const int bit = matrix.scalar_levels ? j * matrix.bits
                                        : (width - (j + 1) * matrix.group) * matrix.bits;
    const int word = bit >> 5;
    const int offset = bit & 31;
    uint32_t value = row[word] >> offset;
    const int needed = matrix.scalar_levels ? matrix.bits : 12;
    if (offset + needed > 32 && word + 1 < words)
        value |= row[word + 1] << (32 - offset);
    if (matrix.scalar_levels)
        return value & ((1u << matrix.bits) - 1u);
    value &= 4095u;
    return matrix.group == 1 ? value : (value * 2654435761u) >> matrix.hash_shift;
}

__global__ void matvec_kernel(const Matrix* plans, const int64_t* expert_ids, const float* input,
                              float* output, int rows, int width, int input_divisor) {
    const int row_index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row_index >= rows) return;
    const int slot = blockIdx.y;
    const Matrix matrix = plans[expert_ids ? expert_ids[slot] : 0];
    const int words = width * matrix.bits / 32;
    const uint32_t* packed = matrix.codes + int64_t(row_index) * words;
    const float* x = input + int64_t(slot / input_divisor) * width;
    const int lane = threadIdx.x & 31;
    float sum = 0.0f;
    for (int j = lane; j < width / matrix.group; j += 32) {
        const uint32_t index = index_at(matrix, packed, words, width, j);
        const float* values = matrix.lookup + index * matrix.group;
        for (int g = 0; g < matrix.group; ++g)
            sum = fmaf(values[g], x[j * matrix.group + g], sum);
    }
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0)
        output[int64_t(slot) * rows + row_index] = sum * matrix.scales[row_index];
}

__global__ void gather_kernel(const Matrix* plans, const int32_t* token_map, const int64_t* ids,
                              float* output, int width) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= width) return;
    const int slot = blockIdx.y;
    const int64_t token = ids[slot];
    const int bucket = token_map[token * 2];
    const int row = token_map[token * 2 + 1];
    const Matrix matrix = plans[bucket];
    const int words = width * matrix.bits / 32;
    const uint32_t index = index_at(matrix, matrix.codes + int64_t(row) * words,
                                    words, width, col / matrix.group);
    output[int64_t(slot) * width + col] =
        matrix.lookup[index * matrix.group + col % matrix.group] * matrix.scales[row];
}

EXPORT int packed_matvec(const Matrix* plans, const int64_t* ids, const float* input,
                         float* output, int slots, int rows, int width, int divisor,
                         cudaStream_t stream) {
    if (slots <= 0 || slots > 65535 || rows <= 0 || width <= 0 || divisor <= 0)
        return int(cudaErrorInvalidValue);
    matvec_kernel<<<dim3((rows + 3) / 4, slots), 128, 0, stream>>>(
        plans, ids, input, output, rows, width, divisor);
    return int(cudaGetLastError());
}

EXPORT int packed_gather(const Matrix* plans, const int32_t* token_map, const int64_t* ids,
                         float* output, int slots, int width, cudaStream_t stream) {
    if (slots <= 0 || slots > 65535 || width <= 0) return int(cudaErrorInvalidValue);
    gather_kernel<<<dim3((width + 127) / 128, slots), 128, 0, stream>>>(
        plans, token_map, ids, output, width);
    return int(cudaGetLastError());
}
