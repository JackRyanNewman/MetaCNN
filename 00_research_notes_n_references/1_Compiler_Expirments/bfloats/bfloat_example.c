#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>

// Define bfloat16 as uint16_t
typedef uint16_t bfloat16;

// Convert float to bfloat16 (bit shift)
bfloat16 float_to_bf16(float f) {
    return (*(uint32_t*)&f) >> 16;
}

// Convert bfloat16 to float
float bf16_to_float(bfloat16 bf) {
    uint32_t bits = ((uint32_t)bf) << 16;
    return *(float*)&bits;
}

// Vectorized dot product of two bfloat16 arrays (returns float)
float bf16_dot_product(bfloat16* a, bfloat16* b, int n) {
    __m512 acc = _mm512_setzero_ps(); // Accumulate in float
    for (int i = 0; i < n; i += 32) { // Process 32 bfloat16s per 512-bit register
        if (i + 32 > n) break; // Handle edge case
        // Load bfloat16 arrays into 512-bit registers
        __m512i vec_a = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i vec_b = _mm512_loadu_si512((const __m512i*)(b + i));
        // Compute dot product (AVX-512 BF16, accumulates in float)
        acc = _mm512_dpbf16_ps(acc, (__m512bh)vec_a, (__m512bh)vec_b);
    }
    // Sum the float accumulator
    float result[16];
    _mm512_storeu_ps(result, acc);
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) sum += result[i];
    // Handle remaining elements (scalar)
    for (int i = (n / 32) * 32; i < n; i++) {
        sum += bf16_to_float(a[i]) * bf16_to_float(b[i]);
    }
    return sum;
}

int main() {
    #define N 32
    bfloat16 a[N], b[N];
    // Initialize arrays
    for (int i = 0; i < N; i++) {
        a[i] = float_to_bf16((float)(i + 1.0f));
        b[i] = float_to_bf16((float)(i + 2.0f));
    }
    // Compute dot product
    float result = bf16_dot_product(a, b, N);
    printf("Dot product: %.2f\n", result);
    // Verify with scalar computation
    float verify = 0.0f;
    for (int i = 0; i < N; i++) {
        verify += bf16_to_float(a[i]) * bf16_to_float(b[i]);
    }
    printf("Verification (scalar): %.2f\n", verify);
    return 0;
}