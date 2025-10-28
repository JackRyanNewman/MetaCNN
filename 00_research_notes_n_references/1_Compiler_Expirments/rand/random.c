#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define ARR_SIZE 10000000  // 10 million integers
#define MIN 0
#define MAX 100

//make -C "00_research_notes_n_references\1_Compiler_Expirments\rand" all

/* ------------------ XOROSHIRO128+ ------------------ */
uint64_t s0 = 0x123456789abcdefULL;
uint64_t s1 = 0xfedcba987654321ULL;

inline uint64_t xoroshiro128_next() {
    uint64_t result = s0 + s1;
    uint64_t t = s1 ^ s0;
    s0 = ((s0 << 55) | (s0 >> (64 - 55))) ^ t ^ (t << 14);
    s1 = (t << 36) | (t >> (64 - 36));
    return result;
}

inline int xoroshiro128_next_range(int min, int max) {
    return min + (int)(xoroshiro128_next() % (uint64_t)(max - min + 1));
}

/* ------------------ XORSHIFT32 ------------------ */
uint32_t xs = 123456789;

static inline uint32_t xorshift32_next() {
    xs ^= xs << 13;
    xs ^= xs >> 17;
    xs ^= xs << 5;
    return xs;
}

static inline int xorshift32_next_range(int min, int max) {
    return min + (int)(xorshift32_next() % (max - min + 1));
}

/* ------------------ MAIN BENCHMARK ------------------ */
int main() {
    int arr[ARR_SIZE];
    clock_t start, end;
    double time_xoroshiro, time_xorshift;

    // Benchmark xoroshiro128+
    start = clock();
    for (int i = 0; i < ARR_SIZE; i++) {
        arr[i] = xoroshiro128_next_range(MIN, MAX);
    }
    end = clock();
    time_xoroshiro = (double)(end - start) / CLOCKS_PER_SEC;

    printf("xoroshiro128+ filled array in %.6f seconds\n", time_xoroshiro);

    // Reset xorshift32 seed
    xs = 123456789;

    // Benchmark xorshift32
    start = clock();
    for (int i = 0; i < ARR_SIZE; i++) {
        arr[i] = xorshift32_next_range(MIN, MAX);
    }
    end = clock();
    time_xorshift = (double)(end - start) / CLOCKS_PER_SEC;

    printf("xorshift32 filled array in %.6f seconds\n", time_xorshift);

    return 0;
}
