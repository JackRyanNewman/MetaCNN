#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <immintrin.h>
#include <stdalign.h>
#include <omp.h>
          // 47100000 train
          // 7880000  teest
#define SIZE 47100000
#define ITERATIONS 100
#ifndef TESTF
#define TESTF 0
#endif

#ifndef PRINTHEADER
#define PRINTHEADER 0
#endif

#ifndef SIMDON
#define SIMDON 0
#endif

#ifndef FOOBAR
#define FOOBAR 0
#endif

#define Num int

#if SIMDON
    alignas(32) Num dataF[SIZE];
    alignas(32) Num dataB[SIZE];
#else
    static Num data[SIZE];
    static Num data[SIZE];
#endif

#define stride 1

//make make -C "00_research_notes_n_references\0_Compiler_Expirments\looping" all
//// Remove-Item loop_per.exe -ErrorAction Ignore; gcc -O -mavx2 loop_per.c -o loop_per; if ($?) { ./loop_per }


void use_result(float sum) {
    // Volatile to prevent optimization
    static volatile float result;
    result = sum;
}

float tm_for(size_t size) {
    clock_t start = clock(); float sum = 0; 
    if(SIMDON) {_Pragma("omp simd") }
    
    
    for (size_t i = 0; i < size; i++) {
        data[i]+=1;
        sum += data[i];
    }
    if(FOOBAR) use_result(sum);
    return ((float)(clock() - start)) / CLOCKS_PER_SEC;
}


float tm_back(size_t size) {
    clock_t start = clock(); float sum = 0;
    if(SIMDON) {_Pragma("omp simd") }
    size_t i = size+(stride-1);
    do{
        W
        data[i]+=1;
      sum += data[i];
    } while(i != 0 );
    
    if(FOOBAR) use_result(sum);
    return ((float)(clock() - start)) / CLOCKS_PER_SEC;
}

int main() {
    float forward_time = 0.0, backward_time = 0.0;
    
    if(PRINTHEADER==1) printf("\n* Testing loop per (%d iterations, data size: %d, Vect %5d, OPT%d)\n", ITERATIONS, SIZE, SIMDON, FOOBAR);

    // Optional cache warmup
    if(TESTF) tm_for(SIZE);
    else tm_back(SIZE);


    for (int i = 0; i < ITERATIONS; ++i) {
        float bt =0, ft=0; 
        if(TESTF) ft = tm_for(SIZE);
        else bt = tm_back(SIZE);
        forward_time += ft;
        backward_time += bt;
        //printf("Run %2d: Forward: %.4fs, Backward: %.8fs\n", i + 1, ft, bt);
    }
    
    if(TESTF) printf(" |- Average - Forward:  %.8f seconds\n", forward_time / ITERATIONS);
    else      printf(" |- Average - Backward: %.8f seconds\n", backward_time / ITERATIONS);
    
    //printf("Forward is %.2f%% faster\n", ((backward_time - forward_time) / forward_time) * 100);
    
    


    return 0;
}