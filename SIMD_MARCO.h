#ifndef SIMD_ABSTRACTION_H
#define SIMD_ABSTRACTION_H

#include <immintrin.h>
#ifndef SIMDID
	#define SIMID 256
#endif
#ifndef Numeric
	#define Numeric float
#endif


// Detect architecture and set vector width
#if SIMDID == 512 //__AVX512F__
		#define _VECT_PREFIX _mm512
		#define _VECT_WIDTH 512
		#define _VECT_INT_TYPE __m512i
		#define _VECT_FLOAT_TYPE __m512
		#define _VECT_DOUBLE_TYPE __m512d
		#define _VECT_INT_COUNT 16
		#define _VECT_FLOAT_COUNT 16
		#define _VECT_DOUBLE_COUNT 8
#elif SIMDID == 256 //defined(__AVX2__)
		#define _VECT_PREFIX _mm256
		#define _VECT_WIDTH 256
		#define _VECT_INT_TYPE __m256i
		#define _VECT_FLOAT_TYPE __m256
		#define _VECT_DOUBLE_TYPE __m256d
		#define _VECT_INT_COUNT 8
		#define _VECT_FLOAT_COUNT 8
		#define _VECT_DOUBLE_COUNT 4
#elif SIMDID == 128 //defined(__AVX2__)
		#define _VECT_PREFIX _mm
		#define _VECT_WIDTH 128
		#define _VECT_INT_TYPE __m128i
		#define _VECT_FLOAT_TYPE __m128
		#define _VECT_DOUBLE_TYPE __m128d
		#define _VECT_INT_COUNT 4
		#define _VECT_FLOAT_COUNT 4
		#define _VECT_DOUBLE_COUNT 2
#else
		#error "No supported SIMD architecture detected (SSE, AVX2, or AVX512 required)"
#endif

// Vector type definitions
#define VECT_INT _VECT_INT_TYPE
#define VECT_FLOAT _VECT_FLOAT_TYPE
#define VECT_DOUBLE _VECT_DOUBLE_TYPE

// Element count constants
#define VECT_INT_COUNT _VECT_INT_COUNT
#define VECT_FLOAT_COUNT _VECT_FLOAT_COUNT
#define VECT_DOUBLE_COUNT _VECT_DOUBLE_COUNT

// ---- token-paste helpers that force macro expansion ----
// primitive join helpers
#define _JOIN2(a,b) a##b
#define _EXPAND_JOIN2(a,b) _JOIN2(a,b)

#define _JOIN3(a,b,c) a##_##b##_##c
#define _EXPAND_JOIN3(a,b,c) _JOIN3(a,b,c)

// for cases like prefix_loadu_si256 we need a 4-part join where
// the last two parts (si and width) are glued without an underscore
// to produce "si256"
#define _JOIN4(a,b,c,d) a##_##b##_##c##d
#define _EXPAND_JOIN4(a,b,c,d) _JOIN4(a,b,c,d)

// Another useful helper: prefix_op_type where op and type are simple
#define _JOIN_OP_TYPE(prefix, op, type) _EXPAND_JOIN3(prefix, op, type)

// ---- Memory ops ----
// Integer load/store need the "si" + SIMDID suffix (e.g. si256, si128, si512)
	#define VECT_LOAD_INT(ptr)  _EXPAND_JOIN4(_VECT_PREFIX, loadu, si, _VECT_WIDTH)(ptr)
	#define VECT_STORE_INT(ptr, vec) _EXPAND_JOIN4(_VECT_PREFIX, storeu, si, _VECT_WIDTH)(ptr, vec)

// Float/double versions use op + ps/pd
	#define VECT_LOAD_FLOAT(ptr)  _JOIN_OP_TYPE(_VECT_PREFIX, loadu, ps)(ptr)
	#define VECT_STORE_FLOAT(ptr, vec) _JOIN_OP_TYPE(_VECT_PREFIX, storeu, ps)(ptr, vec)
	#define VECT_LOAD_DOUBLE(ptr) _JOIN_OP_TYPE(_VECT_PREFIX, loadu, pd)(ptr)
	#define VECT_STORE_DOUBLE(ptr, vec) _JOIN_OP_TYPE(_VECT_PREFIX, storeu, pd)(ptr, vec)

// Aligned load/store (requires alignment)
	#define VECT_LOAD_INT_ALIGNED(ptr)  _EXPAND_JOIN4(_VECT_PREFIX, load, si, _VECT_WIDTH)(ptr)
	#define VECT_STORE_INT_ALIGNED(ptr, vec) _EXPAND_JOIN4(_VECT_PREFIX, store, si, _VECT_WIDTH)(ptr, vec)
	#define VECT_LOAD_FLOAT_ALIGNED(ptr)  _JOIN_OP_TYPE(_VECT_PREFIX, load, ps)(ptr)
	#define VECT_STORE_FLOAT_ALIGNED(ptr, vec) _JOIN_OP_TYPE(_VECT_PREFIX, store, ps)(ptr, vec)
	#define VECT_LOAD_DOUBLE_ALIGNED(ptr) _JOIN_OP_TYPE(_VECT_PREFIX, load, pd)(ptr)
	#define VECT_STORE_DOUBLE_ALIGNED(ptr, vec) _JOIN_OP_TYPE(_VECT_PREFIX, store, pd)(ptr, vec)

// ---- Arithmetic ops ----
	#define VECT_ADD_INT(a, b)    _JOIN_OP_TYPE(_VECT_PREFIX, add, epi32)(a, b)
	#define VECT_SUB_INT(a, b)    _JOIN_OP_TYPE(_VECT_PREFIX, sub, epi32)(a, b)
	#define VECT_MULLO_INT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, mullo, epi32)(a, b)

	#define VECT_ADD_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, add, ps)(a, b)
	#define VECT_SUB_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, sub, ps)(a, b)
	#define VECT_MUL_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, mul, ps)(a, b)
	#define VECT_DIV_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, div, ps)(a, b)

	#define VECT_ADD_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, add, pd)(a, b)
	#define VECT_SUB_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, sub, pd)(a, b)
	#define VECT_MUL_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, mul, pd)(a, b)
	#define VECT_DIV_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, div, pd)(a, b)

// ---- Min/Max ops ----
	#define VECT_MIN_INT(a, b)    _JOIN_OP_TYPE(_VECT_PREFIX, min, epi32)(a, b)
	#define VECT_MAX_INT(a, b)    _JOIN_OP_TYPE(_VECT_PREFIX, max, epi32)(a, b)

	#define VECT_MIN_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, min, ps)(a, b)
	#define VECT_MAX_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, max, ps)(a, b)

	#define VECT_MIN_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, min, pd)(a, b)
	#define VECT_MAX_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, max, pd)(a, b)

// ---- Comparison ops ----
	#define VECT_EQ_INT(a, b)    _JOIN_OP_TYPE(_VECT_PREFIX, cmpeq, epi32)(a, b)
	#define VECT_GT_INT(a, b)    _JOIN_OP_TYPE(_VECT_PREFIX, cmpgt, epi32)(a, b)

	#define VECT_EQ_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, cmp, ps)(a, b, _CMP_EQ_OQ)
	#define VECT_LT_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, cmp, ps)(a, b, _CMP_LT_OQ)
	#define VECT_LE_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, cmp, ps)(a, b, _CMP_LE_OQ)
	#define VECT_GT_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, cmp, ps)(a, b, _CMP_GT_OQ)
	#define VECT_GE_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, cmp, ps)(a, b, _CMP_GE_OQ)

	#define VECT_EQ_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, cmp, pd)(a, b, _CMP_EQ_OQ)
	#define VECT_LT_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, cmp, pd)(a, b, _CMP_LT_OQ)
	#define VECT_LE_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, cmp, pd)(a, b, _CMP_LE_OQ)
	#define VECT_GT_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, cmp, pd)(a, b, _CMP_GT_OQ)
	#define VECT_GE_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, cmp, pd)(a, b, _CMP_GE_OQ)

// ---- Bitwise ops ----
	#define VECT_AND_INT(a, b)  _EXPAND_JOIN4(_VECT_PREFIX, and, si, _VECT_WIDTH)(a, b)
	#define VECT_OR_INT(a, b)   _EXPAND_JOIN4(_VECT_PREFIX, or, si, _VECT_WIDTH)(a, b)
	#define VECT_XOR_INT(a, b)  _EXPAND_JOIN4(_VECT_PREFIX, xor, si, _VECT_WIDTH)(a, b)
	#define VECT_ANDNOT_INT(a, b) _EXPAND_JOIN4(_VECT_PREFIX, andnot, si, _VECT_WIDTH)(a, b)

	#define VECT_AND_FLOAT(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, and, ps)(a, b)
	#define VECT_OR_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, or, ps)(a, b)
	#define VECT_XOR_FLOAT(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, xor, ps)(a, b)
	#define VECT_ANDNOT_FLOAT(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, andnot, ps)(a, b)

	#define VECT_AND_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, and, pd)(a, b)
	#define VECT_OR_DOUBLE(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, or, pd)(a, b)
	#define VECT_XOR_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, xor, pd)(a, b)
	#define VECT_ANDNOT_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, andnot, pd)(a, b)

// ---- Blending / Select ops ----
	#define VECT_BLENDV_FLOAT(a, b, mask) _JOIN_OP_TYPE(_VECT_PREFIX, blendv, ps)(a, b, mask)
	#define VECT_BLENDV_DOUBLE(a, b, mask) _JOIN_OP_TYPE(_VECT_PREFIX, blendv, pd)(a, b, mask)

// ---- Mathematical ops ----
	#define VECT_SQRT_FLOAT(a)  _JOIN_OP_TYPE(_VECT_PREFIX, sqrt, ps)(a)
	#define VECT_SQRT_DOUBLE(a) _JOIN_OP_TYPE(_VECT_PREFIX, sqrt, pd)(a)

	#define VECT_ABS_FLOAT(a)   _JOIN_OP_TYPE(_VECT_PREFIX, andnot, ps)(_JOIN_OP_TYPE(_VECT_PREFIX, set1, ps)(-0.0f), a)
	#define VECT_ABS_DOUBLE(a)  _JOIN_OP_TYPE(_VECT_PREFIX, andnot, pd)(_JOIN_OP_TYPE(_VECT_PREFIX, set1, pd)(-0.0), a)

// FMA operations (if available)
		#define VECT_FMA_FLOAT(a, b, c)  _JOIN_OP_TYPE(_VECT_PREFIX, fmadd, ps)(a, b, c)  // a*b+c
		#define VECT_FMA_DOUBLE(a, b, c) _JOIN_OP_TYPE(_VECT_PREFIX, fmadd, pd)(a, b, c)
		#define VECT_FMS_FLOAT(a, b, c)  _JOIN_OP_TYPE(_VECT_PREFIX, fmsub, ps)(a, b, c)  // a*b-c
		#define VECT_FMS_DOUBLE(a, b, c) _JOIN_OP_TYPE(_VECT_PREFIX, fmsub, pd)(a, b, c)

// ---- Horizontal ops ----
	#define VECT_HADD_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, hadd, ps)(a, b)
	#define VECT_HADD_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, hadd, pd)(a, b)
	#define VECT_HSUB_FLOAT(a, b)  _JOIN_OP_TYPE(_VECT_PREFIX, hsub, ps)(a, b)
	#define VECT_HSUB_DOUBLE(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, hsub, pd)(a, b)

// ---- Shuffle and permute ----
	#define VECT_SHUFFLE_FLOAT(a, b, imm) _JOIN_OP_TYPE(_VECT_PREFIX, shuffle, ps)(a, b, imm)
	#define VECT_SHUFFLE_DOUBLE(a, b, imm) _JOIN_OP_TYPE(_VECT_PREFIX, shuffle, pd)(a, b, imm)

// ---- Conversion ops ----
		#define VECT_CVTINT_FLOAT(a)   _JOIN_OP_TYPE(_VECT_PREFIX, cvtepi32, ps)(a)
		#define VECT_CVTFLOAT_INT(a)   _JOIN_OP_TYPE(_VECT_PREFIX, cvttps, epi32)(a)
		#define VECT_CVTFLOAT_DOUBLE(a) _JOIN_OP_TYPE(_VECT_PREFIX, cvtps, pd)(a)
		#define VECT_CVTDOUBLE_FLOAT(a) _JOIN_OP_TYPE(_VECT_PREFIX, cvtpd, ps)(a)

// ---- Initialization ----
	#define VECT_SET1_INT(val)   _JOIN_OP_TYPE(_VECT_PREFIX, set1, epi32)(val)
	#define VECT_SETZERO_INT()   _EXPAND_JOIN4(_VECT_PREFIX, setzero, si, _VECT_WIDTH)()
	#define VECT_SET1_FLOAT(val) _JOIN_OP_TYPE(_VECT_PREFIX, set1, ps)(val)
	#define VECT_SETZERO_FLOAT() _JOIN_OP_TYPE(_VECT_PREFIX, setzero, ps)()
	#define VECT_SET1_DOUBLE(val) _JOIN_OP_TYPE(_VECT_PREFIX, set1, pd)(val)
	#define VECT_SETZERO_DOUBLE() _JOIN_OP_TYPE(_VECT_PREFIX, setzero, pd)()

// ---- Movemask (extract sign bits) ----
	#define VECT_MOVEMASK_FLOAT(a)  _JOIN_OP_TYPE(_VECT_PREFIX, movemask, ps)(a)
	#define VECT_MOVEMASK_DOUBLE(a) _JOIN_OP_TYPE(_VECT_PREFIX, movemask, pd)(a)

// ---- Test ops (check if all zeros/ones) ----
	#define VECT_TESTZ_INT(a, b) _JOIN_OP_TYPE(_VECT_PREFIX, testz, si256)(a, b)

// ============================================================================
// VECT_NUMERIC_* - Automatic operations based on Numeric type
// ============================================================================

#if Numeric == double
	// Numeric is double
	#define VECT_NUMERIC VECT_DOUBLE
	#define VECT_NUMERIC_COUNT VECT_DOUBLE_COUNT

	#define VECT_LOAD_NUMERIC(ptr) VECT_LOAD_DOUBLE(ptr)
	#define VECT_STORE_NUMERIC(ptr, vec) VECT_STORE_DOUBLE(ptr, vec)
	#define VECT_LOAD_ALIGNED_NUMERIC(ptr) VECT_LOAD_ALIGNED_DOUBLE(ptr)
	#define VECT_STORE_ALIGNED_NUMERIC(ptr, vec) VECT_STORE_ALIGNED_DOUBLE(ptr, vec)

	#define VECT_ADD_NUMERIC(a, b) VECT_ADD_DOUBLE(a, b)
	#define VECT_SUB_NUMERIC(a, b) VECT_SUB_DOUBLE(a, b)
	#define VECT_MUL_NUMERIC(a, b) VECT_MUL_DOUBLE(a, b)
	#define VECT_DIV_NUMERIC(a, b) VECT_DIV_DOUBLE(a, b)

	#define VECT_MIN_NUMERIC(a, b) VECT_MIN_DOUBLE(a, b)
	#define VECT_MAX_NUMERIC(a, b) VECT_MAX_DOUBLE(a, b)

	#define VECT_SQRT_NUMERIC(a) VECT_SQRT_DOUBLE(a)
	#define VECT_ABS_NUMERIC(a) VECT_ABS_DOUBLE(a)
	#define VECT_FMA_NUMERIC(a, b, c) VECT_FMA_DOUBLE(a, b, c)
	#define VECT_FMS_NUMERIC(a, b, c) VECT_FMS_DOUBLE(a, b, c)

	#define VECT_EQ_NUMERIC(a, b) VECT_EQ_DOUBLE(a, b)
	#define VECT_LT_NUMERIC(a, b) VECT_LT_DOUBLE(a, b)
	#define VECT_LE_NUMERIC(a, b) VECT_LE_DOUBLE(a, b)
	#define VECT_GT_NUMERIC(a, b) VECT_GT_DOUBLE(a, b)
	#define VECT_GE_NUMERIC(a, b) VECT_GE_DOUBLE(a, b)

	#define VECT_AND_NUMERIC(a, b) VECT_AND_DOUBLE(a, b)
	#define VECT_OR_NUMERIC(a, b) VECT_OR_DOUBLE(a, b)
	#define VECT_XOR_NUMERIC(a, b) VECT_XOR_DOUBLE(a, b)
	#define VECT_ANDNOT_NUMERIC(a, b) VECT_ANDNOT_DOUBLE(a, b)

	#define VECT_BLENDV_NUMERIC(a, b, mask) VECT_BLENDV_DOUBLE(a, b, mask)

	#define VECT_SET1_NUMERIC(val) VECT_SET1_DOUBLE(val)
	#define VECT_SETZERO_NUMERIC() VECT_SETZERO_DOUBLE()

	#define VECT_MOVEMASK_NUMERIC(a) VECT_MOVEMASK_DOUBLE(a)

	#define VECT_HADD_NUMERIC(a, b) VECT_HADD_DOUBLE(a, b)
	#define VECT_HSUB_NUMERIC(a, b) VECT_HSUB_DOUBLE(a, b)
#else 	// Numeric is float (default)

	#define VECT_NUMERIC VECT_FLOAT
	#define VECT_NUMERIC_COUNT VECT_FLOAT_COUNT
	
	#define VECT_NUMERIC_LOAD(ptr) VECT_LOAD_FLOAT(ptr)
	#define VECT_NUMERIC_STORE(ptr, vec) VECT_STORE_FLOAT(ptr, vec)
	#define VECT_NUMERIC_LOAD_ALIGNED(ptr) VECT_LOAD_FLOAT_ALIGNED(ptr)
	#define VECT_NUMERIC_STORE_ALIGNED(ptr, vec) VECT_STORE_FLOAT_ALIGNED(ptr, vec)
	
	#define VECT_NUMERIC_ADD(a, b) VECT_ADD_FLOAT(a, b)
	#define VECT_NUMERIC_SUB(a, b) VECT_SUB_FLOAT(a, b)
	#define VECT_NUMERIC_MUL(a, b) VECT_MUL_FLOAT(a, b)
	#define VECT_NUMERIC_DIV(a, b) VECT_DIV_FLOAT(a, b)
	
	#define VECT_NUMERIC_MIN(a, b) VECT_MIN_FLOAT(a, b)
	#define VECT_NUMERIC_MAX(a, b) VECT_MAX_FLOAT(a, b)
	
	#define VECT_NUMERIC_SQRT(a) VECT_SQRT_FLOAT(a)
	#define VECT_NUMERIC_ABS(a) VECT_ABS_FLOAT(a)
	#define VECT_NUMERIC_FMA(a, b, c) VECT_FMA_FLOAT(a, b, c)
	#define VECT_NUMERIC_FMS(a, b, c) VECT_FMS_FLOAT(a, b, c)
	
	#define VECT_NUMERIC_EQ(a, b) VECT_EQ_FLOAT(a, b)
	#define VECT_NUMERIC_LT(a, b) VECT_LT_FLOAT(a, b)
	#define VECT_NUMERIC_LE(a, b) VECT_LE_FLOAT(a, b)
	#define VECT_NUMERIC_GT(a, b) VECT_GT_FLOAT(a, b)
	#define VECT_NUMERIC_GE(a, b) VECT_GE_FLOAT(a, b)
	
	#define VECT_NUMERIC_AND(a, b) VECT_AND_FLOAT(a, b)
	#define VECT_NUMERIC_OR(a, b) VECT_OR_FLOAT(a, b)
	#define VECT_NUMERIC_XOR(a, b) VECT_XOR_FLOAT(a, b)
	#define VECT_NUMERIC_ANDNOT(a, b) VECT_ANDNOT_FLOAT(a, b)
	
	#define VECT_NUMERIC_BLENDV(a, b, mask) VECT_BLENDV_FLOAT(a, b, mask)
	
	#define VECT_NUMERIC_SET1(val) VECT_SET1_FLOAT(val)
	#define VECT_NUMERIC_SETZERO() VECT_SETZERO_FLOAT()
	
	#define VECT_NUMERIC_MOVEMASK(a) VECT_MOVEMASK_FLOAT(a)
	
	#define VECT_NUMERIC_HADD(a, b) VECT_HADD_FLOAT(a, b)
	#define VECT_NUMERIC_HSUB(a, b) VECT_HSUB_FLOAT(a, b)

#endif

#endif // SIMD_ABSTRACTION_H