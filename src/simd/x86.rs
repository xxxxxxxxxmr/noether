//! x86_64 SIMD implementations using AVX2

use super::traits::*;
use core::arch::x86_64::*;
use std::mem::transmute;

#[repr(align(32))]
pub struct Avx2Vector<T>(pub [T; 32]);

// Implementation for f64 (double precision)
impl SimdSupport for f64 {
    type Vector = Avx2Vector<f64>;
}

impl SimdField for f64 {
    #[target_feature(enable = "avx2")]
    unsafe fn load_simd(ptr: *const Self) -> Self::Vector {
        let vec = _mm256_loadu_pd(ptr as *const f64);
        Avx2Vector(transmute(vec))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn store_simd(vector: Self::Vector, ptr: *mut Self) {
        _mm256_storeu_pd(ptr as *mut f64, transmute(vector.0));
    }

    #[target_feature(enable = "avx2")]
    unsafe fn add_simd(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let a_vec: __m256d = transmute(a.0);
        let b_vec: __m256d = transmute(b.0);
        Avx2Vector(transmute(_mm256_add_pd(a_vec, b_vec)))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn sub_simd(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let a_vec: __m256d = transmute(a.0);
        let b_vec: __m256d = transmute(b.0);
        Avx2Vector(transmute(_mm256_sub_pd(a_vec, b_vec)))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mul_simd(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let a_vec: __m256d = transmute(a.0);
        let b_vec: __m256d = transmute(b.0);
        Avx2Vector(transmute(_mm256_mul_pd(a_vec, b_vec)))
    }
}