use super::traits::*;
use core::arch::aarch64::*;
use std::mem::transmute;

#[repr(align(16))]
pub struct NeonVector<T>(pub [T; 2]); // NEON processes 2 f64s at a time

impl SimdSupport for f64 {
    #[cfg(target_arch = "aarch64")]
    type Vector = NeonVector<f64>;
}

impl SimdField for f64 {
    #[target_feature(enable = "neon")]
    unsafe fn load_simd(ptr: *const Self) -> Self::Vector {
        let vec = vld1q_f64(ptr);
        NeonVector(transmute::<float64x2_t, [f64; 2]>(vec))
    }

    #[target_feature(enable = "neon")]
    unsafe fn store_simd(vector: Self::Vector, ptr: *mut Self) {
        let vec: float64x2_t = transmute::<[f64; 2], float64x2_t>(vector.0);
        vst1q_f64(ptr, vec);
    }

    #[target_feature(enable = "neon")]
    unsafe fn add_simd(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let a_vec: float64x2_t = transmute::<[f64; 2], float64x2_t>(a.0);
        let b_vec: float64x2_t = transmute::<[f64; 2], float64x2_t>(b.0);
        NeonVector(transmute::<float64x2_t, [f64; 2]>(vaddq_f64(a_vec, b_vec)))
    }

    #[target_feature(enable = "neon")]
    unsafe fn sub_simd(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let a_vec: float64x2_t = transmute::<[f64; 2], float64x2_t>(a.0);
        let b_vec: float64x2_t = transmute::<[f64; 2], float64x2_t>(b.0);
        NeonVector(transmute::<float64x2_t, [f64; 2]>(vsubq_f64(a_vec, b_vec)))
    }

    #[target_feature(enable = "neon")]
    unsafe fn mul_simd(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let a_vec: float64x2_t = transmute::<[f64; 2], float64x2_t>(a.0);
        let b_vec: float64x2_t = transmute::<[f64; 2], float64x2_t>(b.0);
        NeonVector(transmute::<float64x2_t, [f64; 2]>(vmulq_f64(a_vec, b_vec)))
    }
}