use super::config::SimdFeatureLevel;
use crate::Field;
use num_traits::Zero;

/// Marker trait for types that support SIMD operations
pub trait SimdSupport: Sized {
    /// The SIMD vector type corresponding to this scalar type
    type Vector;
    
    /// Returns the current SIMD feature level
    #[inline]
    fn simd_feature_level() -> SimdFeatureLevel {
        SimdFeatureLevel::detect()
    }
    
    /// Returns true if SIMD operations are available
    #[inline]
    fn has_simd_support() -> bool {
        !matches!(Self::simd_feature_level(), SimdFeatureLevel::None)
    }
}

/// SIMD operations for field elements
pub trait SimdField: Field + SimdSupport + Copy + Zero {
    /// Loads scalar values into a SIMD vector
    unsafe fn load_simd(ptr: *const Self) -> Self::Vector;
    
    /// Stores SIMD vector back to scalar values
    unsafe fn store_simd(vector: Self::Vector, ptr: *mut Self);
    
    /// Performs SIMD addition
    unsafe fn add_simd(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    
    /// Performs SIMD subtraction
    unsafe fn sub_simd(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    
    /// Performs SIMD multiplication
    unsafe fn mul_simd(a: Self::Vector, b: Self::Vector) -> Self::Vector;
}