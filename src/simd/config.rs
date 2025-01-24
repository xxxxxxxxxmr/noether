/// Represents the available SIMD features on the current platform
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimdFeatureLevel {
    None,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl SimdFeatureLevel {
    /// Detects the best available SIMD feature level
    #[inline]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Self::Neon;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::None
        }
    }

    /// Returns the optimal vector width for the current feature level
    #[inline]
    pub const fn vector_width(&self) -> usize {
        match self {
            Self::None => 1,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => 32, // 256 bits
            #[cfg(target_arch = "aarch64")]
            Self::Neon => 16, // 128 bits
        }
    }
}