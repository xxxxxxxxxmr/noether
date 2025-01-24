use super::config::SimdFeatureLevel;
use super::traits::*;
use std::marker::PhantomData;
use num_traits::Euclid;

#[repr(align(32))]
pub struct SimdVector<T: SimdField + Copy> {
    data: Vec<T>,
    simd_level: SimdFeatureLevel,
    _phantom: PhantomData<T>,
}

impl<T: SimdField + Copy> SimdVector<T> {
    pub fn new(data: &[T]) -> Self {
        let mut aligned_data = Vec::with_capacity(data.len());
        aligned_data.extend_from_slice(data);
        
        Self {
            data: aligned_data,
            simd_level: SimdFeatureLevel::detect(),
            _phantom: PhantomData,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len(), "Vector lengths must match");
        
        let mut result = Vec::with_capacity(self.data.len());
        unsafe {
            result.set_len(self.data.len()); // Preallocate the full length
        }
        
        match self.simd_level {
            SimdFeatureLevel::None => {
                // Scalar fallback
                for i in 0..self.data.len() {
                    result[i] = self.data[i] + other.data[i];
                }
            },
            #[cfg(target_arch = "aarch64")]
            SimdFeatureLevel::Neon => {
                let chunk_size = 2; // NEON processes 2 f64s at a time
                let (chunks, remainder) = self.data.len().div_rem_euclid(&chunk_size);
                
                unsafe {
                    // Process chunks using SIMD
                    for i in 0..chunks {
                        let offset = i * chunk_size;
                        let a = T::load_simd(self.data[offset..].as_ptr());
                        let b = T::load_simd(other.data[offset..].as_ptr());
                        let sum = T::add_simd(a, b);
                        T::store_simd(sum, result[offset..].as_mut_ptr());
                    }
                    
                    // Handle remaining elements
                    let start = chunks * chunk_size;
                    for i in 0..remainder {
                        result[start + i] = self.data[start + i] + other.data[start + i];
                    }
                }
            },
            #[cfg(target_arch = "x86_64")]
            SimdFeatureLevel::Avx2 => {
                let chunk_size = 4; // AVX2 processes 4 f64s at a time
                let (chunks, remainder) = self.data.len().div_rem_euclid(chunk_size);
                
                unsafe {
                    // Process chunks using SIMD
                    for i in 0..chunks {
                        let offset = i * chunk_size;
                        let a = T::load_simd(self.data[offset..].as_ptr());
                        let b = T::load_simd(other.data[offset..].as_ptr());
                        let sum = T::add_simd(a, b);
                        T::store_simd(sum, result[offset..].as_mut_ptr());
                    }
                    
                    // Handle remaining elements
                    let start = chunks * chunk_size;
                    for i in 0..remainder {
                        result[start + i] = self.data[start + i] + other.data[start + i];
                    }
                }
            },
        }
        
        Self {
            data: result,
            simd_level: self.simd_level,
            _phantom: PhantomData,
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }
}