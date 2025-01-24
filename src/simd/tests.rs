#[cfg(test)]
mod tests {
    use crate::simd::{SimdVector, SimdFeatureLevel};

    #[test]
    fn test_simd_addition() {
        let data1: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let data2: Vec<f64> = (0..100).map(|x| (x * 2) as f64).collect();
        
        let simd_vec1 = SimdVector::new(&data1);
        let simd_vec2 = SimdVector::new(&data2);
        
        let result = simd_vec1.add(&simd_vec2);
        
        // Verify results using public methods
        for i in 0..100 {
            let result_val = result.get(i).unwrap();
            assert!((*result_val - (data1[i] + data2[i])).abs() < 1e-10);
        }
    }

    #[test]
    fn test_simd_feature_detection() {
        let level = SimdFeatureLevel::detect();
        println!("Detected SIMD feature level: {:?}", level);
        
        #[cfg(target_arch = "x86_64")]
        assert!(matches!(
            level,
            SimdFeatureLevel::None | SimdFeatureLevel::Avx2
        ));

        #[cfg(target_arch = "aarch64")]
        assert!(matches!(
            level,
            SimdFeatureLevel::None | SimdFeatureLevel::Neon
        ));

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert!(matches!(level, SimdFeatureLevel::None));
    }

    #[test]
    fn test_simd_vector_creation() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let simd_vec = SimdVector::new(&data);
        assert_eq!(simd_vec.len(), data.len());
    }

    #[test]
    #[should_panic(expected = "Vector lengths must match")]
    fn test_simd_addition_mismatched_lengths() {
        let data1: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let data2: Vec<f64> = (0..50).map(|x| x as f64).collect();
        
        let simd_vec1 = SimdVector::new(&data1);
        let simd_vec2 = SimdVector::new(&data2);
        
        let _ = simd_vec1.add(&simd_vec2);
    }

    #[test]
    fn test_simd_vector_operations() {
        let data1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let data2: Vec<f64> = vec![2.0, 3.0, 4.0, 5.0];
        
        let simd_vec1 = SimdVector::new(&data1);
        let simd_vec2 = SimdVector::new(&data2);
        
        let result = simd_vec1.add(&simd_vec2);
        
        // Test using as_slice
        let result_slice = result.as_slice();
        assert_eq!(result_slice, &[3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_width() {
        let level = SimdFeatureLevel::detect();
        let width = level.vector_width();
        
        #[cfg(target_arch = "x86_64")]
        assert!(width == 1 || width == 32);

        #[cfg(target_arch = "aarch64")]
        assert!(width == 16);

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert_eq!(width, 1);
    }

    #[test]
    fn test_simd_addition_edge_cases() {
        // Test with very large numbers
        let data1 = vec![f64::MAX / 2.0, f64::MAX / 3.0];
        let data2 = vec![f64::MAX / 2.0, f64::MAX / 3.0];
        let simd_vec1 = SimdVector::new(&data1);
        let simd_vec2 = SimdVector::new(&data2);
        let result = simd_vec1.add(&simd_vec2);
        assert!((result.get(0).unwrap() - f64::MAX).abs() < f64::EPSILON);
        
        // Test with infinities
        let data3 = vec![f64::INFINITY, -f64::INFINITY];
        let data4 = vec![1.0, -1.0];
        let simd_vec3 = SimdVector::new(&data3);
        let simd_vec4 = SimdVector::new(&data4);
        let result2 = simd_vec3.add(&simd_vec4);
        assert!(result2.get(0).unwrap().is_infinite() && result2.get(0).unwrap().is_sign_positive());
        assert!(result2.get(1).unwrap().is_infinite() && result2.get(1).unwrap().is_sign_negative());

        // Test with NaN
        let data5 = vec![f64::NAN, 1.0];
        let data6 = vec![1.0, 1.0];
        let simd_vec5 = SimdVector::new(&data5);
        let simd_vec6 = SimdVector::new(&data6);
        let result3 = simd_vec5.add(&simd_vec6);
        assert!(result3.get(0).unwrap().is_nan());
        assert!((result3.get(1).unwrap() - 2.0).abs() < f64::EPSILON);

        // Test with zeros
        let data7 = vec![0.0, -0.0];
        let data8 = vec![0.0, 0.0];
        let simd_vec7 = SimdVector::new(&data7);
        let simd_vec8 = SimdVector::new(&data8);
        let result4 = simd_vec7.add(&simd_vec8);
        assert_eq!(*result4.get(0).unwrap(), 0.0);
        assert_eq!(*result4.get(1).unwrap(), 0.0);
    }
}