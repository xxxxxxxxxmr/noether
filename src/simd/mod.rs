//! SIMD support for algebraic operations

mod config;
mod traits;
#[cfg(target_arch = "x86_64")]
mod x86;
#[cfg(target_arch = "aarch64")]
mod arm;
mod wrapper;

#[cfg(test)]
mod tests;

pub use self::config::*;
pub use self::traits::*;
pub use self::wrapper::*;

#[cfg(target_arch = "x86_64")]
pub use self::x86::*;
#[cfg(target_arch = "aarch64")]
pub use self::arm::*;
