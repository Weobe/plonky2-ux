use alloc::vec::Vec;
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use crate::gadgets::arithmetic_ux::UXTarget;

pub trait WriteUX<const BITS: usize> {
    fn write_target_ux(&mut self, x: UXTarget<BITS>) -> IoResult<()>;
}

impl<const BITS: usize> WriteUX<BITS> for Vec<u8> {
    #[inline]
    fn write_target_ux(&mut self, x: UXTarget<BITS>) -> IoResult<()> {
        self.write_target(x.0)
    }
}

pub trait ReadUX<const BITS: usize> {
    fn read_target_ux(&mut self) -> IoResult<UXTarget<BITS>>;
}

impl<const BITS: usize> ReadUX<BITS> for Buffer<'_> {
    #[inline]
    fn read_target_ux(&mut self) -> IoResult<UXTarget<BITS>> {
        Ok(UXTarget(self.read_target()?))
    }
}
