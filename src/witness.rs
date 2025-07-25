use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::generator::GeneratedValues;
use plonky2::iop::witness::{Witness, WitnessWrite};

use crate::gadgets::arithmetic_ux::UXTarget;

pub trait WitnessUX<F: PrimeField64, const BITS: usize>: Witness<F> {
    fn set_ux_target(&mut self, target: UXTarget<BITS>, value: u32) -> Result<()>;
    fn get_ux_target(&self, target: UXTarget<BITS>) -> (u32, u32);
}

impl<T: Witness<F>, F: PrimeField64, const BITS: usize> WitnessUX<F, BITS> for T {
    fn set_ux_target(&mut self, target: UXTarget<BITS>, value: u32) -> Result<()> {
        self.set_target(target.0, F::from_canonical_u32(value))
    }

    fn get_ux_target(&self, target: UXTarget<BITS>) -> (u32, u32) {
        let x_u64 = self.get_target(target.0).to_canonical_u64();
        let low = x_u64 as u32;
        let high = (x_u64 >> BITS) as u32;
        (low, high)
    }
}

pub trait GeneratedValuesUX<F: Field, const BITS: usize> {
    fn set_ux_target(&mut self, target: UXTarget<BITS>, value: u32) -> Result<()>;
}

impl<F: Field, const BITS: usize> GeneratedValuesUX<F, BITS> for GeneratedValues<F> {
    fn set_ux_target(&mut self, target: UXTarget<BITS>, value: u32) -> Result<()> {
        self.set_target(target.0, F::from_canonical_u32(value))
    }
}
