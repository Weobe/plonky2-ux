use alloc::vec::Vec;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::gates::add_many_ux::UXAddManyGate;

use crate::gates::arithmetic_ux::UXArithmeticGate;
use crate::gates::subtraction_ux::UXSubtractionGate;

#[derive(Clone, Copy, Debug)]
pub struct UXTarget<const BITS: usize>(pub Target);

pub trait CircuitBuilderUX<F: RichField + Extendable<D>, const D: usize> {
    fn add_virtual_ux_target<const BITS: usize>(&mut self) -> UXTarget<BITS>;

    fn add_virtual_ux_targets<const BITS: usize>(&mut self, n: usize) -> Vec<UXTarget<BITS>>;

    /// Returns a UXTarget for the value `c`, which is assumed to be at most 32 bits.
    fn constant_ux<const BITS: usize>(&mut self, c: u32) -> UXTarget<BITS>;

    fn zero_ux<const BITS: usize>(&mut self) -> UXTarget<BITS>;

    fn one_ux<const BITS: usize>(&mut self) -> UXTarget<BITS>;

    fn connect_ux<const BITS: usize>(&mut self, x: UXTarget<BITS>, y: UXTarget<BITS>);

    fn assert_zero_ux<const BITS: usize>(&mut self, x: UXTarget<BITS>);

    /// Checks for special cases where the value of
    /// `x * y + z`
    /// can be determined without adding a `UXArithmeticGate`.
    fn arithmetic_ux_special_cases<const BITS: usize>(
        &mut self,
        x: UXTarget<BITS>,
        y: UXTarget<BITS>,
        z: UXTarget<BITS>,
    ) -> Option<(UXTarget<BITS>, UXTarget<BITS>)>;

    // Returns x * y + z.
    fn mul_add_ux<const BITS: usize>(
        &mut self,
        x: UXTarget<BITS>,
        y: UXTarget<BITS>,
        z: UXTarget<BITS>,
    ) -> (UXTarget<BITS>, UXTarget<BITS>);

    fn add_ux<const BITS: usize>(
        &mut self,
        a: UXTarget<BITS>,
        b: UXTarget<BITS>,
    ) -> (UXTarget<BITS>, UXTarget<BITS>);

    fn add_many_ux<const BITS: usize>(
        &mut self,
        to_add: &[UXTarget<BITS>],
    ) -> (UXTarget<BITS>, UXTarget<BITS>);

    fn add_uxs_with_carry<const BITS: usize>(
        &mut self,
        to_add: &[UXTarget<BITS>],
        carry: UXTarget<BITS>,
    ) -> (UXTarget<BITS>, UXTarget<BITS>);

    fn mul_ux<const BITS: usize>(
        &mut self,
        a: UXTarget<BITS>,
        b: UXTarget<BITS>,
    ) -> (UXTarget<BITS>, UXTarget<BITS>);

    // Returns x - y - borrow, as a pair (result, borrow), where borrow is 0 or 1 depending on whether borrowing from the next digit is required (iff y + borrow > x).
    fn sub_ux<const BITS: usize>(
        &mut self,
        x: UXTarget<BITS>,
        y: UXTarget<BITS>,
        borrow: UXTarget<BITS>,
    ) -> (UXTarget<BITS>, UXTarget<BITS>);
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderUX<F, D> for CircuitBuilder<F, D> {
    fn add_virtual_ux_target<const BITS: usize>(&mut self) -> UXTarget<BITS> {
        UXTarget(self.add_virtual_target())
    }

    fn add_virtual_ux_targets<const BITS: usize>(&mut self, n: usize) -> Vec<UXTarget<BITS>> {
        self.add_virtual_targets(n)
            .into_iter()
            .map(UXTarget)
            .collect()
    }

    /// Returns a UXTarget for the value `c`, which is assumed to be at most 32 bits.
    fn constant_ux<const BITS: usize>(&mut self, c: u32) -> UXTarget<BITS> {
        assert!(c < 1 << BITS);
        UXTarget(self.constant(F::from_canonical_u32(c)))
    }

    fn zero_ux<const BITS: usize>(&mut self) -> UXTarget<BITS> {
        UXTarget(self.zero())
    }

    fn one_ux<const BITS: usize>(&mut self) -> UXTarget<BITS> {
        UXTarget(self.one())
    }

    fn connect_ux<const BITS: usize>(&mut self, x: UXTarget<BITS>, y: UXTarget<BITS>) {
        self.connect(x.0, y.0)
    }

    fn assert_zero_ux<const BITS: usize>(&mut self, x: UXTarget<BITS>) {
        self.assert_zero(x.0)
    }

    /// Checks for special cases where the value of
    /// `x * y + z`
    /// can be determined without adding a `UXArithmeticGate`.
    fn arithmetic_ux_special_cases<const BITS: usize>(
        &mut self,
        x: UXTarget<BITS>,
        y: UXTarget<BITS>,
        z: UXTarget<BITS>,
    ) -> Option<(UXTarget<BITS>, UXTarget<BITS>)> {
        let x_const = self.target_as_constant(x.0);
        let y_const = self.target_as_constant(y.0);
        let z_const = self.target_as_constant(z.0);

        // If both terms are constant, return their (constant) sum.
        let first_term_const = if let (Some(xx), Some(yy)) = (x_const, y_const) {
            Some(xx * yy)
        } else {
            None
        };

        if let (Some(a), Some(b)) = (first_term_const, z_const) {
            let sum = (a + b).to_canonical_u64();
            assert!(sum < (1u64 << (BITS * 2)));
            let modd = 1u64 << BITS;
            let (low, high) = (sum % modd, (sum >> BITS) as u32);
            return Some((
                self.constant_ux(low.try_into().expect("Value doesn't fit into u32")),
                self.constant_ux(high),
            ));
        }

        None
    }

    // Returns x * y + z.
    fn mul_add_ux<const BITS: usize>(
        &mut self,
        x: UXTarget<BITS>,
        y: UXTarget<BITS>,
        z: UXTarget<BITS>,
    ) -> (UXTarget<BITS>, UXTarget<BITS>) {
        if let Some(result) = self.arithmetic_ux_special_cases(x, y, z) {
            return result;
        }

        let gate = UXArithmeticGate::<F, D, BITS>::new_from_config(&self.config);
        let (row, copy) = self.find_slot(gate, &[], &[]);

        self.connect(Target::wire(row, gate.wire_ith_multiplicand_0(copy)), x.0);
        self.connect(Target::wire(row, gate.wire_ith_multiplicand_1(copy)), y.0);
        self.connect(Target::wire(row, gate.wire_ith_addend(copy)), z.0);

        let output_low = UXTarget(Target::wire(row, gate.wire_ith_output_low_half(copy)));
        let output_high = UXTarget(Target::wire(row, gate.wire_ith_output_high_half(copy)));

        (output_low, output_high)
    }

    fn add_ux<const BITS: usize>(
        &mut self,
        a: UXTarget<BITS>,
        b: UXTarget<BITS>,
    ) -> (UXTarget<BITS>, UXTarget<BITS>) {
        let one = self.one_ux();
        self.mul_add_ux(a, one, b)
    }

    fn add_many_ux<const BITS: usize>(
        &mut self,
        to_add: &[UXTarget<BITS>],
    ) -> (UXTarget<BITS>, UXTarget<BITS>) {
        match to_add.len() {
            0 => (self.zero_ux(), self.zero_ux()),
            1 => (to_add[0], self.zero_ux()),
            2 => self.add_ux(to_add[0], to_add[1]),
            _ => {
                let num_addends = to_add.len();
                let gate = UXAddManyGate::<F, D, BITS>::new_from_config(&self.config, num_addends);
                let (row, copy) =
                    self.find_slot(gate, &[F::from_canonical_usize(num_addends)], &[]);

                for j in 0..num_addends {
                    self.connect(
                        Target::wire(row, gate.wire_ith_op_jth_addend(copy, j)),
                        to_add[j].0,
                    );
                }
                let zero = self.zero();
                self.connect(Target::wire(row, gate.wire_ith_carry(copy)), zero);

                let output_low = UXTarget(Target::wire(row, gate.wire_ith_output_result(copy)));
                let output_high = UXTarget(Target::wire(row, gate.wire_ith_output_carry(copy)));

                (output_low, output_high)
            }
        }
    }

    fn add_uxs_with_carry<const BITS: usize>(
        &mut self,
        to_add: &[UXTarget<BITS>],
        carry: UXTarget<BITS>,
    ) -> (UXTarget<BITS>, UXTarget<BITS>) {
        if to_add.len() == 1 {
            return self.add_ux(to_add[0], carry);
        }

        let num_addends = to_add.len();

        let gate = UXAddManyGate::<F, D, BITS>::new_from_config(&self.config, num_addends);
        let (row, copy) = self.find_slot(gate, &[F::from_canonical_usize(num_addends)], &[]);

        for j in 0..num_addends {
            self.connect(
                Target::wire(row, gate.wire_ith_op_jth_addend(copy, j)),
                to_add[j].0,
            );
        }
        self.connect(Target::wire(row, gate.wire_ith_carry(copy)), carry.0);

        let output = UXTarget(Target::wire(row, gate.wire_ith_output_result(copy)));
        let output_carry = UXTarget(Target::wire(row, gate.wire_ith_output_carry(copy)));

        (output, output_carry)
    }

    fn mul_ux<const BITS: usize>(
        &mut self,
        a: UXTarget<BITS>,
        b: UXTarget<BITS>,
    ) -> (UXTarget<BITS>, UXTarget<BITS>) {
        let zero = self.zero_ux();
        self.mul_add_ux(a, b, zero)
    }

    // Returns x - y - borrow, as a pair (result, borrow), where borrow is 0 or 1 depending on whether borrowing from the next digit is required (iff y + borrow > x).
    fn sub_ux<const BITS: usize>(
        &mut self,
        x: UXTarget<BITS>,
        y: UXTarget<BITS>,
        borrow: UXTarget<BITS>,
    ) -> (UXTarget<BITS>, UXTarget<BITS>) {
        let gate = UXSubtractionGate::<F, D, BITS>::new_from_config(&self.config);
        let (row, copy) = self.find_slot(gate, &[], &[]);

        self.connect(Target::wire(row, gate.wire_ith_input_x(copy)), x.0);
        self.connect(Target::wire(row, gate.wire_ith_input_y(copy)), y.0);
        self.connect(
            Target::wire(row, gate.wire_ith_input_borrow(copy)),
            borrow.0,
        );

        let output_result = UXTarget(Target::wire(row, gate.wire_ith_output_result(copy)));
        let output_borrow = UXTarget(Target::wire(row, gate.wire_ith_output_borrow(copy)));

        (output_result, output_borrow)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use rand::rngs::OsRng;
    use rand::Rng;

    use super::*;

    const BITS: usize = 29;
    #[test]
    pub fn test_add_many_uxs() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        const NUM_ADDENDS: usize = 8;

        let config = CircuitConfig::standard_recursion_config();

        let pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let mut rng = OsRng;
        let mut to_add: Vec<UXTarget<BITS>> = Vec::new();
        let mut sum = 0u64;
        let modd = 1u64 << BITS;
        for _ in 0..NUM_ADDENDS {
            let x: u32 = (rng.gen::<u64>() % modd).try_into()?;
            sum += x as u64;
            to_add.push(builder.constant_ux(x as u32));
        }
        let carry = builder.zero_ux();
        let (result_low, result_high) = builder.add_uxs_with_carry(&to_add, carry);
        let expected_low = builder.constant_ux((sum % (1u64 << BITS)) as u32);
        let expected_high = builder.constant_ux((sum >> BITS) as u32);

        builder.connect_ux(result_low, expected_low);
        builder.connect_ux(result_high, expected_high);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }
}
