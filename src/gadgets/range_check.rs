use alloc::vec;
use alloc::vec::Vec;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::{
    gates::range_check_ux::UXRangeCheckGate,
    gadgets::arithmetic_ux::UXTarget
};

pub fn range_check_ux_circuit<F: RichField + Extendable<D>, const D: usize, const BITS: usize>(
    builder: &mut CircuitBuilder<F, D>,
    vals: Vec<UXTarget<BITS>>
) {
    let num_input_limbs = vals.len();
    let gate = UXRangeCheckGate::<F, D, BITS>::new(num_input_limbs);
    let row = builder.add_gate(gate, vec![]);

    for i in 0..num_input_limbs {
        builder.connect(Target::wire(row, gate.wire_ith_input_limb(i)), vals[i].0);
    }
}
