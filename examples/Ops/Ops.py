"""A small accelerator exercising the ops this generator can lower.

Every instruction here is a regression test for a template that was present in
templates.txt but unreachable because oracle_visitor.py had no mapping for it:
`constant`, the generic `reduce` (with a non-add combiner), `select_lt`, and
`exp` as an alias of `exponential` -- plus `broadcast`, whose `dimensions=`
attribute the generator used to ignore.

The data model is one bf16 scratchpad of 64-lane rows, plus HBM (d0).
"""

from act.taidl import Accelerator
from act.generators import generate_oracle

ops = Accelerator("Ops")
ops.add_data_model("d1", [64], [64], "bf16")


def _unary(name, semantics):
    instr = ops.add_instruction(name, ["n"], ["p", "o"])
    instr.set_inputs([["d1", ["@a.p"], ["@c.n"]]])
    instr.set_outputs([["d1", ["@a.o"], ["@c.n"]]])
    instr.add_semantics(semantics)


# reduce with a max combiner + a scalar-free broadcast back across lanes.
# `reduce_add` could never express this: the only combiners shipped were add_*.
_unary("redmax", """
ENTRY redmax {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %r = bf16[`@c.n`] reduce(%In1), dimensions={1}, to_apply=%max_bf16;
    ROOT %Out0 = bf16[`@c.n`,64] broadcast(%r), dimensions={0};
}
""")

# `constant` + `divide`: a reciprocal needs a literal 1, which was previously
# impossible to name anywhere in a semantics block.
_unary("recip", """
ENTRY recip {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %one_c = bf16[1] constant(1);
    %one = bf16[`@c.n`,64] broadcast(%one_c), dimensions={};
    ROOT %Out0 = bf16[`@c.n`,64] divide(%one, %In1);
}
""")

# A float literal, which only reaches the generator through the EXPRESSION
# token: `0.5` does not lex as INT, and simplify_vals used to floor it to 0.
_unary("halve", """
ENTRY halve {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %half_c = bf16[1] constant(`0.5`);
    %half = bf16[`@c.n`,64] broadcast(%half_c), dimensions={};
    ROOT %Out0 = bf16[`@c.n`,64] multiply(%In1, %half);
}
""")

# `exp` as an alias for `exponential` (the grammar accepted both; only one lowered).
_unary("expo", """
ENTRY expo {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] exp(%In1);
}
""")

# `select_lt`: relu, as select(x < 0, 0, x).
_unary("relu", """
ENTRY relu {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %zero_c = bf16[1] constant(0);
    %zero = bf16[`@c.n`,64] broadcast(%zero_c), dimensions={};
    ROOT %Out0 = bf16[`@c.n`,64] select_lt(%In1, %zero, %zero, %In1);
}
""")

# Reduction along the other axis, to pin down that `dimensions` is honoured.
instr = ops.add_instruction("colsum", ["n"], ["p", "o"])
instr.set_inputs([["d1", ["@a.p"], ["@c.n"]]])
instr.set_outputs([["d1", ["@a.o"], ["1"]]])
instr.add_semantics("""
ENTRY colsum {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %r = bf16[64] reduce_add(%In1), dimensions={0};
    ROOT %Out0 = bf16[1,64] broadcast(%r), dimensions={1};
}
""")

# HBM <-> d1, so a kernel can get data in and out.
instr = ops.add_instruction("ld", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])
instr.set_outputs([["d1", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics("""
ENTRY ld {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")

instr = ops.add_instruction("st", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["d0", ["@a.addr_out"], ["@c.n * 128"]]])
instr.add_semantics("""
ENTRY st {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%a);
}
""")


# Generate programming APIs and test oracle (functional simulator)
generate_oracle(ops)
