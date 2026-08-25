"""Numerics for the ops wired up in oracle_visitor.py.

Each case runs `ld -> op -> st` on the Ops accelerator and checks the result
against a NumPy reference. bf16 has an 8-bit significand, so anything that
rounds to bf16 is checked to a 2^-8 relative tolerance; the exactly-representable
cases (max, select, sums of the same values) are checked exactly.
"""

import os

import numpy as np
import jax.numpy as jnp
import pytest

from conftest import EXAMPLES_DIR, generate_and_import_oracle

N = 4          # rows
LANES = 64
BYTES = N * LANES * 2
BF16_EPS = 2.0 ** -8


@pytest.fixture(scope="module")
def oracle():
    """Generate the Ops oracle into a temp dir and import it."""
    return generate_and_import_oracle(os.path.join(EXAMPLES_DIR, "Ops"))


def run(oracle, op, data, rows_out=N):
    kernel, api = oracle

    @kernel(hbm=2 * BYTES,
            input=[{'addr': 0, 'shape': (N, LANES), 'dtype': jnp.bfloat16}],
            output=[{'addr': BYTES, 'shape': (rows_out, LANES), 'dtype': jnp.bfloat16}])
    def k():
        api.ld(n=N, addr_in=0, addr_out=0)
        getattr(api, op)(n=N, p=0, o=N)
        api.st(n=rows_out, addr_in=N, addr_out=BYTES)

    k('fsim-compile')()
    outputs, _ = k('fsim')(jnp.array(data, dtype=jnp.bfloat16))
    return np.asarray(outputs[0]).astype(np.float32)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    return {
        "mixed": np.asarray(jnp.array(rng.standard_normal((N, LANES)) * 3.0,
                                      dtype=jnp.bfloat16)).astype(np.float32),
        "positive": np.asarray(jnp.array(np.abs(rng.standard_normal((N, LANES))) + 0.25,
                                         dtype=jnp.bfloat16)).astype(np.float32),
    }


def test_reduce_max(oracle, data):
    """Generic `reduce` with the max_bf16 combiner -- exact, no arithmetic."""
    x = data["mixed"]
    got = run(oracle, "redmax", x)
    want = np.broadcast_to(x.max(axis=1, keepdims=True), (N, LANES))
    np.testing.assert_array_equal(got, want)


def test_constant_reciprocal(oracle, data):
    """`constant` naming an integer literal, consumed by divide."""
    x = data["positive"]
    got = run(oracle, "recip", x)
    np.testing.assert_allclose(got, 1.0 / x, rtol=BF16_EPS)


def test_float_constant(oracle, data):
    """A float literal must not be floored to 0 on the way through."""
    x = data["mixed"]
    got = run(oracle, "halve", x)
    np.testing.assert_array_equal(got, x * 0.5)   # exact: halving only shifts the exponent


def test_exp_alias(oracle, data):
    x = data["mixed"]
    got = run(oracle, "expo", x)
    np.testing.assert_allclose(got, np.exp(x), rtol=BF16_EPS)


def test_select_lt(oracle, data):
    """relu via select_lt -- exact, it only ever selects an input value."""
    x = data["mixed"]
    got = run(oracle, "relu", x)
    np.testing.assert_array_equal(got, np.maximum(x, 0.0))


def test_broadcast_honours_dimensions(oracle, data):
    """reduce along dim 0 then broadcast along dim 1.

    This is the case that used to silently pick up `dim` from the preceding
    reduce instead of reading the attribute.
    """
    x = data["positive"]
    got = run(oracle, "colsum", x, rows_out=1)
    want = x.sum(axis=0)[None, :]
    np.testing.assert_allclose(got, want, rtol=4 * BF16_EPS)
