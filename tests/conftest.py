"""Shared test utilities for TAIDL-TO oracle tests"""

import os
import sys
import shutil
import tempfile
import importlib
import importlib.util

import numpy as np
import jax
import jax.numpy as jnp


EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def load_bf16_matrix(path, shape):
    np_uint8 = np.fromfile(path, dtype=np.uint8)
    np_uint8 = np_uint8.reshape(shape[0], shape[1], 2)
    j_uint8 = jnp.array(np_uint8, dtype=jnp.uint8)
    return jax.lax.bitcast_convert_type(j_uint8, jnp.bfloat16)


def _purge_oracle_modules():
    """Drop any previously-imported `oracle` package.

    Every generated target lays its API down as a package literally named
    `oracle`, so importing a second target finds the first one still cached in
    sys.modules and silently hands back the wrong accelerator's API. Clearing
    the cache is what lets more than one target be exercised in a single run.
    """
    for name in [n for n in sys.modules if n == "oracle" or n.startswith("oracle.")]:
        del sys.modules[name]


def generate_and_import_oracle(example_dir, target=None, spec_name=None):
    """Generate an oracle into a temp dir and return (kernel, api).

    `example_dir` holds `<Target>.py`; the target name defaults to the directory
    name, which is also what the generator names the output under `targets/`.
    """
    if target is None:
        target = os.path.basename(os.path.normpath(example_dir))
    if spec_name is None:
        spec_name = target + ".py"

    work_dir = tempfile.mkdtemp()
    shutil.copy(os.path.join(example_dir, spec_name), work_dir)
    os.chdir(work_dir)

    spec = importlib.util.spec_from_file_location(
        target + "_spec", os.path.join(work_dir, spec_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    oracle_dir = os.path.join(work_dir, "targets", target)
    _purge_oracle_modules()
    sys.path.insert(0, oracle_dir)

    oracle_decorator = importlib.import_module("oracle.decorator")
    oracle_api = importlib.import_module("oracle.api")
    oracle_decorator.set_simulation_backend('CPU')

    return oracle_decorator.kernel, oracle_api


def run_kernel(kernel, api, kernel_module_name, assembly_dir, data_dir):
    """Run a kernel and return max diff against golden"""
    sys.path.insert(0, assembly_dir)
    kernel_module = importlib.import_module(kernel_module_name)
    qkv_kernel = kernel_module.qkv(kernel, api)
    qkv_kernel('fsim-compile')()

    Q = load_bf16_matrix(os.path.join(data_dir, "Q.dat"), (64, 64))
    K = load_bf16_matrix(os.path.join(data_dir, "K.dat"), (64, 64))
    V = load_bf16_matrix(os.path.join(data_dir, "V.dat"), (64, 64))

    outputs, _ = qkv_kernel('fsim')(Q, K, V)
    golden = load_bf16_matrix(os.path.join(data_dir, "attention.dat"), (64, 64))

    return float(jnp.max(jnp.abs(outputs[0] - golden)))
