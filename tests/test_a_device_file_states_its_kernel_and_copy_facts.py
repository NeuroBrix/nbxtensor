"""First-launch step 3: the kernel and copy facts are READ from the library's device file (reset point 6), not measured
at a first launch. Each file that states them gives them through `device_facts.measured_facts`, with their source."""
import pathlib

import yaml

import nbxtensor
from nbxtensor.backend import device_facts

DEVICES = pathlib.Path(nbxtensor.__file__).parent / "backend" / "devices"


def _file(rel):
    return yaml.safe_load((DEVICES / rel).read_text())


def test_volta_states_its_kernel_and_copy_facts():
    m = device_facts.measured_facts(_file("nvidia/volta.yml"))
    assert m["matmul_tflops"] == {"float16": 42.3, "float32": 13.5} and m["memory_gbps"] == 789
    assert m["expert_copy"] == {"block": 1024, "warps": 16, "gbps": 5.5}
    assert set(m["gemv_gbps"]) == {"triton", "compiled"} and m["source"].startswith("measured")


def test_the_m4_pro_states_its_kernel_and_copy_facts():
    m = device_facts.measured_facts(_file("apple/apple_m4_pro.yml"))
    assert m["matmul_tflops"]["bfloat16"] == 5.9 and m["h2d_pageable_gbps"] == 66.3 and "source" in m


def test_a_file_without_the_block_states_none():
    assert device_facts.measured_facts({"memory": {}}) == {}


def test_only_kernel_and_copy_facts_are_given():
    m = device_facts.measured_facts({"measured": {"memory_gbps": 1, "weight_load_gbps": {}, "source": "s"}})
    assert m == {"memory_gbps": 1, "source": "s"}
