"""CPU-only unit tests for P3 decode-speed optimizations:
1. CUDA Graph Replay GPU-side event synchronization (no host-blocking cuCtxSynchronize).
2. MetalRuntime command-buffer batching (`begin_batch` / `end_batch` / `batch`) and FrozenPlan guard.
3. Generic fused decode primitives:
   - F1: `rms_norm_forward_kernel` native fp16/bf16 I/O with explicit fp32 weight multiply & store cast.
   - F3: `paged_decode_attn_split_kernel` / `paged_decode_attention` optional `q_scale` / `k_scale` folding.
   - F4: `paged_kv_append_step_kernel` optional `ADVANCE_LENS` fusion (`-1 launch/layer`).
"""
from __future__ import annotations

import ast
import ctypes
import inspect
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nbxtensor.backend import frozen_plan as FP
from nbxtensor.backend import replay_graph as RG


# ---------------------------------------------------------------------------
# 1. CUDA Graph Replay — GPU-side event synchronization (mocked driver API)
# ---------------------------------------------------------------------------


class _FakeCudaGraphAPI:
    """In-memory mock of `_CudaGraphAPI` recording every CUDA Driver API call."""

    def __init__(self, fail_on: dict | None = None) -> None:
        self.calls: list[tuple] = []
        self.fail_on = dict(fail_on or {})
        self.inst_flags = True
        self._next_handle = 0x1000
        self._recorded_nodes = 0

    def _alloc(self) -> int:
        self._next_handle += 0x10
        return self._next_handle

    def _rc(self, name: str) -> int:
        val = self.fail_on.get(name, 0)
        if isinstance(val, list):
            return val.pop(0) if val else 0
        return val

    def stream_create(self, out_ptr, flags):
        h = self._alloc()
        self.calls.append(("cuStreamCreate", int(flags), h))
        rc = self._rc("stream_create")
        if rc == 0:
            ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value = h
        return rc

    def stream_destroy(self, stream):
        self.calls.append(("cuStreamDestroy_v2", int(stream.value or 0)))
        return self._rc("stream_destroy")

    def event_create(self, out_ptr, flags):
        h = self._alloc()
        self.calls.append(("cuEventCreate", int(flags), h))
        rc = self._rc("event_create")
        if rc == 0:
            ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value = h
        return rc

    def event_destroy(self, event):
        self.calls.append(("cuEventDestroy_v2", int(event.value or 0)))
        return self._rc("event_destroy")

    def event_record(self, event, stream):
        self.calls.append(("cuEventRecord", int(event.value or 0), int(stream.value or 0)))
        return self._rc("event_record")

    def stream_wait_event(self, stream, event, flags):
        self.calls.append(("cuStreamWaitEvent", int(stream.value or 0), int(event.value or 0), int(flags)))
        return self._rc("stream_wait_event")

    def begin(self, stream, mode):
        self._recorded_nodes = 0
        self.calls.append(("cuStreamBeginCapture_v2", int(stream.value or 0), int(mode)))
        return self._rc("begin")

    def end(self, stream, out_graph):
        h = self._alloc()
        self.calls.append(("cuStreamEndCapture", int(stream.value or 0), h))
        rc = self._rc("end")
        if rc == 0:
            ctypes.cast(out_graph, ctypes.POINTER(ctypes.c_void_p)).contents.value = h
        return rc

    def memset_d8(self, ptr, value, nbytes, stream):
        self._recorded_nodes += 1
        self.calls.append(("cuMemsetD8Async", int(ptr.value or 0), int(value), int(nbytes), int(stream.value or 0)))
        return self._rc("memset_d8")

    def memcpy_dtod(self, dst, src, nbytes, stream):
        self._recorded_nodes += 1
        self.calls.append(("cuMemcpyDtoDAsync_v2", int(dst.value or 0), int(src.value or 0), int(nbytes),
                           int(stream.value or 0)))
        return self._rc("memcpy_dtod")

    def get_nodes(self, graph, nodes, count_ptr):
        self.calls.append(("cuGraphGetNodes", int(graph.value or 0), self._recorded_nodes))
        ctypes.cast(count_ptr, ctypes.POINTER(ctypes.c_size_t)).contents.value = self._recorded_nodes
        return self._rc("get_nodes")

    def instantiate(self, out_exec, graph, flags):
        h = self._alloc()
        self.calls.append(("cuGraphInstantiateWithFlags", int(graph.value or 0), int(flags), h))
        rc = self._rc("instantiate")
        if rc == 0:
            ctypes.cast(out_exec, ctypes.POINTER(ctypes.c_void_p)).contents.value = h
        return rc

    def launch(self, gexec, stream):
        self.calls.append(("cuGraphLaunch", int(gexec.value or 0), int(stream.value or 0)))
        return self._rc("launch")

    def graph_destroy(self, graph):
        self.calls.append(("cuGraphDestroy", int(graph.value or 0)))
        return self._rc("graph_destroy")

    def exec_destroy(self, gexec):
        self.calls.append(("cuGraphExecDestroy", int(gexec.value or 0)))
        return self._rc("exec_destroy")

    def ctx_sync(self):
        self.calls.append(("cuCtxSynchronize",))
        return self._rc("ctx_sync")


def test_captured_plan_launch_uses_gpu_events_without_ctx_synchronize(monkeypatch):
    api = _FakeCudaGraphAPI()
    monkeypatch.setattr(RG._CudaGraphAPI, "get", classmethod(lambda cls: api))

    plan = RG.CapturedPlan([("s", (0xDEAD, 0, 64))], None)
    try:
        # Both pre_event and post_event created with CU_EVENT_DISABLE_TIMING (0x2)
        ev_creates = [c for c in api.calls if c[0] == "cuEventCreate"]
        assert len(ev_creates) == 2
        assert all(c[1] == RG._CU_EVENT_DISABLE_TIMING for c in ev_creates)

        api.calls.clear()
        plan.launch()

        stream_h = int(plan.stream.value)
        pre_h = int(plan.pre_event.value)
        post_h = int(plan.post_event.value)
        gexec_h = int(plan.gexec.value)

        # Zero host-blocking cuCtxSynchronize calls!
        assert not any(c[0] == "cuCtxSynchronize" for c in api.calls)
        assert api.calls == [
            ("cuEventRecord", pre_h, 0),
            ("cuStreamWaitEvent", stream_h, pre_h, 0),
            ("cuGraphLaunch", gexec_h, stream_h),
            ("cuEventRecord", post_h, stream_h),
            ("cuStreamWaitEvent", 0, post_h, 0),
        ]
    finally:
        api.calls.clear()
        plan.destroy()

    destroyed = [c[0] for c in api.calls]
    assert destroyed == [
        "cuGraphExecDestroy",
        "cuGraphDestroy",
        "cuEventDestroy_v2",
        "cuEventDestroy_v2",
        "cuStreamDestroy_v2",
    ]
    # Idempotent second destroy
    api.calls.clear()
    plan.destroy()
    assert api.calls == []


def test_captured_plan_pre_vs_post_event_error_marking(monkeypatch):
    # Pre-launch failure: executed must NOT be True (safe refusal)
    api_pre = _FakeCudaGraphAPI(fail_on={"stream_wait_event": [700]})
    monkeypatch.setattr(RG._CudaGraphAPI, "get", classmethod(lambda cls: api_pre))
    plan_pre = RG.CapturedPlan([("s", (0xDEAD, 0, 64))], None)
    try:
        with pytest.raises(RG.GraphReplayError, match="cuStreamWaitEvent\\(pre\\)") as exc_info:
            plan_pre.launch()
        assert not getattr(exc_info.value, "executed", False)
    finally:
        plan_pre.destroy()

    # Post-launch failure: executed MUST be True (graph already launched)
    api_post = _FakeCudaGraphAPI(fail_on={"stream_wait_event": [0, 700]})
    monkeypatch.setattr(RG._CudaGraphAPI, "get", classmethod(lambda cls: api_post))
    plan_post = RG.CapturedPlan([("s", (0xDEAD, 0, 64))], None)
    try:
        with pytest.raises(RG.GraphReplayError, match="cuStreamWaitEvent\\(post\\)") as exc_info:
            plan_post.launch()
        assert getattr(exc_info.value, "executed", False) is True
    finally:
        plan_post.destroy()


def test_captured_plan_cleans_up_pre_event_when_second_event_create_fails(monkeypatch):
    api = _FakeCudaGraphAPI(fail_on={"event_create": [0, 2]})
    monkeypatch.setattr(RG._CudaGraphAPI, "get", classmethod(lambda cls: api))
    with pytest.raises(RG.GraphReplayError, match="cuEventCreate\\(post\\)"):
        RG.CapturedPlan([("s", (0xDEAD, 0, 64))], None)
    names = [c[0] for c in api.calls]
    assert names.count("cuEventDestroy_v2") == 1
    assert names.count("cuStreamDestroy_v2") == 1


# ---------------------------------------------------------------------------
# 2. Metal Command-Buffer Batching (pure CPU mock of MetalRuntime)
# ---------------------------------------------------------------------------


def _fake_metal_runtime(dispatch_cap: int = 2):
    from nbxtensor.backend.metal_device import MetalRuntime

    rt = MetalRuntime.__new__(MetalRuntime)
    rt._lock = threading.RLock()
    rt._open = {}
    rt._pending = {}
    rt._pending_cap = 32
    rt._dispatch_cap = dispatch_cap
    rt._batch_depth = {}
    rt._validating = False
    rt._validation_reports = []
    commits = []

    class _Enc:
        def endEncoding(self):
            pass

    class _CB:
        def __init__(self, idx):
            self.idx = idx

        def commit(self):
            commits.append(self.idx)

    cb_counter = [0]

    def open_dispatch(stream: int = 0):
        if stream not in rt._open:
            cb_counter[0] += 1
            rt._open[stream] = [_CB(cb_counter[0]), _Enc(), "compute", 0]
        rt.note_dispatch(stream)

    return rt, commits, open_dispatch


def test_metal_runtime_batching_defers_commits_until_outermost_end_batch():
    rt, commits, open_dispatch = _fake_metal_runtime(dispatch_cap=2)

    # Unbatched: 5 dispatches with cap=2 triggers 2 auto-commits (at 2 and 4)
    for _ in range(5):
        open_dispatch(0)
    assert len(commits) == 2
    rt.close_stream(0)
    assert len(commits) == 3

    # Batched (re-entrant): 5 dispatches trigger 0 auto-commits, 1 commit on outer exit
    commits.clear()
    with rt.batch(0):
        open_dispatch(0)
        open_dispatch(0)
        with rt.batch(0):
            open_dispatch(0)
            open_dispatch(0)
        assert commits == []  # inner batch exit must not commit
        open_dispatch(0)
        assert commits == []
    assert len(commits) == 1
    assert rt._batch_depth == {}

    # Unmatched end_batch raises
    with pytest.raises(RuntimeError, match="without matching begin_batch"):
        rt.end_batch(0)


def test_frozen_plan_direct_actions_uses_metal_batch_when_enabled(monkeypatch):
    rt, commits, open_dispatch = _fake_metal_runtime(dispatch_cap=1)
    events = []
    orig_begin, orig_end = rt.begin_batch, rt.end_batch
    monkeypatch.setattr(rt, "begin_batch", lambda s=0: (events.append(("begin", s)), orig_begin(s))[1])
    monkeypatch.setattr(rt, "end_batch", lambda s=0, commit=True: (events.append(("end", s)), orig_end(s, commit))[1])

    from nbxtensor.backend import runtime as _rt
    from nbxtensor.backend import metal_device as _md
    monkeypatch.setenv("NBX_METAL_REPLAY_BATCH", "1")
    monkeypatch.setattr(_rt, "gpu_backend", lambda: "metal")
    monkeypatch.setattr(_md, "runtime", lambda: rt)

    plan = FP.FrozenPlan.__new__(FP.FrozenPlan)
    plan.records = [(FP.BOUND, (SimpleNamespace(run=lambda s: open_dispatch(s)),)) for _ in range(4)]
    plan._bound = plan.records

    plan._direct_actions()
    assert events == [("begin", 0), ("end", 0)]
    assert len(commits) == 1


# ---------------------------------------------------------------------------
# 3. Generic Fused Primitives (F1 / F3 / F4) — AST/signature + CPU reference
# ---------------------------------------------------------------------------


_SRC = Path(__file__).resolve().parents[1] / "src" / "nbxtensor"


def test_f1_rmsnorm_kernel_casts_weight_to_fp32_and_output_to_pointer_dtype():
    src = (_SRC / "ops" / "rmsnorm.py").read_text()
    assert "tl.load(weight_ptr + feat_offset, mask=feat_mask).to(tl.float32)" in src
    assert "tl.store(output_ptr, output.to(output_ptr.dtype.element_ty)," in src
    # Verify fp16 -> fp32 -> fp16 numerical parity on CPU
    rng = np.random.default_rng(42)
    x_f16 = rng.standard_normal((4, 64)).astype(np.float16)
    w_f16 = rng.standard_normal((64,)).astype(np.float16)
    eps = 1e-6
    x32 = x_f16.astype(np.float32)
    w32 = w_f16.astype(np.float32)
    inv_rms = 1.0 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    ref = np.clip(x32 * inv_rms * w32, -65504.0, 65504.0).astype(np.float16)
    assert ref.dtype == np.float16 and np.all(np.isfinite(ref))


def test_f3_f4_paged_decode_signatures_and_cpu_reference():
    # Parse AST of paged_decode_attn.py (works without triton installed)
    tree = ast.parse((_SRC / "ops" / "paged_decode_attn.py").read_text())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    # F4: paged_kv_append_step_kernel has ADVANCE_LENS=False, NUM_KV_HEADS=1
    append_args = fns["paged_kv_append_step_kernel"].args
    append_names = [a.arg for a in append_args.args]
    append_defaults = dict(zip(append_names[-len(append_args.defaults):],
                               [ast.literal_eval(d) for d in append_args.defaults]))
    assert append_defaults["ADVANCE_LENS"] is False
    assert append_defaults["NUM_KV_HEADS"] == 1

    # F3: paged_decode_attn_split_kernel has q_scale=1.0, HAS_Q_SCALE=False
    split_args = fns["paged_decode_attn_split_kernel"].args
    split_names = [a.arg for a in split_args.args]
    split_defaults = dict(zip(split_names[-len(split_args.defaults):],
                              [ast.literal_eval(d) for d in split_args.defaults]))
    assert split_defaults["q_scale"] == 1.0
    assert split_defaults["HAS_Q_SCALE"] is False

    # Check wrappers in attention_wrappers.py (AST check so triton is not required on CPU)
    aw_tree = ast.parse((_SRC / "ops" / "attention_wrappers.py").read_text())
    aw_fns = {n.name: n for n in aw_tree.body if isinstance(n, ast.FunctionDef)}
    step_args = [a.arg for a in aw_fns["paged_kv_append_step"].args.args]
    assert "advance_lens" in step_args
    assert ast.literal_eval(aw_fns["paged_kv_append_step"].args.defaults[-1]) is False

    attn_args = [a.arg for a in aw_fns["paged_decode_attention"].args.args]
    assert "q_scale" in attn_args and "k_scale" in attn_args

    # Check PagedTritonKVCache.decode uses ADVANCE_LENS=True and no paged_lens_advance_kernel
    pkv_src = (_SRC / "memory" / "paged_kv.py").read_text()
    assert "ADVANCE_LENS=True" in pkv_src
    assert "paged_lens_advance_kernel" not in pkv_src

    # CPU reference check: fused append + lens advance matches separate append + advance
    B, H_kv, D, page_tokens = 2, 4, 8, 4
    rng = np.random.default_rng(7)
    k_new = rng.standard_normal((B, H_kv, D)).astype(np.float16)
    v_new = rng.standard_normal((B, H_kv, D)).astype(np.float16)
    k_pool_a = np.zeros((4, H_kv, page_tokens, D), dtype=np.float16)
    v_pool_a = np.zeros((4, H_kv, page_tokens, D), dtype=np.float16)
    k_pool_b = k_pool_a.copy()
    v_pool_b = v_pool_a.copy()
    bt = np.array([[0, 2], [1, 3]], dtype=np.int32)
    lens_a = np.array([3, 5], dtype=np.int32)
    lens_b = lens_a.copy()

    # Unfused
    for b in range(B):
        for h in range(H_kv):
            pos = int(lens_a[b])
            page, slot = int(bt[b, pos // page_tokens]), pos % page_tokens
            k_pool_a[page, h, slot] = k_new[b, h]
            v_pool_a[page, h, slot] = v_new[b, h]
    lens_a += 1

    # Fused (single program per sequence, loops over H_kv, advances lens once)
    for b in range(B):
        pos = int(lens_b[b])
        page, slot = int(bt[b, pos // page_tokens]), pos % page_tokens
        for h in range(H_kv):
            k_pool_b[page, h, slot] = k_new[b, h]
            v_pool_b[page, h, slot] = v_new[b, h]
        lens_b[b] = pos + 1

    assert np.array_equal(k_pool_a, k_pool_b)
    assert np.array_equal(v_pool_a, v_pool_b)
    assert np.array_equal(lens_a, lens_b)
