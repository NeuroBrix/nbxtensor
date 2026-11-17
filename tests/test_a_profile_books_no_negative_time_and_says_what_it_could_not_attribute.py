"""The CUDA kernel-time session books every instant of its timeline once: rows and the time between launches never
overlap, no class is negative, and the wall time no interval holds is ONE printed figure, `unattributed`. No GPU: the
device is the simulated stream of the session's own test.

What this test does if the code were wrong:
- the overlap cell fails when launches on two streams that run together, and a copy on top of them, are each booked
  their whole interval (the same instants counted in two classes), or when the classes and `unattributed` do not add
  up to the wall exactly;
- the error cell fails when an end stamped before its start is skipped or booked as a negative time instead of
  raised with both stamps.
"""
from __future__ import annotations

import pytest

from nbxtensor.backend import cuda_profile

from test_a_cuda_kernel_time_session_times_each_launch_between_two_events import _launch, _session


def _classes(table) -> float:
    return table.gpu_ms + table.cadence["between_ms"]


def test_overlapping_intervals_are_booked_once_and_the_classes_with_the_unattributed_make_the_wall():
    dev, session = _session()
    _launch(dev, session, "gemv", 1_000_000, stream=0)
    _launch(dev, session, "gemv", 1_000_000, stream=7)          # runs together with the first
    _launch(dev, session, "copy host to device", 800_000, stream=3)
    dev.spend(300_000)                                           # host time that holds no interval
    _launch(dev, session, "gemv", 500_000, stream=0)
    dev.spend(200_000)
    table = session.close()
    c = table.cadence
    assert all(r[1] >= 0 for r in table.rows.values()) and c["between_ms"] >= 0 and c["unattributed_ms"] >= 0
    assert _classes(table) + c["unattributed_ms"] == pytest.approx(table.wall_s * 1e3, abs=1e-9)
    assert table.gpu_ms < 1.0 + 1.0 + 0.8 + 0.5                  # the shared instants are not summed twice
    assert "unattributed" in table.format()


def test_an_end_stamped_before_its_start_is_an_error_that_names_both_stamps():
    dev, session = _session()
    _launch(dev, session, "gemv", 500_000)
    mark = session.begin(0)
    dev.work(500_000, 0)
    mark[0].end(mark, "broken")
    dev.stamp[mark[4]] = dev.stamp[mark[3]] - 100_000            # the device's end before its start
    with pytest.raises(RuntimeError, match="end of 'broken'.*before its start"):
        session.close()
