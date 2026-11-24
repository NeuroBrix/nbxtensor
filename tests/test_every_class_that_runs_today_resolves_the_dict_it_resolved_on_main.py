"""Nothing changes for a class that runs today: every compute capability a device file lists (its
`compute_capability` and the minors of its `memory.by_capability`) resolves, through the library's own path
(`_configs._profiles_for`: exact file, major base with its `by_capability` figures, declared prefix), to the device
dict and shared-memory budget it resolved to on library main 50597afcc7a3.

Main's dicts are the fixture `_device_dicts_on_main_50597afcc7a3.json`, written by `_record_device_dicts_on_main.py`
(beside this file) from a worktree of that commit as it stands. No card: resolution reads the device files only.

No difference is allowed: every capability's budget and dict must be identical to main's, byte for byte as JSON
(8.6 resolves to ampere.yml with its memory.by_capability "8.6" figures, 12.x to blackwell_consumer.yml, as on main).
A future device-file change that moves a value a listed capability reads must name and justify its difference here.

What it catches: a device file given to a capability that resolved elsewhere (422a351 gave 8.6 a file of its own and
so dropped ampere.yml's block sizes, SDPA thresholds and three memory values from its resolution) - red."""
import json
from pathlib import Path

import pytest

from _record_device_dicts_on_main import listed_capabilities, resolve

FIXTURE = Path(__file__).with_name("_device_dicts_on_main_50597afcc7a3.json")
MAIN = json.loads(FIXTURE.read_text())["capabilities"]


def test_the_fixture_was_recorded_on_main_for_every_capability_listed_today():
    """Every capability a device file lists today is in main's fixture, and each resolved to a dict on main."""
    listed = set(listed_capabilities())
    assert listed == set(MAIN), sorted(listed ^ set(MAIN))
    unresolved = {c for c in listed if (MAIN.get(c) or {}).get("profile") is None}
    assert not unresolved, f"listed today, no resolution on main: {sorted(unresolved)}"


@pytest.mark.parametrize("capability", sorted(MAIN))
def test_the_capability_resolves_the_dict_it_resolved_on_main(capability):
    """The capability's shared-memory budget and device dict are main's, identical."""
    now, then = resolve(capability), MAIN[capability]
    assert now["budget"] == then["budget"], (capability, then["budget"], now["budget"])
    assert now["profile"] == then["profile"], capability
