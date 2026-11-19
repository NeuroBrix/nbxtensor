"""Importing this module routes Triton launches through nbxtensor's launcher (`nbxtensor.backend.launcher.install`).

Every module of this package that defines a kernel imports it, beside its own `import triton`: the routing of launch
sites belongs with them, as `wrappers` says of its own. Before, only `wrappers`, `token_bitmask` and `kv_host_pages`
installed the seam, so a process whose first kernel came from another module (the chunked gated delta rule imported
alone) launched through upstream Triton, which on Apple refuses an NBXTensor ("Arg 0: expected an MPS tensor", Mac M4
Pro, 2026-11-19). It is not installed by `nbxtensor.ops` itself: that would pull Triton and numpy into every light
module of the package, the engine's stdlib-only seams among them. `install()` is idempotent, honours
NBX_LAUNCHER=triton (the differential arm) and returns False without Triton."""
from nbxtensor.backend.launcher import install as _install

_install()
