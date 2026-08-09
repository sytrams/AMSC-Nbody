# Test layout

- `unit/cpu`: fast tests for host-only code.
- `unit/cuda`: focused CUDA component tests.
- `integration/cuda`: tests spanning multiple CUDA tree stages.
- `integration/metal`: reserved for headless Metal integration tests.
- `support`: fixtures and helpers shared by test sources.
- `data`: only small, deterministic fixtures.
- `scripts`: explicit slow or diagnostic test runners.

Test suite names identify the component under test, and test names
describe behavior. Test identifiers do not contain underscores.
Randomized cases use fixed seeds.

CUDA tests derive from `CudaTest`. On a machine without a usable CUDA
device, the fixture reports the cases as skipped rather than failed.

The files in `scripts/legacy` are retained temporarily as migration
references. The CMake and CTest commands in the project README are the
supported way to compile and run the reorganized tests.
