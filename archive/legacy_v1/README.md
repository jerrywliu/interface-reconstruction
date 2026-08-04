# Pre-Package Implementation Snapshot

These files predate the current package layout and frozen paper algorithm. They
used bare imports, hard-coded experiment settings, and a separate reconstruction
path. They are retained as a source snapshot, not as an executable public API.

Original locations included:

- `facet.py` and `run_old.py` at the repository root;
- `main/algos/{advection,interface_reconstruction,local_reconstruction,plic,static_interface_reconstruction}.py`;
- `main/structs/strand.py`;
- `util/initialize/initialize_areas_old.py`;
- the non-pytest example scripts formerly stored directly under `test/`;
- the root `test.vtp` sample artifact.

The supported implementation is described in `docs/CODE_STRUCTURE.md`. This
snapshot is deliberately isolated so imports cannot accidentally select it.
