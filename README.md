# ToySim Data-link

Common definition of:

- Data structures
- IPC communication (between modules)
- Network communication (simulation/car <-> ToySim UI)

## TODO

### Short

1. Add SPMCQueue IPC variant for Linux (supports IPC using shared memory)
2. Unify network protocol for simulation and car
3. Refine Data structures to be more client type (sim/car) agnostic and clearly distinguishable (add header)

### Long

1. For production-grade IPC solution check:
  - https://github.com/commaai/msgq/
  - https://github.com/commaai/openpilot/tree/master/cereal
  - https://capnproto.org/capnp-tool.html