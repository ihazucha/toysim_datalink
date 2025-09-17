# ToySim - Data-link

For common definition of:

* Data structures
* IPC communication
* Network communication (simulation/vehicle <-> ToySim UI) 

## TODO

* Fix MPMCQueue topics - currently only messages with no topic can be received
* Make data structures and communication platform-agnostic (currently Python objects) to use [MCAP](https://mcap.dev) format.

## Inspiration
* https://github.com/commaai/msgq/
* https://github.com/commaai/openpilot/tree/master/cereal
