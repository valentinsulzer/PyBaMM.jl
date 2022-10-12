# PyBaMM.jl

PyBaMM.jl is a common interface binding for the [PyBaMM](pybamm.org) Python battery modeling package. 
It uses the [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl) interop in order to retrieve the equations from python in a Julia-readable form.

## Installation

PyBaMM will be automatically installed when PyBaMM.jl is installed, via [CondaPkg](https://github.com/cjdoris/CondaPkg.jl)

If you want to bring your own python, or want to use the system Python on linux, you'll need to install your own PyBaMM. Please follow the instructions provided by [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl)

For example, in Linux/Julia do

```bash
pip install pybamm
```

To install the package from source, clone the GitHub repo, then activate:
```julia
] activate .
```

To install as a Julia package:
```julia
] add "https://github.com/tinosulzer/PyBaMM.jl"
```

## Using PyBaMM.jl

See examples in docs folder.

The link from PyBaMM to PyBaMM.jl is one-way and one-time. 
PyBaMM is used to load a model (any PyBaMM model can be used) and parameter values, discretize the model in space, and generate Julia functions to represent the discretized model. From then on, we are entirely in Julia and can use all the tools from the `SciML` ecosystem without having slow calls back to Python.

