# PyBaMM.jl

PyBaMM.jl is a common interface binding for the [PyBaMM](pybamm.org) Python battery modeling package. 
It uses the [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) interop in order to retrieve the equations from python in a Julia-readable form.

## Installation

If you're using MacOS or Windows and have PyCall set to default settings, PyBaMM will be automatically installed when PyBaMM.jl is installed via Pkg. To automatically install PyBaMM on linux, just set ENV["PYTHON"]="", build PyCall.jl, and then rebuild PyBaMM.jl.

If you want to bring your own python, or want to use the system Python on linux, you'll need to install your own PyBaMM. 
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

