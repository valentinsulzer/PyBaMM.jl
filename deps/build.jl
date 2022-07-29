using PyCall
import Conda
using Pkg

if PyCall.conda
    Conda.pip_interop(true)
    Conda.pip("install", "pybamm")
else
    try 
        pyimport("pybamm")
    catch err
        error("Either Install PyBaMM manually by running `pip install pybamm or set ENV[\"PYTHON\"]=\"\" and rebuild pycall, restart julia and rebuild pybamm to automatically install pybamm in the PyCall-provided julia environment.")
    end
end


