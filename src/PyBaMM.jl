module PyBaMM

using PyCall

py"""
import pybamm
import numpy as np
"""

include("variables.jl")

export get_variable

end # module