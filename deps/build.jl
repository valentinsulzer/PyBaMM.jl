using PyCall
import Conda
using Pkg

ENV["PYTHON"] = ""
Pkg.build("PyCall")
Conda.pip_interop(true)
Conda.pip("install", "git+https://github.com/pybamm-team/pybamm.git@issue-1129-julia")
