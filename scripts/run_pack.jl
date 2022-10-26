using PyBaMM
pybamm = pyimport("pybamm")


Np = 1
Ns = 2

model = pybamm.lithium_ion.DFN(name="DFN")

pack_sim = build_pack(Np, Ns, model) #pack_sim: ODE Problem

alg = QBDF()

solve(pack_sim, alg(linsolve = KLUFactorization(), concrete_jac = true))