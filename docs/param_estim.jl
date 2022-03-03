using OrdinaryDiffEq

function f(du,u,p,t)
    du[1] = dx = p[1]*u[1] - u[1]*u[2]
    du[2] = dy = -3*u[2] + u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]
prob = ODEProblem(f,u0,tspan,p)

sol = solve(prob,Tsit5())
t = collect(range(0,stop=10,length=200))
data = Array([(p[1] * sol(t[i])[1] + .01randn()) for i in 1:length(t)])

using DiffEqParamEstim
function loss(sol)
    p = sol.prob.p
    sumsq = 0.0
    for i in 1:length(t)
        sumsq += (data[i] - p[1]*sol(t[i])[1])^2
    end
    sumsq
end
