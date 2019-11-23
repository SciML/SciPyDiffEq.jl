using SciPyDiffEq
using Test

function lorenz(u,p,t)
 du1 = 10.0(u[2]-u[1])
 du2 = u[1]*(28.0-u[3]) - u[2]
 du3 = u[1]*u[2] - (8/3)*u[3]
 [du1, du2, du3]
end
u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(lorenz,u0,tspan)
sol = solve(prob,SciPyDiffEq.RK45())
sol = solve(prob,SciPyDiffEq.RK23())
sol = solve(prob,SciPyDiffEq.Radau())
sol = solve(prob,SciPyDiffEq.BDF())
sol = solve(prob,SciPyDiffEq.LSODA())

function lorenz(du,u,p,t)
 du[1] = 10.0(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
end
u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(lorenz,u0,tspan)
sol = solve(prob,SciPyDiffEq.RK45())
sol(4.0)
sol = solve(prob,SciPyDiffEq.RK23())
sol(4.0)
sol = solve(prob,SciPyDiffEq.Radau())
sol(4.0)
sol = solve(prob,SciPyDiffEq.BDF())
sol(4.0)
sol = solve(prob,SciPyDiffEq.LSODA())
sol(4.0)

#using Plots; plot(sol)
