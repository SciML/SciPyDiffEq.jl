using SciPyDiffEq
using SciPyDiffEq: ODEProblem, solve
using SciMLBase: successful_retcode
using Test

function lorenz(u, p, t)
    du1 = 10.0(u[2] - u[1])
    du2 = u[1] * (28.0 - u[3]) - u[2]
    du3 = u[1] * u[2] - (8 / 3) * u[3]
    return [du1, du2, du3]
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz, u0, tspan)
sol = solve(prob, SciPyDiffEq.RK45())
@test successful_retcode(sol)
sol = solve(prob, SciPyDiffEq.RK23())
@test successful_retcode(sol)
sol = solve(prob, SciPyDiffEq.Radau())
@test successful_retcode(sol)
sol = solve(prob, SciPyDiffEq.BDF())
@test successful_retcode(sol)
sol = solve(prob, SciPyDiffEq.LSODA())
@test successful_retcode(sol)
sol = solve(prob, SciPyDiffEq.odeint())
@test successful_retcode(sol)

function lorenz(du, u, p, t)
    du[1] = 10.0(u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    return du[3] = u[1] * u[2] - (8 / 3) * u[3]
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz, u0, tspan)
sol = solve(prob, SciPyDiffEq.RK45())
@test successful_retcode(sol)
sol(4.0)
sol = solve(prob, SciPyDiffEq.RK23())
@test successful_retcode(sol)
sol(4.0)
sol = solve(prob, SciPyDiffEq.Radau())
@test successful_retcode(sol)
sol(4.0)
sol = solve(prob, SciPyDiffEq.BDF())
@test successful_retcode(sol)
sol(4.0)
sol = solve(prob, SciPyDiffEq.LSODA())
@test successful_retcode(sol)
sol(4.0)
sol = solve(prob, SciPyDiffEq.odeint())
@test successful_retcode(sol)
sol(4.0)

#using Plots; plot(sol,vars=(1,2,3))
