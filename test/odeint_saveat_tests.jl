using SciPyDiffEq
using SciPyDiffEq: ODEProblem, solve
using SciMLBase: successful_retcode
using Test

f(u, p, t) = 1.01 * u
prob = ODEProblem(f, [1.0], (0.0, 1.0))
sol = solve(prob, SciPyDiffEq.odeint(); saveat = 0.1)
@test successful_retcode(sol)
