# SciPyDiffEq.jl

SciPyDiffEq.jl exposes SciPy's ordinary differential equation solvers through the
SciML common solve interface. Construct problems with `SciMLBase` and call
`CommonSolve.solve` with an algorithm from this package.

```julia
using CommonSolve: solve
using SciMLBase: ODEProblem
using SciPyDiffEq: RK45

prob = ODEProblem((u, p, t) -> -u, 1.0, (0.0, 1.0))
sol = solve(prob, RK45())
```
