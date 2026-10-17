# SciPyDiffEq.jl

SciPyDiffEq.jl exposes SciPy's ordinary differential equation solvers through the
SciML common solve interface. `using SciPyDiffEq` brings the common interface --
`ODEProblem`, `solve` and the solution types -- into scope along with the algorithms;
see the [API page](api.md) for the full reexported list.

```julia
using SciPyDiffEq

prob = ODEProblem((u, p, t) -> -u, 1.0, (0.0, 1.0))
sol = solve(prob, SciPyDiffEq.RK45())
```
