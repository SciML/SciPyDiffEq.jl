# API

```@docs
SciPyDiffEq.SciPyAlgorithm
SciPyDiffEq.RK45
SciPyDiffEq.RK23
SciPyDiffEq.Radau
SciPyDiffEq.BDF
SciPyDiffEq.LSODA
SciPyDiffEq.odeint
```

## Reexported SciML common interface

`using SciPyDiffEq` also brings in the parts of the SciML common interface needed to
build an ODE problem, solve it, and inspect the result, so they do not have to be
imported separately. SciPyDiffEq does not define these names -- they are owned and
documented by [SciMLBase](https://docs.sciml.ai/SciMLBase/stable/) (and `solve` by
[CommonSolve](https://github.com/SciML/CommonSolve.jl)), and that is where their
documentation lives:

  - Problems: `ODEProblem`, `EnsembleProblem`
  - Functions: `ODEFunction`
  - Solutions: `ODESolution`, `EnsembleSolution`, `EnsembleSummary`, `DEStats`
  - Ensemble algorithms: `EnsembleSerial`, `EnsembleThreads`, `EnsembleDistributed`,
    `EnsembleSplitThreads`, and the `EnsembleAnalysis` module
  - Solving: `solve`, `remake`
  - Return status: `ReturnCode`, `successful_retcode`
  - `NullParameters`

Note that the SciPy algorithms above are public but *not* exported, so they are still
written qualified: `SciPyDiffEq.RK45()`.

Anything else from SciMLBase must be imported from SciMLBase directly. Three groups are
deliberately absent:

  - **DAE, SDE, DDE and every other non-ODE problem type.** `SciMLBase.__solve` is
    defined here only for `AbstractODEProblem`.
  - **Callbacks.** Nothing is passed through to SciPy, so `ContinuousCallback` and
    friends are not part of this package's surface.
  - **The integrator interface** (`init`, `step!`, `solve!`, `reinit!`, ...).
    SciPyDiffEq implements `SciMLBase.__solve` only; it has no integrator.
