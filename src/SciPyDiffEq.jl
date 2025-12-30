module SciPyDiffEq

using Reexport: @reexport
using DiffEqBase: DiffEqBase, ReturnCode
@reexport using DiffEqBase
using SciMLBase: SciMLBase, ODEProblem
using CommonSolve: solve
using PyCall: PyCall, PyNULL, pyimport, pyimport_conda
using PrecompileTools: PrecompileTools, @compile_workload, @setup_workload

"""
    SciPyAlgorithm

Abstract supertype for all SciPy ODE solver algorithms.
"""
abstract type SciPyAlgorithm <: DiffEqBase.AbstractODEAlgorithm end

"""
    RK45()

Explicit Runge-Kutta method of order 5(4) from SciPy. This is the Dormand-Prince
method, suitable for non-stiff problems.

See also: [`RK23`](@ref), [`Radau`](@ref), [`BDF`](@ref), [`LSODA`](@ref)
"""
struct RK45 <: SciPyAlgorithm end

"""
    RK23()

Explicit Runge-Kutta method of order 3(2) from SciPy. Suitable for non-stiff
problems with lower accuracy requirements.

See also: [`RK45`](@ref), [`Radau`](@ref), [`BDF`](@ref), [`LSODA`](@ref)
"""
struct RK23 <: SciPyAlgorithm end

"""
    Radau()

Implicit Runge-Kutta method of the Radau IIA family of order 5 from SciPy.
Suitable for stiff problems.

See also: [`RK45`](@ref), [`RK23`](@ref), [`BDF`](@ref), [`LSODA`](@ref)
"""
struct Radau <: SciPyAlgorithm end

"""
    BDF()

Implicit multi-step variable-order (1 to 5) method based on backward
differentiation formulas from SciPy. Suitable for stiff problems.

See also: [`RK45`](@ref), [`RK23`](@ref), [`Radau`](@ref), [`LSODA`](@ref)
"""
struct BDF <: SciPyAlgorithm end

"""
    LSODA()

Adams/BDF method with automatic stiffness detection and switching from SciPy.
Originally from the FORTRAN library ODEPACK.

See also: [`RK45`](@ref), [`RK23`](@ref), [`Radau`](@ref), [`BDF`](@ref), [`odeint`](@ref)
"""
struct LSODA <: SciPyAlgorithm end

"""
    odeint()

SciPy's `odeint` function, which wraps the FORTRAN solver LSODA from ODEPACK.
This is the legacy SciPy interface. Note that `saveat` is required when using
this algorithm.

See also: [`LSODA`](@ref), [`RK45`](@ref), [`BDF`](@ref)
"""
struct odeint <: SciPyAlgorithm end

const integrate = PyNULL()

function __init__()
    copy!(integrate, pyimport_conda("scipy.integrate", "scipy", "conda-forge"))
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem,
        alg::SciPyAlgorithm, timeseries = [], ts = [], ks = [];
        dense = true, dt = nothing,
        dtmax = abs(prob.tspan[2] - prob.tspan[1]),
        dtmin = eps(eltype(prob.tspan)), save_everystep = false,
        saveat = eltype(prob.tspan)[], timeseries_errors = true,
        reltol = 1e-3, abstol = 1e-6, maxiters = 10_000,
        kwargs...)
    p = prob.p
    tspan = prob.tspan
    u0 = prob.u0

    if DiffEqBase.isinplace(prob)
        f = function (t, u)
            du = similar(u)
            prob.f(du, u, p, t)
            du
        end
    else
        f = (t, u) -> prob.f(u, p, t)
    end

    _saveat = isempty(saveat) ? nothing : saveat
    if _saveat isa Array
        __saveat = _saveat
    elseif _saveat isa Number
        __saveat = Array(tspan[1]:_saveat:tspan[2])
    elseif _saveat isa Nothing
        if save_everystep
            __saveat = nothing
        else
            __saveat = [tspan[1], tspan[2]]
        end
    else
        __saveat = Array(_saveat)
    end

    if alg isa odeint
        __saveat === nothing && error("saveat is required for odeint!")
        sol,
        fullout = integrate.odeint(f, u0, __saveat,
            hmax = dtmax,
            rtol = reltol, atol = abstol,
            full_output = 1, tfirst = true,
            mxstep = maxiters)
        tcur = fullout["tcur"]
        retcode = fullout["tcur"] == __saveat[end] ? ReturnCode.Success : ReturnCode.Failure
        ts = __saveat
        y = sol

        if u0 isa AbstractArray
            timeseries = Vector{typeof(u0)}(undef, length(ts))
            for i in 1:length(ts)
                timeseries[i] = @view y[i, :]
            end
        else
            timeseries = y
        end

    else
        sol = integrate.solve_ivp(f, tspan, u0,
            first_step = dt,
            max_step = dtmax,
            rtol = reltol, atol = abstol,
            t_eval = __saveat,
            dense_output = dense,
            method = string(alg)[13:(end - 2)])
        ts = sol["t"]
        y = sol["y"]
        retcode = sol["success"] == false ? ReturnCode.Failure : ReturnCode.Success

        if u0 isa AbstractArray
            timeseries = Vector{typeof(u0)}(undef, length(ts))
            for i in 1:length(ts)
                timeseries[i] = @view y[:, i]
            end
        else
            timeseries = y
        end
    end

    if !(alg isa odeint) && dense
        _interp = PyInterpolation(sol["sol"])
    else
        _interp = DiffEqBase.LinearInterpolation(ts, timeseries)
    end

    DiffEqBase.build_solution(prob, alg, ts, timeseries,
        interp = _interp,
        dense = dense,
        retcode = retcode,
        timeseries_errors = timeseries_errors)
end

struct PyInterpolation{T} <: DiffEqBase.AbstractDiffEqInterpolation
    pydense::T
end
function (PI::PyInterpolation)(t, idxs, deriv, p, continuity)
    if idxs !== nothing
        return PI.pydense(t)[idxs]
    else
        return PI.pydense(t)
    end
end
DiffEqBase.interp_summary(::PyInterpolation) = "Interpolation from SciPy"

@setup_workload begin
    # Define a simple test problem for precompilation
    function _precompile_f(u, p, t)
        du1 = 10.0 * (u[2] - u[1])
        du2 = u[1] * (28.0 - u[3]) - u[2]
        du3 = u[1] * u[2] - (8.0 / 3.0) * u[3]
        [du1, du2, du3]
    end
    _precompile_u0 = [1.0, 0.0, 0.0]
    _precompile_tspan = (0.0, 1.0)

    @compile_workload begin
        # Only precompile if scipy is available
        # This check is needed because PyCall may not have scipy at precompile time
        try
            _integrate = pyimport("scipy.integrate")
            prob = ODEProblem(_precompile_f, _precompile_u0, _precompile_tspan)
            # Precompile the most commonly used solver (RK45)
            solve(prob, RK45())
        catch
            # scipy not available at precompile time, skip workload
        end
    end
end

end # module
