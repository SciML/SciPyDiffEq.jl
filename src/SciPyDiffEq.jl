module SciPyDiffEq

using Reexport
@reexport using DiffEqBase
using PyCall

abstract type SciPyAlgoritm <: DiffEqBase.AbstractODEAlgorithm end
struct RK45 <: SciPyAlgoritm end
struct RK23 <: SciPyAlgoritm end
struct Radau <: SciPyAlgoritm end
struct BDF <: SciPyAlgoritm end
struct LSODA <: SciPyAlgoritm end
struct odeint <: SciPyAlgoritm end

const integrate = PyNULL()

function __init__()
    copy!(integrate, pyimport_conda("scipy.integrate", "scipy", "conda-forge"))
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem,
                            alg::SciPyAlgoritm, timeseries = [], ts = [], ks = [];
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
        sol, fullout = integrate.odeint(f, u0, __saveat,
                                        hmax = dtmax,
                                        rtol = reltol, atol = abstol,
                                        full_output = 1, tfirst = true,
                                        mxstep = maxiters)
        tcur = fullout["tcur"]
        retcode = fullout["tcur"] == __saveat[end] ? :Success : :Failure
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
        retcode = sol["success"] == false ? :Failure : :Success

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

end # module
