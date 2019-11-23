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

const integrate = PyNULL()

function __init__()
  copy!(integrate, pyimport_conda("scipy.integrate", "scipy", "conda-forge"))
end

function DiffEqBase.__solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::SciPyAlgoritm,timeseries=[],ts=[],ks=[];
    dense = true, dt = nothing, dtmax = abs(prob.tspan[2] - prob.tspan[1]),
    dtmin = eps(eltype(prob.tspan)),
    saveat=eltype(prob.tspan)[],timeseries_errors=true,reltol = 1e-3, abstol = 1e-6,
    kwargs...)

    p = prob.p
    tspan = prob.tspan
    u0 = prob.u0

    if DiffEqBase.isinplace(prob)
        f = function (t,u)
            du = similar(u)
            prob.f(du,u,p,t)
            du
        end
    else
        f = (t,u) -> prob.f(u,p,t)
    end

    _saveat = isempty(saveat) ? nothing : saveat
    if _saveat isa Array
        __saveat = _saveat
    elseif _saveat isa Number
        __saveat = Array(tspan[1]:_saveat:tspan[2])
    elseif _saveat isa Nothing
        __saveat = nothing
    else
        __saveat = Array(_saveat)
    end

    sol = integrate.solve_ivp(f,tspan,u0,
                              first_step = dt,
                              max_step = dtmax,
                              rtol = reltol, atol = abstol,
                              t_eval = __saveat,
                              dense_output=dense)

    ts = sol["t"]
    y = sol["y"]

    if typeof(u0) <: AbstractArray
        timeseries = Vector{typeof(u0)}(undef,length(ts))
        for i=1:length(ts)
            timeseries[i] = @view y[:,i]
        end
    else
        timeseries = y
    end

    if dense
        _interp = PyInterpolation(sol["sol"])
    else
        _interp = DiffEqBase.LinearInterpolation(ts,timeseries)
    end

    DiffEqBase.build_solution(prob,alg,ts,timeseries,
                              interp = _interp,
                              timeseries_errors = timeseries_errors)
end

struct PyInterpolation{T} <: DiffEqBase.AbstractDiffEqInterpolation
    pydense::T
end
function (PI::PyInterpolation)(t,idxs,deriv,p,continuity)
    if idxs !== nothing
        return PI.pydense(t)[idxs]
    else
        return PI.pydense(t)
    end
end
DiffEqBase.interp_summary(::PyInterpolation) = "Interpolation from SciPy"


end # module
