using SciMLTesting, SciPyDiffEq, Test

# The SciML common interface SciPyDiffEq deliberately reexports so that
# `using SciPyDiffEq` is enough to build an ODE problem, solve it, and inspect the
# result. Owned and documented upstream; kept in sync with the reexport `export` block
# in src/SciPyDiffEq.jl and the API page in docs/src/api.md.
const REEXPORTS = (
    :DEStats, :EnsembleAnalysis, :EnsembleDistributed, :EnsembleProblem, :EnsembleSerial,
    :EnsembleSolution, :EnsembleSplitThreads, :EnsembleSummary, :EnsembleThreads,
    :NullParameters, :ODEFunction, :ODEProblem, :ODESolution, :ReturnCode, :remake,
    :solve, :successful_retcode,
)

run_qa(SciPyDiffEq; reexports_allow = REEXPORTS)

@testset "Reexport surface" begin
    # Every approved reexport must actually be reachable from `using SciPyDiffEq`, so
    # the allow-list cannot drift into approving names the package no longer provides.
    # `isdefined(@__MODULE__, ...)` tests the property directly: this file's
    # `using SciPyDiffEq` is what has to bring the name into scope.
    @testset "$name" for name in REEXPORTS
        @test name in names(SciPyDiffEq)
        @test isdefined(@__MODULE__, name)
    end
end
