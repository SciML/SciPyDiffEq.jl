using SciPyDiffEq
using Aqua
using JET
using Test

@testset "Aqua" begin
    Aqua.test_all(SciPyDiffEq)
end

@testset "JET" begin
    JET.test_package(SciPyDiffEq; target_defined_modules = true)
end
