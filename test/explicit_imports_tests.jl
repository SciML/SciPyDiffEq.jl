using SciPyDiffEq
using ExplicitImports

@test check_no_implicit_imports(SciPyDiffEq) === nothing
@test check_no_stale_explicit_imports(SciPyDiffEq) === nothing
