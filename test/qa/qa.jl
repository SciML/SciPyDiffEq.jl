using SciMLTesting, SciPyDiffEq, JET, Test

run_qa(
    SciPyDiffEq;
    explicit_imports = true,
    jet_kwargs = (; target_defined_modules = true),
    ei_kwargs = (;
        # `solve` is the CommonSolve verb (non-public there); imported to dispatch on it.
        all_explicit_imports_are_public = (; ignore = (:solve,)),
        # SciMLBase interface names extended/used here are not declared public in SciMLBase;
        # Success/Failure are SciMLBase.ReturnCode enum members (public on 1.11+, not on 1.10).
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDiffEqInterpolation, :AbstractODEAlgorithm, :AbstractODEProblem,
                :LinearInterpolation, :__solve, :build_solution, :interp_summary,
                :Success, :Failure,
            ),
        ),
    ),
)
