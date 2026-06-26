using SciMLTesting, SciPyDiffEq, JET, Test

run_qa(
    SciPyDiffEq;
    explicit_imports = true,
    jet_kwargs = (; target_defined_modules = true),
    ei_kwargs = (;
        # SciMLBase interface names extended/used here are still not declared public in SciMLBase.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDiffEqInterpolation, :LinearInterpolation, :__solve, :interp_summary,
            ),
        ),
    ),
)
