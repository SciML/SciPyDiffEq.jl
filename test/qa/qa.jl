using SciMLTesting, SciPyDiffEq, JET, Test

run_qa(
    SciPyDiffEq;
    explicit_imports = true,
    jet_kwargs = (; target_modules = (SciPyDiffEq,)),
    ei_kwargs = (;
        # SciMLBase-owned interface names extended/used here that are not public API.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDiffEqInterpolation, :LinearInterpolation, :__solve, :interp_summary,
            ),
        ),
    ),
)
