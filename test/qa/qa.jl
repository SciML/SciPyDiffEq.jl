using SciMLTesting, SciPyDiffEq, JET, Test

run_qa(
    SciPyDiffEq;
    explicit_imports = true,
    jet_kwargs = (; target_defined_modules = true),
    ei_kwargs = (;
        # SciMLBase-owned interface names extended/used here that are still not
        # declared public in SciMLBase 3.27.0 (verified Base.ispublic == false).
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDiffEqInterpolation, :LinearInterpolation, :__solve, :interp_summary,
            ),
        ),
    ),
)
