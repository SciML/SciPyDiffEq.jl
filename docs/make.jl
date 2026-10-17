using Documenter
using SciPyDiffEq

makedocs(
    modules = [SciPyDiffEq],
    sitename = "SciPyDiffEq.jl",
    checkdocs = :all,
    doctest = true,
    linkcheck = true,
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)
