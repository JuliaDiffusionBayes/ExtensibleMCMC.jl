using Documenter, ExtensibleMCMC

makedocs(;
    modules=[ExtensibleMCMC],
    format=Documenter.HTML(
        mathengine = Documenter.MathJax(
            Dict(
                :TeX => Dict(
                    :equationNumbers => Dict(
                        :autoNumber => "AMS"
                    ),
                    :Macros => Dict(
                        :dd => "{\\textrm d}",
                        :RR => "\\mathbb{R}",
                        :wt => ["\\widetilde{#1}", 1]
                    ),
                )
            )
        )
    ),
    pages=[
        "Home" => "index.md",
        "Overview" => Any[
            "Basic use" => joinpath("overview", "basic_use.md"),
            "Callbacks" => joinpath("overview", "callbacks.md"),
            "Updates" => joinpath("overview", "updates.md"),
            "Priors" => joinpath("overview", "priors.md"),
        ],
        "Advanced use" => Any[
            "Internal structure" => joinpath("advanced_use", "internal_structure.md"),
            "Callbacks" => joinpath("advanced_use", "callbacks.md"),
            "Updates & Decorators" => joinpath("advanced_use", "updates_and_decorators.md"),
            "Priors" => joinpath("advanced_use", "priors.md"),
            "Workspaces" => joinpath("advanced_use", "workspaces.md"),
            "MCMC Schedule" => joinpath("advanced_use", "mcmc_schedule.md"),
        ],
        "Index" => "module_index.md"
    ],
    repo="https://github.com/JuliaDiffusionBayes/ExtensibleMCMC.jl/blob/{commit}{path}#L{line}",
    sitename="ExtensibleMCMC.jl",
    authors="Sebastiano Grazzi, Frank van der Meulen, Marcin Mider, Moritz Schauer",
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaDiffusionBayes/ExtensibleMCMC.jl",
)
