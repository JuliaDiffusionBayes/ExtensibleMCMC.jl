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
        ),
        collapselevel = 1,
    ),
    pages=[
        "Home" => "index.md",
        "Get started" => joinpath("get_started", "basic_use.md"),
        "User manual" => Any[
            "Internal structure" => joinpath("manual", "internal_structure.md"),
            "Updates & Decorators" => joinpath("manual", "updates_and_decorators.md"),
            "Callbacks" => joinpath("manual", "callbacks.md"),
            "Priors" => joinpath("manual", "priors.md"),
            "Workspaces" => joinpath("manual", "workspaces.md"),
            "MCMC Schedule" => joinpath("manual", "mcmc_schedule.md"),
        ],
        "How to..." => Any[
            "(TODO) First how to" => joinpath("how_to_guides", "first_how_to.md"),
            "(TODO) Definy my own workspace" => joinpath("how_to_guides", "define_my_own_workspace.md"),
        ],
        "Tutorials" => Any[
            "Estimate mean of a bivariate Gaussian" => joinpath("tutorials", "mean_of_bivariate_gaussian.md"),
        ],
        "Index" => "module_index.md"
    ],
    repo="https://github.com/JuliaDiffusionBayes/ExtensibleMCMC.jl/blob/{commit}{path}#L{line}",
    sitename="ExtensibleMCMC.jl",
    authors="Sebastiano Grazzi, Frank van der Meulen, Marcin Mider, Moritz Schauer",
    #assets=String[],
)

deploydocs(;
    repo="github.com/JuliaDiffusionBayes/ExtensibleMCMC.jl",
)
