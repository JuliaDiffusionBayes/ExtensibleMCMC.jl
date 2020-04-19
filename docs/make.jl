using Documenter, ExtensibleMCMC

makedocs(;
    modules=[ExtensibleMCMC],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/JuliaDiffusionBayes/ExtensibleMCMC.jl/blob/{commit}{path}#L{line}",
    sitename="ExtensibleMCMC.jl",
    authors="Sebastiano Grazzi, Frank van der Meulen, Marcin Mider, Moritz Schauer",
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaDiffusionBayes/ExtensibleMCMC.jl",
)
