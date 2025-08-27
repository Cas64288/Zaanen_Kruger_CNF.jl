# Zaanen_Kruger_CNF.jl

This repo contains code which aims to machine learn images of a 2d slice of a 49^n dimentinal nodal hypersurface. 
Nodal images were produce within the Frost Research Group at Imperial College London using this repo https://github.com/Frost-group/Astowell.jl


Testing_cnf.jl  -> main training file 

Uses Conditional Continuous Normalizing Flow (CNF) Method based on the FFJORD algorithm to learn the probability distribution of a dataset of fractal-like images. (Original repo https://github.com/impICNF/ContinuousNormalizingFlows.jl/tree/main?tab=readme-ov-file)
