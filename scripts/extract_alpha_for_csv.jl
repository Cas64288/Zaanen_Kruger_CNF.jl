# If you have CSV data from using the 2D_backflow_edit_un.jl
# You can extract alpha values from it 
# Probably won't be needed 
using CSV
using DataFrames
using Printf
input_csv = "nodal_surface_smooth.csv"

df = CSV.read(input_csv, DataFrame)

alphas = sort(unique(df.alpha))

for α in alphas
    sub = df[df.alpha .== α, :]
    # Note: my naming conventions, nodal_surface_alpha_0.0.csv ext
    fname = @sprintf("nodal_surface_alpha_%.1f.csv", α)
    println("Writing $(nrow(sub)) rows to $fname …")
    CSV.write(fname, sub)
end
println("Done")
