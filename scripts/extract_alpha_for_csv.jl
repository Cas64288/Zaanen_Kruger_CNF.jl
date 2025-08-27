using CSV
using DataFrames
using Printf
# Path to your combined CSV:
input_csv = "nodal_surface_smooth.csv"

# Read entire file once
df = CSV.read(input_csv, DataFrame)

# Get the distinct alpha values, sorted
alphas = sort(unique(df.alpha))

for α in alphas
    # Filter rows where alpha == this value
    sub = df[df.alpha .== α, :]

    # Construct output filename, e.g. nodal_surface_alpha_0.0.csv
    fname = @sprintf("nodal_surface_alpha_%.1f.csv", α)

    # Write out the subset
    println("Writing $(nrow(sub)) rows to $fname …")
    CSV.write(fname, sub)
end
println("Done! Split into $(length(alphas)) files.")
