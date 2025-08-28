#=
Code extracts nodal lines only from the nodal surface; see the example of images.
Input images are assumed to use red and blue channels to distinguish regions.
Nodal points are places where red > blue in one pixel, and the opposite in a neighbour.
=#
using FileIO, Images, ImageTransformations, Colors
using CSV, DataFrames

img_dir = raw"path_to_img\Fractal_surface_img"  # ← replace with actual image directory
output_csv = "nodal_points_output.csv"

# extracts alpha from file name 
function extract_alpha(fname::String)
    m = match(r"_a([0-9.]+)\.png$", fname)
    return m === nothing ? 0.0f0 : parse(Float32, m.captures[1])
end

# loads mask and extracts nodal points 
function extract_nodal_points(filename, α)
    img = load(filename)
    img_resized = imresize(img, (50, 50))  # resize not be necessary, change at your discretion
    img_array = channelview(img_resized)  # shape: C x H x W

    H, W = size(img_array, 2), size(img_array, 3)
    red = img_array[1, :, :]
    blue = img_array[3, :, :]
    # Create a binary mask: 1.0 if red > blue, else 0.0
    mask = map((r, b) -> r > b ? 1.0f0 : 0.0f0, red, blue)

    nodal_pts = []
    #Loop through interior pixels
    for i in 2:H-1, j in 2:W-1
        center = mask[i, j]
        neighbors = (mask[i+1,j], mask[i-1,j], mask[i,j+1], mask[i,j-1])
        # If any neighbour has a different mask value → nodal transition
        if any(n != center for n in neighbors)
            x = i / H
            y = j / W
            push!(nodal_pts, (x, y, α))
        end
    end

    return nodal_pts
end

all_nodal_data = []

for file in readdir(img_dir)
    if endswith(file, ".png")
        α = extract_alpha(file)
        fullpath = joinpath(img_dir, file)
        nodal_pts = extract_nodal_points(fullpath, α)
        append!(all_nodal_data, nodal_pts)
    end
end

# Save to CSV
df = DataFrame(x = Float32[], y = Float32[], alpha = Float32[])
for (x, y, α) in all_nodal_data
    push!(df, (x, y, α))
end

CSV.write(output_csv, df)
println("Saved/$output_csv")
