### Nodal Surface Visualizer (Julia Script)
# Edited version of Jarvist Frost's code added loop to produce more images 
using LinearAlgebra, Images, FileIO, ImageMagick, Printf
using DataFrames, CSV, Logging


# ────────── Parameters ──────────
const KMAX  = 4
const N     = 49        # # of reciprocal-space vectors & “particles”
const L     = 2π        # box size
const R0    = 0.2       # backflow cutoff

# ────────── Build small reciprocal-space grid ──────────
function K_small(; N=N)
    k = zeros(N, 2)
    n = 1
    for kx in -KMAX:KMAX, ky in -KMAX:KMAX
        if kx^2 + ky^2 ≤ KMAX^2 && n ≤ N
            k[n, :] = (kx, ky)
            n += 1
        end
    end
    return k
end

# ────────── Backflow weight ──────────
η(r; a=1.0) = a^3 / (r^3 + R0^3)

# ────────── A-matrix builder ──────────
function A(r, k; a=0.0)
    A = Matrix{ComplexF64}(undef, N, N)
    for j in 1:N
        # compute backflow-shifted coordinate rj
        rj = if a == 0.0
            r[j, :]
        else
            diffs  = r[j, :] .- r                 # N×2 array
            dists  = sqrt.(sum(abs2, diffs; dims=2))
            weights = η.(dists; a=a)
            weights[j] = 0.0
            r[j, :] .+ vec(sum(weights .* diffs; dims=1))
        end
        for i in 1:N
            A[i, j] = exp(im * dot(k[i, :], rj))
        end
    end
    return A
end

# ────────── Sample determinant across 2D grid ──────────
function sampleimg(r, k; S=100, a=0.0)
    img    = zeros(ComplexF64, S+1, S+1)
    coords = range(-L/2, L/2; length=S+1)
    for (i, x) in enumerate(coords)
        r[1,1] = x
        for (j, y) in enumerate(coords)
            r[1,2] = y
            img[i, j] = det( A(r, k; a=a) )
        end
    end
    return img
end

# ────────── Also collect (x,y,|det|) for CSV export ──────────
function collect_samples(r, k; S=100, a=0.0)
    data = Vector{Tuple{Float64,Float64,Float64}}()
    coords = range(-L/2, L/2; length=S+1)
    for (i, x) in enumerate(coords)
        r[1,1] = x
        for (j, y) in enumerate(coords)
            r[1,2] = y
            d = det( A(r, k; a=a) )
            push!(data, (x, y, abs(d)))
        end
    end
    return data
end

# ────────── Render image ──────────
function renderimg(img; POW=0.07)
    S      = size(img,1) - 1
    rgb    = similar(img, RGB{N0f8})
    maxabs = maximum(abs.(img))
    for idx in eachindex(img)
        c = (abs(img[idx]) / maxabs)^POW
        rgb[idx] = real(img[idx]) > 0 ? RGB{N0f8}(1,0,0)*c : RGB{N0f8}(0,0,1)*c
    end
    return imresize(rgb, (50,50))
end

# ────────── Main single-shot run ──────────
function main()
    # random initial particle positions
    r = rand(N,2) * L .- (L/2)
    k = K_small()

    a = 0.5
    @info "Sampling with backflow a=$a..."
    img = sampleimg(r, k; S=200, a=a)
    rendered = renderimg(img; POW=0.02)

    fname = @sprintf("nodal_a%.2f.png", a)
    save(fname, rendered)
    @info "Saved image to $fname"
end

# ────────── Batch over many alpha’s ──────────
const alphas = range(0.0, stop=0.8, length=1000)

function batch_over_alphas()
    r = rand(N,2) * L .- (L/2)
    k = K_small()
    for a in alphas
        @info "Processing alpha = $a"
        img = sampleimg(r, k; S=200, a=a)
        rendered = renderimg(img; POW=0.02)
        imgfile = @sprintf("img_a%.4f.png", a)
        save(imgfile, rendered)

        data = collect_samples(r, k; S=200, a=a)
        df   = DataFrame(x = getindex.(data,1),
                         y = getindex.(data,2),
                         abs_det = getindex.(data,3))
        csvfile = @sprintf("nodal_data_a%.4f.csv", a)
        CSV.write(csvfile, df)
        @info "  ↳ saved $imgfile and $csvfile"
    end
end

# ────────── Script entrypoint ──────────
main()
# or to do the full sweep:
# batch_over_alphas()

