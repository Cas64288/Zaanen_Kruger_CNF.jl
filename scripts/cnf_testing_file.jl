# OLD CODE
using cuDNN
using CUDA; CUDA.device!(0)

using Images, FileIO, ImageTransformations
using ContinuousNormalizingFlows
using Lux, Zygote, ADTypes
using Distributions, DataFrames, MLJBase
using Random
using Tables
using ComponentArrays
using Functors   # ← make sure this is loaded

# ────────────────────────────────────────────────────────────────────────────
# Monkey-patch ComponentArrays to CPU-ify any CuArray fields before indexing
# ────────────────────────────────────────────────────────────────────────────
import ComponentArrays: ComponentArray
const _ComponentArray_orig = ComponentArray

function ComponentArray(nt::NamedTuple)
    # move any CuArray → Array
    nt_cpu = fmap(Array, nt)
    return _ComponentArray_orig(nt_cpu)
end
# ────────────────────────────────────────────────────────────────────────────

# ----------------------------------------
# 1) Load your images & build DataFrame
# ----------------------------------------
function load_img(fname)
    img = load(fname)
    arr = channelview(img)              # 3×H×W
    return vec(Float32.(arr))           # (3*H*W)-vector
end

function extract_alpha(fname)
    m = match(r"_a([0-9]+(?:\.[0-9]+)?)\.png$", fname)
    return m === nothing ? 0f0 : parse(Float32, m.captures[1])
end

img_dir   = raw"/home/ck422/UROP_2025/Astowell.jl-main/Astowell.jl-main/notebooks/Fractal_surface_img_reduced_ed/"
pngs      = filter(f->endswith(f,".png"), readdir(img_dir))
n_samples = length(pngs)
first_vec = load_img(joinpath(img_dir, pngs[1]))
n_pixels  = length(first_vec)

# pre-allocate
X  = Array{Float32}(undef, n_pixels, n_samples)
α  = Array{Float32}(undef, n_samples)

for (i,fname) in enumerate(pngs)
    X[:,i] = load_img(joinpath(img_dir,fname))
    α[i]   = extract_alpha(fname)
end

# stack image + α, convert to Float64, transpose ⇒ DataFrame
r   = vcat(X, reshape(α,1,:))         # Float32 (n_pixels+1)×n_samples
r64 = Float64.(r)
df  = DataFrame(transpose(r64), :auto)

# ----------------------------------------
# 2) Build & move your ICNF to GPU
# ----------------------------------------
nvars = n_pixels      # original dims
naugs = 1             # you still only augment α
n_in  = nvars + naugs # total input dims

# small 2-layer net (you can increase widths as needed)
nn = Chain(
  Dense(n_in => 64, relu),
  Dense(64   => 64, relu),
  Dense(64   => n_in, relu),
)

# set up parameters, push to GPU
ps, st = Lux.setup(Random.default_rng(), nn) .|> gpu_device()
icnf = construct(
  RNODE, nn, nvars, naugs;
  compute_mode = DIJacVecMatrixMode(AutoZygote()),
  tspan        = (0.0f0, 5.0f0),
  λ₁           = 1.0f-2,
  λ₂           = 1.0f-2,
  λ₃           = 1.0f-2,
  sol_kwargs   = (verbose=false,),
  device       = gpu_device(),
)

# ----------------------------------------
# 3) Wrap in an MLJ model & train
# ----------------------------------------
model = ICNFModel(icnf; adtype=AutoZygote(), n_epochs=200, batch_size=4)
mach  = machine(model, df)

# Now MLJ/ContinuousNormalizingFlows will CPU-ify parameters internally
fit!(mach, verbosity=2)

# ----------------------------------------
# 4) Inspect / sample
# ----------------------------------------
ps_f, st_f = fitted_params(mach)
d_learned  = ICNFDist(icnf, TestMode(), ps_f, st_f)

# PDF on held-out α’s (first 5 examples):
println("True α’s:     ", α[1:5])
println("Learned pdf:  ", pdf.(d_learned, transpose(df[1:5, end])))

# Draw new joint samples:
new = rand(d_learned, n_samples)
println("New samples size = ", size(new))  # (n_in)×n_samples
