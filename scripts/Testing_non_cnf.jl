## Neural network to learn image structure conditioned on alpha using MSE loss
using CUDA, cuDNN, Lux, Zygote, Optimisers, Random, FileIO, Images, Functors, ColorTypes, Printf
using Plots
# I also did phased traning, a set of epochs with different learning rates 
# Later I do a decaying learning rate 
CUDA.allowscalar(false) # NEEDED 

# Loads images 
function load_img(filename)
    img = load(filename)
    img_array_chw = channelview(img) # C x H x W
    img_array_hwc = permutedims(img_array_chw, (2, 3, 1))
    return vec(Float32.(img_array_hwc))
end
# Extract alpha from file names like "img_a0.75.png", this is my naming convention for images 
function extract_alpha(fname::String)
    m = match(r"_a([0-9]+(?:\.[0-9]+)?)\.png$", fname)
    return m === nothing ? 0.0f0 : parse(Float32, m.captures[1])
end

# Load image dataset
img_dir = raw"/home/ck422/UROP_2025/Astowell.jl-main/Astowell.jl-main/notebooks/Fractal_img_50_ext/"
img_files = filter(f -> endswith(f, ".png"), readdir(img_dir))
n = length(img_files)

sample_vec = load_img(joinpath(img_dir, img_files[1]))
n_image = length(sample_vec)
H, W = 50, 50 # Or whatever the dimentions of your image are

X_cpu = Array{Float32}(undef, n_image, n)
alpha_cpu = Array{Float32}(undef, 1, n)
for (i, fname) in enumerate(img_files)
    X_cpu[:, i] = load_img(joinpath(img_dir, fname))
    alpha_cpu[1, i] = extract_alpha(fname)
end

const latent_dim = 256
const n_in = latent_dim + 1

# Using the powerful decoder architecture with leakyrelu as our activation function 
model = Chain(
    Dense(n_in => 2048),
    BatchNorm(2048),
    x -> leakyrelu.(x, 0.01f0),

    Dense(2048 => 4096),
    BatchNorm(4096),
    x -> leakyrelu.(x, 0.01f0),

    Dense(4096 => 8192),
    BatchNorm(8192),
    x -> leakyrelu.(x, 0.01f0),

    Dense(8192 => 8192),   # extra layer
    BatchNorm(8192),
    x -> leakyrelu.(x, 0.01f0),

    Dense(8192 => 12288),
    BatchNorm(12288),
    x -> leakyrelu.(x, 0.01f0),

    Dense(12288 => 12288), 
    BatchNorm(12288),
    x -> leakyrelu.(x, 0.01f0),

    Dense(12288 => n_image)
)


# Move data to GPU
X = cu(X_cpu)
alpha = cu(alpha_cpu)

# Lux setup
rng = Random.default_rng()
ps, st = Lux.setup(rng, model)
ps = fmap(cu, ps)
st = fmap(cu, st)
flat_ps, re = Optimisers.destructure(ps)
flat_ps = cu(flat_ps)

# # MSE loss between predicted and ground truth images
function loss_fn_deterministic(p, z)
    global st
    current_ps = re(p)
    z_cond = vcat(z, alpha)
    ŷ, st = model(z_cond, current_ps, st)
    return sum(abs2, ŷ .- X) / n
end

# Traning 
loss_history = Float32[]
callback = function (p, l)
    push!(loss_history, l)
    if length(loss_history) % 100 == 0
        println("Iteration: $(length(loss_history)) Loss: $(l)")
    end
    return false
end

# Phase 1
println("Training with learning rate of 1e-3")
opt1 = ADAM(1e-3)
opt_state1 = Optimisers.setup(opt1, flat_ps)

for i in 1:5000
    global flat_ps, opt_state1
    z = CUDA.randn(Float32, latent_dim, n)
    loss, grads = Zygote.withgradient(p -> loss_fn_deterministic(p, z), flat_ps)
    opt_state1, flat_ps = Optimisers.update!(opt_state1, flat_ps, grads[1])
    callback(flat_ps, loss)
end

# Phase 2-
println("Training with learning rate of 1e-5")
opt2 = ADAM(1e-5)
opt_state2 = Optimisers.setup(opt2, flat_ps)

for i in 1:5000
    global flat_ps, opt_state2
    z = CUDA.randn(Float32, latent_dim, n)
    loss, grads = Zygote.withgradient(p -> loss_fn_deterministic(p, z), flat_ps)
    opt_state2, flat_ps = Optimisers.update!(opt_state2, flat_ps, grads[1])
    callback(flat_ps, loss)
end
println("Training finished")


# Plotting loss function 
plot(loss_history, xlabel="Iteration", ylabel="Loss", title="Training Loss Over Time (MSE)", label="Loss", yaxis=:log10)
savefig("training_loss_plot_extra.png")

println("Generating img")
ps_trained = re(flat_ps)
output_dir = "generated_sequence_low_loss_extra"
mkpath(output_dir)
st_inference = Lux.testmode(st)
z_test = CUDA.randn(Float32, latent_dim, 1)

# loop to produce images witrh different alpha values 
for α_val in 0.0f0:0.05f0:1.2f0
    z_cond_test = vcat(z_test, fill(α_val, 1, 1))
    ŷ, _ = model(z_cond_test, ps_trained, st_inference)
    ŷ_img = clamp.(ŷ, 0f0, 1f0)

    ŷ_img_array = Array(ŷ_img[:, 1])
    ŷ_array = reshape(ŷ_img_array, (H, W, 3))
    ŷ_image = colorview(RGB, permutedims(ŷ_array, (3, 1, 2)))
    
    save_name = @sprintf "alpha_%.2f.png" α_val
    save(joinpath(output_dir, save_name), ŷ_image)
end
println("Saved/'$(output_dir)'")
