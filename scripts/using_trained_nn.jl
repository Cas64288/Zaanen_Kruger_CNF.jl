# Code is meant to be ran after using Testing_cnf.jl
using Logging, TerminalLoggers
global_logger(TerminalLogger())

using ContinuousNormalizingFlows, Lux, OrdinaryDiffEq
using LuxCUDA, CUDA, cuDNN
using ComponentArrays, Distributions, Random, SciMLBase, SciMLSensitivity
using Images, FileIO, JLD2, MLUtils

const cpu_dev = cpu_device()
const gpu_dev = gpu_device()

# Loading model and data 
@info "Loading model..."
jld_data = jldopen(raw"C:/Users/casan/Downloads/UROP_2025/model_cnn_epoch_100.jld2", "r")
ps_loaded = jld_data["ps"]
st_loaded = jld_data["st"]
H = jld_data["H"]
W = jld_data["W"]
C = jld_data["C"]
data_mean = jld_data["data_mean"]
data_std = jld_data["data_std"]
close(jld_data)

#  RE-DEFINE MODEL STRUCTURE 
# Parameters must match those used to make the model in Testing_cnf.jl
nvars = H * W * C
nconds = 1
naugs = 0
n_in = nvars + naugs

# NOTE: "Image dimensions must be divisible by 4 for this CNN."
hidden_dim = 64
start_H, start_W = H ÷ 4, W ÷ 4

# Define a reusable Residual Block
function ResBlock(channels)
    return SkipConnection(
        Chain(
            GroupNorm(channels, channels ÷ 4),
            leakyrelu,
            Conv((3, 3), channels => channels; pad=1),
            GroupNorm(channels, channels ÷ 4),
            leakyrelu 
        ),
        +
    )
end
# VERY IMPORTANT: nn chain, Resblock and icnf function must be the same as the ones in Testing_cnf.jl
nn = Chain(
    Dense(n_in + nconds => start_H * start_W * hidden_dim),
    Lux.WrappedFunction(x -> reshape(x, start_H, start_W, hidden_dim, size(x, 2))),
    ResBlock(hidden_dim),
    ResBlock(hidden_dim),
    ConvTranspose((4, 4), hidden_dim => hidden_dim ÷ 2; stride=2, pad=1),
    GroupNorm(hidden_dim ÷ 2, (hidden_dim ÷ 2) ÷ 4),
    leakyrelu, 
    ConvTranspose((4, 4), hidden_dim ÷ 2 => C; stride=2, pad=1),
    Lux.WrappedFunction(MLUtils.flatten)
)

icnf = construct(
    FFJORD,
    nn,
    nvars,
    naugs;
    compute_mode = LuxVecJacMatrixMode(AutoZygote()),
    device = gpu_dev,
    tspan = (0.0f0, 1.0f0),
    sol_kwargs = (; save_everystep = false, alg = VCABM(), sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP())),
)

# GENERATION FUNCTION 
# Code is ran on cpu ()
function generate_images(ps_trained, st_trained, alpha_val)
    # Move model components to the CPU for the solve
    ps_cpu = cpu_device()(ps_trained)
    st_cpu = cpu_device()(st_trained)
    st_test_cpu = Lux.testmode(st_cpu)

    # 1. Sample a single random noise VECTOR on the CPU
    z_cpu = rand(icnf.basedist)
    
    # 2. Prepare the conditional alpha value on the CPU
    ys_cpu = fill(Float32(alpha_val), 1, 1)

    # 3. Define the dynamics function for inference
    function inference_dynamics_cpu(u, p, t)
        # u is a vector, so we reshape it to a 1-column matrix for the network
        u_matrix = reshape(u, :, 1)
        nn_input = vcat(u_matrix, ys_cpu)
        
        # Call the neural network directly to get the derivative
        du_matrix, _ = icnf.nn(nn_input, p, st_test_cpu)
        
        # Return a vector to match the shape of the input u
        return vec(du_matrix)
    end

    # 4. Define and solve the reverse ODE problem using our custom function
    tspan_reverse = (1.0f0, 0.0f0)
    prob_reverse = ODEProblem(inference_dynamics_cpu, z_cpu, tspan_reverse, ps_cpu)
    
    sol = solve(prob_reverse; icnf.sol_kwargs...)
    generated_vector_normalized = sol.u[end]

    # 5. De-normalize the output
    generated_vector_denormalized = (generated_vector_normalized .* data_std) .+ data_mean
    
    # 6. Clamp, reshape, and return the final image
    img_data_hwc = clamp.(generated_vector_denormalized, 0.0f0, 1.0f0)
    img_array = reshape(img_data_hwc, H, W, C)
    return colorview(RGB, permutedims(img_array, (3, 1, 2)))
end

# --- 5. USE THE MODEL ---
# The parameters and state are loaded to the CPU, then moved to GPU for storage
ps_gpu = gpu_dev(ComponentArray(ps_loaded))
st_gpu = gpu_dev(st_loaded)

@info "Generating image sequence..."
output_dir = "generated_img_final_test12"
mkpath(output_dir)
for alpha in 0.0:0.1:1.0
    img = generate_images(ps_gpu, st_gpu, alpha)
    save(joinpath(output_dir, "alpha_$(alpha).png"), img)
end
@info "Imag saved/'$output_dir'"
