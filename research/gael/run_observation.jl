using GeneralisedFilters
using Distributions
using LinearAlgebra
using LogExpFunctions
using ProgressMeter
using OffsetArrays
using Random
using SSMProblems
using StatsBase
using Plots
using MCMCDiagnosticTools

include("utils.jl")
include("pmcmc.jl")
include("ehmm.jl")

rng = MersenneTwister(SEED)

α = 3.0; β = 2.0
Q_prior = InverseGamma(α, β)
Q_true = [rand(rng, Q_prior) for _ in 1:Dx]
println("True Q: ", Q_true)

# Generate model matrices/vectors
μ0 = rand(rng, T, Dx)
Σ0 = rand_cov(rng, T, Dx)
A = rand(rng, T, Dx, Dx)
b = rand(rng, T, Dx)
H = Matrix{T}(I, Dy, Dy)
c = fill(0, Dy)
eps = 1e-5; R = Matrix{T}(I, Dy, Dy) * eps

# Define full model and sample observations
full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Diagonal(Q_true), H, c, R)
_, ys = sample(rng, full_model, K)

# Compute process noise samples w_t and empirical process noise covariance Q_hat
# State dynamics: x_{t+1} = A * x_t + b + w_t, so w_t = x_{t+1} - A * x_t - b
# Here we assume H = I, c = 0, and R = 0 as specified above.

Tsteps = length(ys)  # number of time steps
@assert Tsteps > 0 "Need at least one time steps to compute process noise."

elty = eltype(ys[1])
# Each row of `ws` will store one w_t sample (rows = time-1, cols = state dim)
ws = Matrix{elty}(undef, Tsteps - 1, Dx)
for t in 1:(Tsteps - 1)
    xt = ys[t]
    xt1 = ys[t + 1]
    ws[t, :] = xt1 .- A * xt .- b
end

# Closed-form empirical estimate of the process noise covariance Q
# Bayesian (Inverse-Gamma) update per-dimension: posterior IG(α + n/2, β + 0.5*S)
n = Tsteps - 1
S = vec(sum(ws .^ 2, dims = 1))    # sum of squares per state-dimension
α_post = α + n/2
β_post = β .+ 0.5 .* S

# Use posterior mean estimate for each diagonal element (requires α_post > 1).
# Fallback to posterior mode if α_post <= 1.

Q_diag = β_post ./ (α_post .- 1)    # posterior mean

Q_hat = Matrix(Diagonal(Q_diag))
println("Ground truth posterior mean: ", Q_hat)


N_steps = N_burnin + N_sample
bf = BF(N_particles; threshold=1.0)
ref_traj = nothing

samples = Vector{Vector{T}}(undef, N_sample)
curr = [β / (α - 1)]
println("Initial Q: ", [β / (α - 1)])

function model_builder(θ)
    return create_homogeneous_linear_gaussian_model(
        μ0, Σ0, A, b, Diagonal(θ), H, c, R
    )
end

function prior(θ)
    return only(logpdf(Q_prior, θ))
end

println("Starting sampling using sampler type: ", sampler_type)
# =====================================
# PARTICLE MARGINAL METROPOLIS-HASTINGS
# I think the acceptance rate is small because the likelihood is broken for the R value used.
# =====================================
if sampler_type == PMMH
    samples = pmmh(
        N_steps, N_burnin,
        curr,
        model_builder,
        prior,
        ys, bf, rng,
        (rng, θ) -> θ .+ 0.0001*randn(rng, length(θ))
    )
end
# ==============
# PARTICLE GIBBS
# ==============
if sampler_type == PGIBBS
    function sampler(ref_traj, rng, xs)
        # State innovation residuals: x_t - A x_{t-1} - b
        xs = [only(ref_traj[t] - A * ref_traj[t - 1] .- b) for t in (firstindex(ref_traj) + 1):lastindex(ref_traj)]
        n = length(xs)

        ss = sum(abs2, xs)  # sum of squared residuals

        # Conjugate IG posterior
        α_post = α + n / 2
        β_post = β + ss / 2

        return [rand(rng, InverseGamma(α_post, β_post))]
    end

    samples = pgibbs(
        N_steps, N_burnin,
        curr,
        model_builder,
        sampler,
        ys, bf, rng
    )
end
# ============================
# EMBEDDED HIDDEN MARKOV MODEL
# ============================
if sampler_type == EHMM
    println("EHMM sampler not yet implemented.")
end

println("Posterior mean: ", mean(samples))
println("Effective sample size: ", ess(hcat(samples...)'))

display(plot(only.(samples); label="Chain", xlabel="Iteration", ylabel="Q", legend=:topleft))