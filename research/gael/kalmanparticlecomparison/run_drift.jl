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
include("particlegibbs.jl")
include("particlemarginalmetropolishastings.jl")

rng = MersenneTwister(SEED)

b_prior = MvNormal(zeros(T, Dx), 1I)
b_true = rand(rng, b_prior)
println("True b: ", b_true)

# Generate model matrices/vectors
μ0 = rand(rng, T, Dx)
Σ0 = rand_cov(rng, T, Dx)
A = rand(rng, T, Dx, Dx)
Q = rand_cov(rng, T, Dx)
H = rand(rng, T, Dy, Dx)
c = rand(rng, T, Dy)
R = rand_cov(rng, T, Dy)

# Define full model and sample observations
full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b_true, Q, H, c, R)
_, ys = sample(rng, full_model, K)

# Define augemented dynamics
μ0_aug = [μ0; b_prior.μ]
Σ0_aug = [
    Σ0 zeros(T, Dx, Dx)
    zeros(T, Dx, Dx) b_prior.Σ
]
A_aug = [
    A I
    zeros(T, Dx, Dx) I
]
b_aug = zeros(T, 2 * Dx)
Q_aug = [
    Q zeros(T, Dx, Dx)
    zeros(T, Dx, Dx) zeros(T, Dx, Dx)
]
H_aug = [H zeros(T, Dy, Dx)]

# Create augmented model
aug_model = create_homogeneous_linear_gaussian_model(
    μ0_aug, Σ0_aug, A_aug, b_aug, Q_aug, H_aug, c, R
)
state, _ = GeneralisedFilters.filter(rng, aug_model, KalmanFilter(), ys)
println("Ground truth posterior mean: ", state.μ[(Dx+1):end])

N_steps = N_burnin + N_sample
bf = BF(N_particles; threshold=1.0)
ref_traj = nothing

b_samples = Vector{Vector{T}}(undef, N_sample)
b_curr = [only(b_prior.μ)]
println("Initial b: ", [only(b_prior.μ)])

# ========================================
# CHANGE PARTICLE FILTERING ALGORITHM HERE
# ========================================

b_samples = run_pmmh(
    N_steps, N_burnin,
    μ0, Σ0, A, c, Q, H, R,
    b_prior, ys, bf, ref_traj, b_curr, rng
)

println("Posterior mean: ", mean(b_samples))
println("Effective sample size: ", ess(hcat(b_samples...)'))

# display(plot(only.(b_samples); label="Chain", xlabel="Iteration", ylabel="b_outer", legend=:topleft))