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

μ = 0.0; σ2 = 1.0
b_prior = MvNormal(zeros(T, Dx) .+ μ, σ2 * I)
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

b_samples = Vector{Vector{T}}(undef, N_sample)
b_curr = [only(b_prior.μ)]
println("Initial b: ", b_curr)

function model_builder(θ)
    return create_homogeneous_linear_gaussian_model(
        μ0, Σ0, A, θ, Q, H, c, R
    )
end

println("Starting sampling using sampler type: ", sampler_type)
# =====================================
# PARTICLE MARGINAL METROPOLIS-HASTINGS
# =====================================
if sampler_type == PMMH
    b_samples = pmmh(
        N_steps, N_burnin,
        b_curr,
        model_builder,
        b_prior,
        ys, bf, rng
    )
end
# ==============
# PARTICLE GIBBS
# ==============
if sampler_type == PGIBBS
    function b_sampler(ref_traj, rng, xs)
        xs = [only(ref_traj[t] - A * ref_traj[t - 1]) for t in (firstindex(ref_traj) + 1):lastindex(ref_traj)]
        n = length(xs)
        
        μ_post = (sum(xs) / only(Q) + μ / σ2) / (n / only(Q) + 1 / σ2)
        σ2_post = 1 / (n / only(Q) + 1 / σ2)

        return [rand(rng, Normal(μ_post, sqrt(σ2_post)))]
    end

    b_samples = pgibbs(
        N_steps, N_burnin,
        b_curr,
        model_builder,
        b_sampler,
        ys, bf, rng
    )
end
# ============================
# EMBEDDED HIDDEN MARKOV MODEL
# ============================
if sampler_type == EHMM
    println("EHMM sampler not yet implemented.")
end

println("Posterior mean: ", mean(b_samples))
println("Effective sample size: ", ess(hcat(b_samples...)'))

display(plot(only.(b_samples); label="Chain", xlabel="Iteration", ylabel="b_outer", legend=:topleft))