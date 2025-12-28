using Distributions
using LinearAlgebra
using Random
using SSMProblems
using GeneralisedFilters
using ProgressMeter
using StatsBase
using Plots
using MCMCDiagnosticTools

# Load local research module
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using GaelResearch

const SEED = 12345
const Dx = 1
const Dy = 1
const K = 10
const T = Float64
const N_particles = 100
const N_burnin = 100
const N_sample = 1000
const TUNE_PARTICLES = false

@enum samplers PMMH_TYPE PGIBBS_TYPE EHMM_TYPE
sampler_type::samplers = EHMM_TYPE

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
_, xs, ys = sample(rng, full_model, K)

# Define augmented dynamics
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

function model_builder(θ)
    return create_homogeneous_linear_gaussian_model(
        μ0, Σ0, A, θ, Q, H, c, R
    )
end

Qinv = inv(Q)
Σ_prior_inv = inv(b_prior.Σ)
function b_sampler(ref_traj, rng, xs)
    # compute residuals r_t = x_t - A * x_{t-1} for t=2..T
    residuals = [Array(ref_traj[t] - A * ref_traj[t-1]) for t in (firstindex(ref_traj) + 1):lastindex(ref_traj)]
    n = length(residuals)
    sum_r = reduce(+, residuals)

    μ_prior = b_prior.μ
    Σ_prior = b_prior.Σ

    Σ_post = inv(n * Qinv + Σ_prior_inv)
    μ_post = Σ_post * (Qinv * sum_r + Σ_prior_inv * μ_prior)

    return rand(rng, MvNormal(vec(μ_post), Symmetric(Σ_post)))
end

# Setup AbstractMCMC model
model = ParameterisedSSM(model_builder, b_prior)
bf = BF(N_particles; threshold=1.0)

println("Starting sampling using sampler type: ", sampler_type)

b_samples = if sampler_type == PMMH_TYPE
    println("Estimating log-likelihood variance...")
    b_curr = b_prior.μ
    m_curr = model_builder(b_curr)
    
    N_est, V = estimate_particle_count(rng, m_curr, ys, N -> BF(N; threshold=1.0); initial_N=N_particles)
    println("Log-likelihood variance at N=$N_particles: ", V)
    if TUNE_PARTICLES println("Estimated particles for variance=1.0: ", N_est) end
    
    N_run = TUNE_PARTICLES ? N_est : N_particles
    bf_tuned = BF(N_run; threshold=1.0)
    sampler = PMMH(bf_tuned; d=length(b_prior), adapt_end=N_burnin)
    sample(rng, model, sampler, ys; n_samples=N_sample, n_burnin=N_burnin, init_θ=b_curr)
elseif sampler_type == PGIBBS_TYPE
    sampler = PGibbs(bf, b_sampler)
    sample(rng, model, sampler, ys; n_samples=N_sample, n_burnin=N_burnin)

elseif sampler_type == EHMM_TYPE
    sampler = EHMM(bf, b_sampler)
    sample(rng, model, sampler, ys; n_samples=N_sample, n_burnin=N_burnin)
end

println("Posterior mean: ", mean(b_samples))
println("Effective sample size: ", ess(hcat(b_samples...)'))
