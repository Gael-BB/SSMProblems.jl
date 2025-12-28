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
const Dx = 2
const Dy = 2
const K = 10
const T = Float64
const N_particles = 100
const N_burnin = 100
const N_sample = 1000
const TUNE_PARTICLES = false

@enum samplers PMMH_TYPE PGIBBS_TYPE EHMM_TYPE
sampler_type::samplers = EHMM_TYPE

rng = MersenneTwister(SEED)

α = 3.0; β = 2.0
σ2_prior = InverseGamma(α, β)
σ2_true = rand(rng, σ2_prior)
println("True σ2: ", σ2_true)

# Generate model matrices/vectors
μ0 = rand(rng, T, Dx)
Σ0 = rand_cov(rng, T, Dx)
A = rand(rng, T, Dx, Dx)
b = rand(rng, T, Dx)
Q = rand_cov(rng, T, Dx)
H = rand(rng, T, Dy, Dx)
c = rand(rng, T, Dy)
R = rand_cov(rng, T, Dy)

# Define full model and sample observations
full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, σ2_true .* Q, H, c, σ2_true .* R)
_, xs, ys = sample(rng, full_model, K)

model_init = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
cb = GeneralisedFilters.StateCallback(nothing, nothing)
_, _ = GeneralisedFilters.filter(rng, model_init, KalmanFilter(), ys; callback=cb)

function update_closed_params(α, β, ys, cb, model, Dy)
    α_closed = α
    β_closed = β

    for t in eachindex(ys)
        μ_pred, Σ_pred = GeneralisedFilters.mean_cov(cb.proposed_states[t])

        ŷ = H * μ_pred + c
        S = H * Σ_pred * H' + R
        e = ys[t] - ŷ

        α_closed += Dy / 2
        β_closed += 0.5 * dot(e, S \ e)
    end

    return α_closed, β_closed
end

α_closed, β_closed = update_closed_params(α, β, ys, cb, model_init, Dy)
println("Ground truth posterior mean: ", β_closed / (α_closed - 1))

struct ZeroLikelihoodModel end
GeneralisedFilters.filter(::AbstractRNG, ::ZeroLikelihoodModel, ::GeneralisedFilters.AbstractFilter, ::AbstractVector) = (nothing, -Inf)

function model_builder(θ)
    if θ[1] .<= 0.0
        return ZeroLikelihoodModel()
    end
    return create_homogeneous_linear_gaussian_model(
        μ0, Σ0, A, b, θ[1] .* Q, H, c, θ[1] .* R
    )
end

invQ = inv(Q)
invR = inv(R)
function q_sampler(ref_traj, rng, xs)
    # Residuals: process x_t - (A x_{t-1} + b), observation y_t - (H x_t + c)
    proc_res = [Array(ref_traj[t] .- (A * ref_traj[t - 1] .+ b)) for t in (firstindex(ref_traj) + 1):lastindex(ref_traj)]
    obs_res = [Array(ys[t] .- (H * ref_traj[t] .+ c)) for t in firstindex(ys):lastindex(ys)]

    # Sum of Mahalanobis distances with base covariances (Var = θ * Q/R)
    proc_ss = sum(r -> dot(r, invQ * r), proc_res)
    obs_ss = sum(r -> dot(r, invR * r), obs_res)
    ss = proc_ss + obs_ss

    # Total degrees of freedom contributed by all residual components
    n = length(proc_res) * Dx + length(obs_res) * Dy

    α_post = α + n / 2
    β_post = β + ss / 2

    return [rand(rng, InverseGamma(α_post, β_post))]
end

# Setup AbstractMCMC model
prior_logpdf(θ) = (θ[1] <= 0) ? -Inf : logpdf(σ2_prior, θ[1])
model = ParameterisedSSM(model_builder, prior_logpdf)
bf = BF(N_particles; threshold=1.0)

println("Starting sampling using sampler type: ", sampler_type)

samples = if sampler_type == PMMH_TYPE
    println("Estimating log-likelihood variance...")
    θ_curr = [β / (α - 1)]
    m_curr = model_builder(θ_curr)
    
    N_est, V = estimate_particle_count(rng, m_curr, ys, N -> BF(N; threshold=1.0); initial_N=N_particles)
    println("Log-likelihood variance at N=$N_particles: ", V)
    if TUNE_PARTICLES println("Estimated particles for variance=1.0: ", N_est) end
    
    N_run = TUNE_PARTICLES ? N_est : N_particles
    bf_tuned = BF(N_run; threshold=1.0)
    sampler = PMMH(bf_tuned; d=1, adapt_end=N_burnin)
    sample(rng, model, sampler, ys; n_samples=N_sample, n_burnin=N_burnin, init_θ=θ_curr)
elseif sampler_type == PGIBBS_TYPE
    sampler = PGibbs(bf, q_sampler)
    sample(rng, model, sampler, ys; n_samples=N_sample, n_burnin=N_burnin, init_θ=θ_curr)

elseif sampler_type == EHMM_TYPE
    θ_curr = [β / (α - 1)]
    sampler = EHMM(bf, q_sampler)
    sample(rng, model, sampler, ys; n_samples=N_sample, n_burnin=N_burnin, init_θ=θ_curr)
end

println("Posterior mean: ", mean(samples))
println("Effective sample size: ", ess(hcat(samples...)'))
