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
_, ys = sample(rng, full_model, K)

model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
cb = GeneralisedFilters.StateCallback(nothing, nothing)
_, _ = GeneralisedFilters.filter(rng, model, KalmanFilter(), ys; callback=cb)

α_closed = α
β_closed = β

for t in eachindex(ys)
    global α_closed, β_closed  # or two separate lines
    μ_pred, Σ_pred = GeneralisedFilters.mean_cov(cb.proposed_states[t])

    ŷ = H * μ_pred + c
    S = H * Σ_pred * H' + R
    e = ys[t] - ŷ

    α_closed += Dy / 2
    β_closed += 0.5 * dot(e, S \ e)
end

println("Final posterior mean: ", β_closed / (α_closed - 1))

N_steps = N_burnin + N_sample
bf = BF(N_particles; threshold=1.0)
ref_traj = nothing

samples = Vector{Vector{T}}(undef, N_sample)
curr = [β / (α - 1)]
println("Initial σ2: ", [β / (α - 1)])

struct ZeroLikelihoodModel end
GeneralisedFilters.filter(::ZeroLikelihoodModel, bf, ys) = (nothing, -Inf)

function model_builder(θ)
    if θ[1] .<= 0.0
        return ZeroLikelihoodModel()  # returns -Inf log-likelihood downstream
    end
    return create_homogeneous_linear_gaussian_model(
        μ0, Σ0, A, b, θ .* Q, H, c, θ .* R
    )
end

function prior(θ)
    if θ[1] <= 0.0
        return -Inf
    end
    return logpdf(σ2_prior, θ[1])
end

function q_sampler(ref_traj, rng, xs)
    # Process residuals x_t - (A x_{t-1} + b) and observation residuals y_t - (H x_t + c)
    proc_res = [only(ref_traj[t] - A * ref_traj[t - 1] .- b) for t in (firstindex(ref_traj) + 1):lastindex(ref_traj)]
    obs_res = [only(ys[t] - H * ref_traj[t] .- c) for t in firstindex(ys):lastindex(ys)]

    # Sum of squared residuals scaled by their base variances (since Var = θ * Q/R)
    ss = sum(abs2, proc_res) / only(Q) + sum(abs2, obs_res) / only(R)
    n = length(proc_res) + length(obs_res)

    # Conjugate IG posterior for the shared variance scale θ
    α_post = α + n / 2
    β_post = β + ss / 2

    return [rand(rng, InverseGamma(α_post, β_post))]
end

println("Starting sampling using sampler type: ", sampler_type)
# =====================================
# PARTICLE MARGINAL METROPOLIS-HASTINGS
# =====================================
if sampler_type == PMMH
    samples = pmmh(
        N_steps, N_burnin,
        curr,
        model_builder,
        prior,
        ys, bf, rng,
        (rng, θ) -> θ .+ 0.3*randn(rng, length(θ))
    )
end
# ==============
# PARTICLE GIBBS
# ==============
if sampler_type == PGIBBS
    samples = pgibbs(
        N_steps, N_burnin,
        curr,
        model_builder,
        q_sampler,
        ys, bf, rng
    )
end
# ============================
# EMBEDDED HIDDEN MARKOV MODEL
# ============================
if sampler_type == EHMM
    samples = ehmm(
        N_steps, N_burnin,
        curr,
        model_builder,
        q_sampler,
        ys, rng
    )
end

println("Posterior mean: ", mean(samples))
println("Effective sample size: ", ess(hcat(samples...)'))

# display(plot(only.(samples); label="Chain", xlabel="Iteration", ylabel="σ2", legend=:topleft))
