module RunScale

using Distributions
using LinearAlgebra
using Random
using SSMProblems
using GeneralisedFilters
using ProgressMeter
using StatsBase
using Plots
using MCMCDiagnosticTools
using StaticArrays

# Load local research module
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using GaelResearch

export run_experiment

const Dx = 2
const Dy = 2
const K = 10
const T = Float64
const N_particles = 100
const N_burnin = 100
const N_sample = 1000
const TUNE_PARTICLES = false

@enum samplers PMMH_TYPE PGIBBS_TYPE EHMM_TYPE

# Helper functions and types (Must be top level)
struct ZeroLikelihoodModel end
GeneralisedFilters.filter(::AbstractRNG, ::ZeroLikelihoodModel, ::GeneralisedFilters.AbstractFilter, ::AbstractVector) = (nothing, -Inf)


function run_experiment(seed::Int, sampler_name::Symbol)
    # Map symbol to enum
    sampler_type = if sampler_name == :PMMH
        PMMH_TYPE
    elseif sampler_name == :PGIBBS
        PGIBBS_TYPE
    elseif sampler_name == :EHMM
        EHMM_TYPE
    else
        error("Unknown sampler: $sampler_name")
    end

    rng = MersenneTwister(seed)

    α = 3.0; β = 2.0
    σ2_prior = InverseGamma(α, β)
    σ2_true = rand(rng, σ2_prior)

    # Generate model matrices/vectors
    μ0 = @SVector rand(rng, T, Dx)
    Σ0 = SMatrix{Dx, Dx}(rand_cov(rng, T, Dx))
    A = @SMatrix rand(rng, T, Dx, Dx)
    b = @SVector rand(rng, T, Dx)
    Q = SMatrix{Dx, Dx}(rand_cov(rng, T, Dx))
    H = @SMatrix rand(rng, T, Dy, Dx)
    c = @SVector rand(rng, T, Dy)
    R = SMatrix{Dy, Dy}(rand_cov(rng, T, Dy))

    # Pre-wrap covariances
    Σ0_pd = Distributions.PDMat(ensure_posdef(Matrix(Σ0)))
    Q_pd = Distributions.PDMat(ensure_posdef(Matrix(Q)))
    R_pd = Distributions.PDMat(ensure_posdef(Matrix(R)))

    # Define full model and sample observations
    # Note: Q and R scaler multiplication via .* returns Matrix, so we wrap result
    full_model = create_homogeneous_linear_gaussian_model(
        μ0, Σ0_pd, A, b, 
        Distributions.PDMat(ensure_posdef(Matrix(σ2_true .* Q))), 
        H, c, 
        Distributions.PDMat(ensure_posdef(Matrix(σ2_true .* R)))
    )
    _, xs, ys = sample(rng, full_model, K)

    model_init = create_homogeneous_linear_gaussian_model(μ0, Σ0_pd, A, b, Q_pd, H, c, R_pd)
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
    gt_mean = β_closed / (α_closed - 1)


    function model_builder(θ)
        if θ[1] .<= 0.0
            return ZeroLikelihoodModel()
        end
        return create_homogeneous_linear_gaussian_model(
            μ0, Σ0_pd, A, b, 
            Distributions.PDMat(ensure_posdef(Matrix(θ[1] .* Q))), 
            H, c, 
            Distributions.PDMat(ensure_posdef(Matrix(θ[1] .* R)))
        )
    end

    invQ = inv(Q)
    invR = inv(R)
    
    function q_sampler(ref_traj, rng, xs)
        # Residuals: process x_t - (A x_{t-1} + b), observation y_t - (H x_t + c)
        proc_res = [ref_traj[t] .- (A * ref_traj[t - 1] .+ b) for t in (firstindex(ref_traj) + 1):lastindex(ref_traj)]
        obs_res = [ys[t] .- (H * ref_traj[t] .+ c) for t in firstindex(ys):lastindex(ys)]

        # Sum of Mahalanobis distances
        proc_ss = sum(r -> dot(r, invQ * r), proc_res)
        obs_ss = sum(r -> dot(r, invR * r), obs_res)
        ss = proc_ss + obs_ss

        n = length(proc_res) * Dx + length(obs_res) * Dy

        α_post = α + n / 2
        β_post = β + ss / 2

        return SVector{1}(rand(rng, InverseGamma(α_post, β_post)))
    end

    # Setup AbstractMCMC model
    prior_logpdf(θ) = (θ[1] <= 0) ? -Inf : logpdf(σ2_prior, θ[1])
    model = ParameterisedSSM(model_builder, prior_logpdf)
    bf = BF(N_particles; threshold=1.0)

    θ_curr = [β / (α - 1)]

    
    sampler = begin
        m_curr = model_builder(θ_curr)
        
        N_est = if TUNE_PARTICLES
            n_est, _ = estimate_particle_count(rng, m_curr, ys, N -> BF(N; threshold=1.0); initial_N=N_particles)
            n_est
        else
            N_particles
        end
        
        bf_tuned = BF(N_est; threshold=1.0)
        
        if sampler_type == PMMH_TYPE
            PMMH(bf_tuned; d=1, adapt_end=N_burnin + 100)
        elseif sampler_type == PGIBBS_TYPE
            PGibbs(bf_tuned, q_sampler)
        elseif sampler_type == EHMM_TYPE
             # Same as drift: min(256, max(16, N_est ÷ 50))
             L_val = min(256, max(16, N_est ÷ 50))
             EHMM(bf_tuned, q_sampler, L_val)
        end
    end

    samples = nothing
    elapsed_time = @elapsed begin
        samples = sample(rng, model, sampler, ys; n_samples=N_sample, n_burnin=N_burnin, init_θ=SVector{1}(θ_curr))
    end
    
    est_mean_vec = mean(samples)
    
    sq_error = sum(abs2, est_mean_vec .- gt_mean)
    
    ess_val = ess(stack(samples)')

    # output samples samples.csv
    open("samples.csv", "w") do io
        for sample in samples
            println(io, sample[1])
        end
    end

    return (sq_error, ess_val, elapsed_time)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment(1, :PGIBBS)
end

end # module