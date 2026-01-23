module RunDrift

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

    μ = 0.0; σ2 = 1.0
    b_prior = MvNormal(zeros(T, Dx) .+ μ, σ2 * I)
    b_true = rand(rng, b_prior)

    # Generate model matrices/vectors
    μ0 = @SVector rand(rng, T, Dx)
    Σ0 = SMatrix{Dx, Dx}(rand_cov(rng, T, Dx))
    A = @SMatrix rand(rng, T, Dx, Dx)
    Q = SMatrix{Dx, Dx}(rand_cov(rng, T, Dx))
    H = @SMatrix rand(rng, T, Dy, Dx)
    c = @SVector rand(rng, T, Dy)
    R = SMatrix{Dy, Dy}(rand_cov(rng, T, Dy))

    # Pre-wrap covariances as PDMat (GeneralisedFilters requires AbstractPDMat)
    # We use ensure_posdef ensuring it is numerically PD
    Σ0_pd = Distributions.PDMat(ensure_posdef(Matrix(Σ0)))
    Q_pd = Distributions.PDMat(ensure_posdef(Matrix(Q)))
    R_pd = Distributions.PDMat(ensure_posdef(Matrix(R)))

    # Define full model and sample observations
    full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0_pd, A, b_true, Q_pd, H, c, R_pd)
    _, xs, ys = sample(rng, full_model, K)

    # Define augmented dynamics
    μ0_aug = vcat(μ0, b_prior.μ) # SVector concatenation
    
    # helper for block diag
    z_Dx = @SMatrix zeros(T, Dx, Dx)
    z_Dy_Dx = @SMatrix zeros(T, Dy, Dx)
    
    A_aug = SMatrix{2*Dx, 2*Dx}([
        A I;
        z_Dx I
    ])
    
    Σ0_aug = SMatrix{2*Dx, 2*Dx}([
        Σ0 z_Dx;
        z_Dx b_prior.Σ
    ])
    
    b_aug = @SVector zeros(T, 2 * Dx)
    
    Q_aug = SMatrix{2*Dx, 2*Dx}([
        Q z_Dx;
        z_Dx z_Dx
    ])

    H_aug = SMatrix{Dy, 2*Dx}([H z_Dy_Dx])

    # Wrap augmented covariances
    Σ0_aug_pd = Distributions.PDMat(ensure_posdef(Matrix(Σ0_aug)))
    Q_aug_pd = Distributions.PDMat(ensure_posdef(Matrix(Q_aug)))
    # R is same

    # Create augmented model
    aug_model = create_homogeneous_linear_gaussian_model(
        μ0_aug, Σ0_aug_pd, A_aug, b_aug, Q_aug_pd, H_aug, c, R_pd
    )
    state, _ = GeneralisedFilters.filter(rng, aug_model, KalmanFilter(), ys)
    
    gt_mean_vec = state.μ[(Dx+1):end]
    gt_mean = gt_mean_vec[1] 

    function model_builder(θ)
        θ_static = SVector{Dx}(θ)
        return create_homogeneous_linear_gaussian_model(
            μ0, Σ0_pd, A, θ_static, Q_pd, H, c, R_pd
        )
    end

    Qinv = inv(Q)
    Σ_prior_inv = inv(b_prior.Σ)
    μ_prior = b_prior.μ 
    
    function b_sampler(ref_traj, rng, xs)
        residuals = [ref_traj[t] - A * ref_traj[t-1] for t in (firstindex(ref_traj) + 1):lastindex(ref_traj)]
        n = length(residuals)
        sum_r = reduce(+, residuals)

        Σ_post = inv(n * Qinv + Σ_prior_inv)
        μ_post = Σ_post * (Qinv * sum_r + Σ_prior_inv * μ_prior)
        
        # Ensure Σ_post is PD before sampling if necessary, or let MvNormal handle it (it usually does Cholesky)
        # MvNormal(μ, Σ) expects Σ to be AbstractMatrix, if not PD it errors
        # So we better ensure it
        return SVector{Dx}(rand(rng, MvNormal(vec(μ_post), Symmetric(ensure_posdef(Σ_post)))))
    end

    # Setup AbstractMCMC model
    model = ParameterisedSSM(model_builder, b_prior)
    bf = BF(N_particles; threshold=1.0)

    sampler = begin
        # Always tune particles if TUNE_PARTICLES is true
        b_curr = b_prior.μ
        m_curr = model_builder(b_curr)
        
        N_est = if TUNE_PARTICLES
            n_est, _ = estimate_particle_count(rng, m_curr, ys, N -> BF(N; threshold=1.0); initial_N=N_particles)
            n_est
        else
            N_particles
        end
        
        # Use N_est for all algorithms
        bf_tuned = BF(N_est; threshold=1.0)
        
        if sampler_type == PMMH_TYPE
            PMMH(bf_tuned; d=length(b_prior), adapt_end=N_burnin + 100) # Ensure adaptation covers burnin
        elseif sampler_type == PGIBBS_TYPE
            PGibbs(bf_tuned, b_sampler)
        elseif sampler_type == EHMM_TYPE
            # Heuristic for L: roughly sqrt or fixed size, but ensure it's not tiny.
            # user suggested max particles could be 1M. 
            # If N=1M, L should probably be smaller than N. 
            # Let's pick L = min(256, max(16, N_est ÷ 50)) 
            L_val = min(256, max(16, N_est ÷ 50))
            EHMM(bf_tuned, b_sampler, L_val)
        end
    end

    samples = nothing
    elapsed_time = @elapsed begin
        samples = sample(rng, model, sampler, ys; n_samples=N_sample, n_burnin=N_burnin, init_θ=SVector{Dx}(b_prior.μ))
    end

    est_mean_vec = mean(samples)

    diff = est_mean_vec - gt_mean_vec
    sq_error = sum(abs2, diff)
    
    ess_val = ess(stack(samples)')

    # output samples samples.csv
    open("samples.csv", "w") do io
        for sample in samples
            for i in 1:length(sample)
                print(io, sample[i], ",")
            end
            println(io)
        end
    end

    return (sq_error, ess_val, elapsed_time)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment(1, :PGIBBS)
end

end # module