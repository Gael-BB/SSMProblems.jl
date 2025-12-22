module Algorithms

using AbstractMCMC
using Random
using Distributions
using LinearAlgebra
using GeneralisedFilters
using SSMProblems
using ProgressMeter
using StatsBase
using LogExpFunctions
using ..Models
using ..Utils

export PMMH, PGibbs, EHMM

# Helper for Cholesky with jitter
function _chol_with_jitter(Σ::AbstractMatrix; jitter=1e-8, max_tries::Int=8)
    d = size(Σ, 1)
    Σs = Symmetric(Matrix(Σ))
    ϵ = jitter
    for _ in 1:max_tries
        try
            return cholesky(Σs + ϵ * I, check=true).L
        catch
            ϵ *= 10
        end
    end
    return cholesky(Σs + ϵ * I, check=false).L
end

## PMMH ########################################################################

struct PMMH{F<:GeneralisedFilters.AbstractFilter} <: AbstractMCMC.AbstractSampler
    filter_algo::F
    Σ_prop::AbstractMatrix
    scale::Float64
    adapt::Bool
    adapt_start::Int
    adapt_end::Int
    adapt_interval::Int
    jitter::Float64
end

function PMMH(filter_algo; 
    Σ0 = nothing, 
    scale = nothing, 
    adapt = true, 
    adapt_start = 0, 
    adapt_end = 1000, 
    adapt_interval = 1,
    jitter = 1e-8,
    d = 1
)
    Σ_prop = Σ0 === nothing ? 1e-2 * I(d) : Matrix(Σ0)
    s = scale === nothing ? (2.38^2) / d : float(scale)
    return PMMH(filter_algo, Σ_prop, s, adapt, adapt_start, adapt_end, adapt_interval, jitter)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::ParameterisedSSM,
    sampler::PMMH,
    observations::AbstractVector;
    n_samples::Int,
    n_burnin::Int = 0,
    init_θ = nothing,
    kwargs...
)
    θ = init_θ === nothing ? rand(rng, model.prior) : init_θ
    d = length(θ)
    
    samples = Vector{typeof(θ)}(undef, n_samples)
    
    logprior(θ) = model.prior isa Function ? model.prior(θ) : logpdf(model.prior, θ)
    
    # Online stats for adaptation
    n_stats = 0
    μ_acc = zeros(eltype(θ), d)
    C_acc = zeros(eltype(θ), d, d)
    
    L = _chol_with_jitter(sampler.Σ_prop; jitter=sampler.jitter)
    
    m_init = model.model_builder(θ)
    _, loglik_curr = GeneralisedFilters.filter(rng, m_init, sampler.filter_algo, observations)
    
    @showprogress for i in 1:(n_samples + n_burnin)
        # Propose
        θ_prop = θ .+ (L * randn(rng, d))
        
        m_prop = model.model_builder(θ_prop)
        _, loglik_prop = GeneralisedFilters.filter(rng, m_prop, sampler.filter_algo, observations)
        
        log_alpha = (loglik_prop + logprior(θ_prop)) - (loglik_curr + logprior(θ))
        
        if log(rand(rng)) < log_alpha
            θ = θ_prop
            loglik_curr = loglik_prop
        end
        
        # Adaptation
        if sampler.adapt
            n_stats += 1
            if n_stats == 1
                μ_acc .= θ
            else
                δ = θ .- μ_acc
                μ_acc .+= δ ./ n_stats
                C_acc .+= δ * (θ .- μ_acc)'
            end
            
            if (i >= sampler.adapt_start) && (i <= sampler.adapt_end) && (i % sampler.adapt_interval == 0) && (n_stats > 1)
                Σ_emp = C_acc ./ (n_stats - 1)
                Σ_prop = sampler.scale * Matrix(Symmetric(Σ_emp)) + sampler.jitter * I(d)
                L = _chol_with_jitter(Σ_prop; jitter=sampler.jitter)
            end
        end
        
        if i > n_burnin
            samples[i - n_burnin] = deepcopy(θ)
        end
    end
    
    return samples
end

## PGibbs ######################################################################

struct PGibbs{F<:GeneralisedFilters.AbstractFilter, S<:Function} <: AbstractMCMC.AbstractSampler
    filter_algo::F
    θ_sampler::S
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::ParameterisedSSM,
    sampler::PGibbs,
    observations::AbstractVector;
    n_samples::Int,
    n_burnin::Int = 0,
    init_θ = nothing,
    ref_traj = nothing,
    kwargs...
)
    θ = init_θ === nothing ? rand(rng, model.prior) : init_θ
    samples = Vector{typeof(θ)}(undef, n_samples)
    ref_state = ref_traj
    
    @showprogress for i in 1:(n_samples + n_burnin)
        m = model.model_builder(θ)
        
        cb = GeneralisedFilters.DenseAncestorCallback(nothing)
        bf_state, _ = GeneralisedFilters.filter(rng, m, sampler.filter_algo, observations; ref_state=ref_state, callback=cb)
        
        w = softmax(getfield.(bf_state.particles, :log_w))
        idx = sample(1:length(w), Weights(w))
        ref_state = GeneralisedFilters.get_ancestry(cb.container, idx)
        
        θ = sampler.θ_sampler(ref_state, rng, θ)
        
        if i > n_burnin
            samples[i - n_burnin] = deepcopy(θ)
        end
    end
    
    return samples
end

## EHMM ########################################################################

struct EHMM{S<:Function} <: AbstractMCMC.AbstractSampler
    θ_sampler::S
end

function sample_latent_trajectory(rng::AbstractRNG, model::StateSpaceModel, ys)
    Tsteps = length(ys)
    prior = SSMProblems.prior(model)
    dyn = SSMProblems.dyn(model)
    obs = SSMProblems.obs(model)

    μ0, Σ0 = GeneralisedFilters.calc_initial(prior)
    
    # This assumes Linear Gaussian for now as in the original ehmm.jl
    # In a more general version, this would use GeneralisedFilters.filter
    
    As = [GeneralisedFilters.calc_params(dyn, t)[1] for t in 1:Tsteps]
    bs = [GeneralisedFilters.calc_params(dyn, t)[2] for t in 1:Tsteps]
    Qs = [GeneralisedFilters.calc_params(dyn, t)[3] for t in 1:Tsteps]
    Hs = [GeneralisedFilters.calc_params(obs, t)[1] for t in 1:Tsteps]
    cs = [GeneralisedFilters.calc_params(obs, t)[2] for t in 1:Tsteps]
    Rs = [GeneralisedFilters.calc_params(obs, t)[3] for t in 1:Tsteps]

    m_pred = μ0
    P_pred = ensure_posdef(Σ0)

    m_filt = [zeros(length(μ0)) for _ in 1:Tsteps]
    P_filt = [zeros(length(μ0), length(μ0)) for _ in 1:Tsteps]
    P_pred_store = [zeros(length(μ0), length(μ0)) for _ in 1:Tsteps]

    for t in 1:Tsteps
        P_pred_store[t] = P_pred
        S = Hs[t] * P_pred * Hs[t]' + Rs[t]
        K = P_pred * Hs[t]' / cholesky(S)
        innov = ys[t] - (Hs[t] * m_pred + cs[t])
        
        m_f = m_pred + K * innov
        P_f = (I - K * Hs[t]) * P_pred * (I - K * Hs[t])' + K * Rs[t] * K'
        
        m_filt[t] = m_f
        P_filt[t] = ensure_posdef(P_f)

        if t < Tsteps
            m_pred = As[t] * m_f + bs[t]
            P_pred = ensure_posdef(As[t] * P_filt[t] * As[t]' + Qs[t])
        end
    end

    xs = Vector{Vector{eltype(μ0)}}(undef, Tsteps)
    xs[Tsteps] = rand(rng, MvNormal(m_filt[Tsteps], P_filt[Tsteps]))
    for t in (Tsteps - 1):-1:1
        P_f, m_f = P_filt[t], m_filt[t]
        P_p_next = P_pred_store[t + 1]
        C = P_f * As[t]' / cholesky(P_p_next)
        
        mean = m_f + C * (xs[t + 1] - (As[t] * m_f + bs[t]))
        cov = ensure_posdef(P_f - C * P_p_next * C')
        xs[t] = rand(rng, MvNormal(mean, cov))
    end

    return xs
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::ParameterisedSSM,
    sampler::EHMM,
    observations::AbstractVector;
    n_samples::Int,
    n_burnin::Int = 0,
    init_θ = nothing,
    kwargs...
)
    θ = init_θ === nothing ? rand(rng, model.prior) : init_θ
    samples = Vector{typeof(θ)}(undef, n_samples)
    
    @showprogress for i in 1:(n_samples + n_burnin)
        m = model.model_builder(θ)
        traj = sample_latent_trajectory(rng, m, observations)
        θ = sampler.θ_sampler(traj, rng, θ)
        
        if i > n_burnin
            samples[i - n_burnin] = deepcopy(θ)
        end
    end
    
    return samples
end

end
