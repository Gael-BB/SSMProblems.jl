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


using OffsetArrays

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
    
    n_accepted = 0
    @showprogress for i in 1:(n_samples + n_burnin)
        # Propose
        θ_prop = θ .+ (L * randn(rng, d))
        
        m_prop = model.model_builder(θ_prop)
        _, loglik_prop = GeneralisedFilters.filter(rng, m_prop, sampler.filter_algo, observations)
        
        log_alpha = (loglik_prop + logprior(θ_prop)) - (loglik_curr + logprior(θ))
        
        if log(rand(rng)) < log_alpha
            θ = θ_prop
            loglik_curr = loglik_prop
            n_accepted += 1
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
    
    println("Acceptance rate: ", n_accepted / (n_samples + n_burnin))
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

struct EHMM{F<:GeneralisedFilters.AbstractFilter, S<:Function} <: AbstractMCMC.AbstractSampler
    filter_algo::F
    θ_sampler::S
    L::Int
end

function EHMM(filter_algo, θ_sampler; L=15)
    return EHMM(filter_algo, θ_sampler, L)
end


function log_transition_density(model::StateSpaceModel, t, x_next, x_curr)
    # Generic transition density
    # Try to use SSMProblems interface first if available
    # For now, we rely on checking LinearGaussian properties or falling back
    
    dyn = SSMProblems.dyn(model)
    # Check if dyn has A, b, Q fields (Linear Gaussian)
    if hasproperty(dyn, :A) && hasproperty(dyn, :b) && hasproperty(dyn, :Q)
        # We assume time-invariant or we would need to access A[t] etc if they are arrays
        # But usually dyn.A is the matrix itself.
        # If the model is time-varying, A might be a function or vector.
        # For this research code, we assume HomogeneousLinearGaussianLatentDynamics
        dist = MvNormal(dyn.A * x_curr + dyn.b, dyn.Q)
        return logpdf(dist, x_next)
    else
         error("Generic transition density not implemented for $(typeof(dyn)). Only HomogeneousLinearGaussianLatentDynamics is supported in this research implementation.")
    end
end

function log_observation_density(model::StateSpaceModel, t, x, y)
    obs = SSMProblems.obs(model)
    if hasproperty(obs, :H) && hasproperty(obs, :c) && hasproperty(obs, :R)
        dist = MvNormal(obs.H * x + obs.c, obs.R)
        return logpdf(dist, y)
    else
         error("Generic observation density not implemented for $(typeof(obs)). Only HomogeneousLinearGaussianObservationProcess is supported.")
    end
end

function backward_simulation(rng::AbstractRNG, model::StateSpaceModel, particles, log_weights_T, observations)
    T = length(particles) - 1 # particles is 0:T
    
    # Pre-allocate trajectory
    traj = Vector{eltype(particles[T])}(undef, T + 1)
    
    # 1. Select x_T using the final weights from the filter
    ps_T = particles[T]
    w_T = softmax(log_weights_T)
    idx = sample(rng, 1:length(ps_T), Weights(w_T))
    traj[T + 1] = ps_T[idx]
    
    # 2. Backward pass
    for t in (T-1):-1:0
        x_next = traj[t+2] # x_{t+1}
        y_curr = observations[t + 1] # Indexing observations? ys is 1:T. t is 0:T-1.
        # usually observations corresponds to time 1:T.
        # time t here is 0..T-1.
        # wait. particles[t] is state at time t.
        # if t=0, it's prior. usually no observation at t=0?
        # Standard filter: 0 is prior. 1..T are steps.
        # BUT run_scale.jl ys is 1:K (K steps).
        # filter runs for length(ys).
        # particles has T+1 length (0 to T).
        # We backward sample from T-1 down to 0.
        # At time t, we look at particles[t].
        # We need weight w_t.
        # w_t \propto p(y_t | x_t) * w_{t-1}.
        # If t=0, usually w_0 is uniform (prior). No observation at t=0 unless specified.
        # In run_scale, μ0, Σ0 is prior. observations start at t=1.
        
        ps = particles[t]
        log_ws = Vector{Float64}(undef, length(ps))
        
        for k in eachindex(ps)
            # Recompute weight: w_t \propto p(y_t | x_t).
            # If t=0, weight is 1 (log_w = 0).
            log_w_particle = (t == 0) ? 0.0 : log_observation_density(model, t, ps[k], observations[t])
            
            # BS weight: w_t * p(x_{t+1}|x_t)
            log_ws[k] = log_w_particle + log_transition_density(model, t, x_next, ps[k])
        end
        
        # Normalize weights
        max_log_w = maximum(log_ws)
        ws = exp.(log_ws .- max_log_w)
        ws ./= sum(ws)
        
        # Sample
        idx = sample(rng, 1:length(ps), Weights(ws))
        traj[t + 1] = ps[idx]
    end
    
    return OffsetArray(traj, 0:T)
end

function embedded_hmm_sampling(rng::AbstractRNG, model::StateSpaceModel, particles, ref_traj, L::Int, observations)
    T = length(particles) - 1 # particles is 0:T
    
    # 1. Create Pools
    pools = Vector{Vector{eltype(particles[T])}}(undef, T + 1)
    
    for t in 0:T
        ps = particles[t]
        # particles in container are likely raw states (post-resampling or equal weight approximation).
        # We sample uniformly from them.
        
        # One MUST be ref_traj[t]
        current_ref = ref_traj[t]
        
        # Select L-1 from particles (uniform)
        # We use sampling WITH replacement if particles is smaller than L?
        # Usually N_particles > L.
        # We sample indices.
        
        inds = rand(rng, 1:length(ps), L - 1)
        pool_t = ps[inds]
        push!(pool_t, current_ref)
        
        pools[t + 1] = pool_t
    end
    
    # 2. Forward Pass on the Embedded HMM
    # We need to compute alpha[t][k] = P(x_t = pool[t][k] | y_{1:t})
    # alpha[0][k] = 1/L (or uniform over pool[0] if we view them as samples from prior)
    # Actually, standard HMM forward pass:
    # alpha[t][j] = p(y_t | pool[t][j]) * sum_i (alpha[t-1][i] * p(pool[t][j] | pool[t-1][i]))
    
    log_alpha = [Vector{Float64}(undef, L) for _ in 0:T]
    
    # Initialization (t=0)
    # At t=0, we have samples from prior. 
    # y_0 usually doesn't exist or is not observed in this setup (observations are 1:K).
    # So alpha[0] is uniform 1/L.
    fill!(log_alpha[1], -log(L))
    
    for t in 1:T
        # t here means time step t (1 to K). 
        # pools index is t+1 (because 1-based vector for 0:T).
        pool_prev = pools[t]     # pool at t-1
        pool_curr = pools[t + 1] # pool at t
        
        log_alpha_t = Vector{Float64}(undef, L)
        
        # Precompute observations density for current pool
        log_obs = [log_observation_density(model, t, x, observations[t]) for x in pool_curr]
        
        for j in 1:L
            x_curr = pool_curr[j]
            
            # Compute transition log-probs from all i in prev
            # log( sum_i exp(log_alpha_prev[i] + log_trans[i,j]) )
            
            log_trans_terms = Vector{Float64}(undef, L)
            for i in 1:L
                x_prev = pool_prev[i]
                log_trans_terms[i] = log_alpha[t][i] + log_transition_density(model, t - 1, x_curr, x_prev)
            end
            
            log_sum_trans = logsumexp(log_trans_terms)
            log_alpha_t[j] = log_obs[j] + log_sum_trans
        end
        
        log_alpha[t + 1] = log_alpha_t
    end
    
    # 3. Backward Sampling
    # Sample x_T from alpha[T]
    traj = Vector{eltype(pools[1])}(undef, T + 1)
    
    # Sample index at T
    w_T = softmax(log_alpha[T + 1])
    idx_T = sample(rng, 1:L, Weights(w_T))
    traj[T + 1] = pools[T + 1][idx_T]
    
    curr_idx = idx_T
    
    for t in (T-1):-1:0
        pool_curr = pools[t + 1] # pool at t
        pool_next = pools[t + 2] # pool at t+1
        
        x_next = pool_next[curr_idx]
        
        # Compute backward weights
        # p(x_t=i | x_{t+1}=curr, y_{1:t}) \propto alpha[t][i] * p(x_{t+1} | x_t)
        
        log_ws = Vector{Float64}(undef, L)
        for i in 1:L
            x_curr = pool_curr[i]
            log_ws[i] = log_alpha[t + 1][i] + log_transition_density(model, t, x_next, x_curr)
        end
        
        w_t = softmax(log_ws)
        curr_idx = sample(rng, 1:L, Weights(w_t))
        traj[t + 1] = pool_curr[curr_idx]
    end
    
    return OffsetArray(traj, 0:T)
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
    theta = init_θ === nothing ? rand(rng, model.prior) : init_θ
    samples = Vector{typeof(theta)}(undef, n_samples)
    
    # Initialization: Run a standard Particle Filter to get a valid initial trajectory
    m_init = model.model_builder(theta)
    
    # Run filter WITH callback to get history
    cb_init = GeneralisedFilters.DenseAncestorCallback(nothing)
    bf_state, loglik_curr = GeneralisedFilters.filter(
        rng, 
        m_init, 
        sampler.filter_algo, 
        observations;
        callback = cb_init
    )
    
    # Initial backward simulation to get ref_traj (standard PGBS is fine for init)
    particles = cb_init.container.particles
    final_log_weights = getfield.(bf_state.particles, :log_w)
    ref_traj = backward_simulation(rng, m_init, particles, final_log_weights, observations)
    
    @showprogress for i in 1:(n_samples + n_burnin)
        m = model.model_builder(theta)
        
        # Run Conditional Particle Filter (CSMC)
        cb = GeneralisedFilters.DenseAncestorCallback(nothing)
        
        bf_state, loglik = GeneralisedFilters.filter(
            rng, 
            m, 
            sampler.filter_algo, 
            observations; 
            ref_state = ref_traj, 
            callback = cb
        )
        
        # Embedded HMM Sampling (Pool size L)
        particles = cb.container.particles
        ref_traj = embedded_hmm_sampling(rng, m, particles, ref_traj, sampler.L, observations)
        
        # Sample Parameters given the trajectory
        theta = sampler.θ_sampler(ref_traj, rng, theta)
        
        if i > n_burnin
            samples[i - n_burnin] = deepcopy(theta)
        end
    end
    
    return samples
end

end
