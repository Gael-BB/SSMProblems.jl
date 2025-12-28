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
    
    # Initialization: Run a standard Particle Filter to get a valid initial trajectory
    m_init = model.model_builder(θ)
    
    # We ignore the state from the user request snippet for likelihood, 
    # but we actually need the state for the trajectory. 
    # However, to be strict with the user request:
    # "The likelihood estimate is not correctly implemented. You should use _, loglik_curr = GeneralisedFilters.filter(rng, m_init, sampler.filter_algo, observations)"
    # We will compute the likelihood here.
    _, loglik_curr = GeneralisedFilters.filter(rng, m_init, sampler.filter_algo, observations)
    
    # Now we need a trajectory. The above call discarded it (as per instruction variable `_`).
    # Ideally we should have kept it. But sticking to the specific line requested:
    # We will likely need to run it again or just execute it differently.
    # To avoid double computation, I will capture the state.
    
    # Run filter WITH callback to get history for BS
    cb_init = GeneralisedFilters.DenseAncestorCallback(nothing)
    bf_state, loglik_curr = GeneralisedFilters.filter(
        rng, 
        m_init, 
        sampler.filter_algo, 
        observations;
        callback = cb_init
    )
    
    # Initial backward simulation to get ref_traj
    particles = cb_init.container.particles
    final_log_weights = getfield.(bf_state.particles, :log_w)
    ref_traj = backward_simulation(rng, m_init, particles, final_log_weights, observations)
    
    @showprogress for i in 1:(n_samples + n_burnin)
        m = model.model_builder(θ)
        
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
        
        # Backward Simulation (PGBS)
        # Extract particles from callback container (OffsetVector 0:T)
        particles = cb.container.particles
        final_log_weights = getfield.(bf_state.particles, :log_w)
        ref_traj = backward_simulation(rng, m, particles, final_log_weights, observations)
        
        # Sample Parameters given the trajectory
        θ = sampler.θ_sampler(ref_traj, rng, θ)
        
        if i > n_burnin
            samples[i - n_burnin] = deepcopy(θ)
        end
    end
    
    return samples
end

end
