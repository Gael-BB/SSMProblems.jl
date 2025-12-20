function pmmh(
    N_steps, N_burnin,
    θ0,
    model_builder::Function,
    prior,
    ys, bf, rng::AbstractRNG;

    adapt_cov::Bool = true,
    adapt_start::Int = 0,
    adapt_end::Int = N_burnin,
    adapt_interval::Int = 1,
    Σ0::Union{Nothing,AbstractMatrix} = nothing,
    scale::Union{Nothing,Real} = nothing,
    jitter::Real = 1e-8,
    proposer::Union{Nothing,Function} = nothing,
    log_proposal_ratio::Function = (_, _) -> 0.0,
)
    function _chol_with_jitter(Σ::AbstractMatrix; jitter=1e-8, max_tries::Int=8)
        d = size(Σ, 1)
        @assert size(Σ, 2) == d
        Σs = Symmetric(Matrix(Σ))
        ϵ = jitter
        for _ in 1:max_tries
            try
                return cholesky(Σs + ϵ * I, check=true).L
            catch
                ϵ *= 10
            end
        end
        # last resort: check=false (still add jitter)
        return cholesky(Σs + ϵ * I, check=false).L
    end
    samples = Vector{typeof(θ0)}(undef, N_steps - N_burnin)
    logprior(θ) = prior isa Function ? prior(θ) : logpdf(prior, θ)

    θ = deepcopy(θ0)
    d = length(θ)

    # scaling heuristic for RWM (can override)
    s = scale === nothing ? (2.38^2) / d : float(scale)

    # initial covariance
    Σ_prop = if Σ0 === nothing
        # reasonable default: small isotropic covariance
        1e-2 * I(d)
    else
        Matrix(Σ0)
    end

    # online covariance stats (Welford)
    n_stats = 0
    μ = zeros(eltype(θ), d)
    C = zeros(eltype(θ), d, d)  # sum of outer products for covariance

    function update_stats!(θ_curr)
        # update using the *current chain state* (including repeats)
        # θ_curr is assumed indexable length d
        n_stats += 1
        if n_stats == 1
            @inbounds for j in 1:d
                μ[j] = θ_curr[j]
            end
            return
        end
        # Welford update for vector mean/cov
        δ = similar(μ)
        @inbounds for j in 1:d
            δ[j] = θ_curr[j] - μ[j]
            μ[j] += δ[j] / n_stats
        end
        @inbounds for a in 1:d, b in 1:d
            C[a, b] += δ[a] * (θ_curr[b] - μ[b])
        end
        return
    end

    # current proposal "sqrt" (Cholesky factor)
    L = _chol_with_jitter(Σ_prop; jitter=jitter)

    # default proposer uses (possibly adaptive) covariance unless user overrides
    local_proposer = proposer === nothing ? ((rng, θ) -> θ .+ (L * randn(rng, d))) : proposer

    model_init = model_builder(θ)
    _, loglik_curr = GeneralisedFilters.filter(model_init, bf, ys)

    accepted = 0

    @showprogress for i in 1:N_steps
        # --- propose θ' ---
        θ_prop = local_proposer(rng, θ)

        model_prop = model_builder(θ_prop)
        _, loglik_prop = GeneralisedFilters.filter(model_prop, bf, ys)

        log_alpha = (loglik_prop + logprior(θ_prop)) - (loglik_curr + logprior(θ))
        log_alpha += log_proposal_ratio(θ, θ_prop)  # 0 for symmetric proposals

        if log(rand(rng)) < log_alpha
            θ = θ_prop
            loglik_curr = loglik_prop
            accepted += 1
        end

        # --- update adaptive covariance using current θ ---
        if adapt_cov
            update_stats!(θ)

            # adapt only in [adapt_start, adapt_end], and only every adapt_interval steps
            if (i >= adapt_start) && (i <= adapt_end) && (i % adapt_interval == 0) && (n_stats > 1)
                Σ_emp = C / (n_stats - 1)
                Σ_prop = s * Matrix(Symmetric(Σ_emp)) + jitter * I(d)
                L = _chol_with_jitter(Σ_prop; jitter=jitter)

                # refresh proposer closure to use latest L (if proposer not user-supplied)
                if proposer === nothing
                    local_proposer = (rng, θ) -> θ .+ (L * randn(rng, d))
                end
            end
        end

        if i > N_burnin
            samples[i - N_burnin] = deepcopy(θ)
        end
    end

    acceptance_rate = accepted / N_steps
    println("Acceptance rate: ", acceptance_rate)

    return samples
end

function pgibbs(
    N_steps, N_burnin,
    θ0,
    model_builder::Function,
    θ_sampler::Function,
    ys, bf, rng::AbstractRNG;
    ref_traj=nothing,
)
    samples = Vector{typeof(θ0)}(undef, N_steps - N_burnin)
    θ = deepcopy(θ0)
    ref_state = ref_traj

    @showprogress for i in 1:N_steps
        ### x | θ, y (CSMC)
        model = model_builder(θ)

        cb = GeneralisedFilters.DenseAncestorCallback(nothing)
        bf_state, _ = GeneralisedFilters.filter(model, bf, ys; ref_state=ref_state, callback=cb)
        weights = softmax(getfield.(bf_state.particles, :log_w))
        sampled_idx = sample(1:length(weights), Weights(weights))
        ref_state = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)

        ### θ | x, y
        θ = θ_sampler(ref_state, rng, θ)

        if i > N_burnin
            samples[i - N_burnin] = deepcopy(θ)
        end
    end

    return samples
end