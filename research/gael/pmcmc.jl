"""
    pmmh(N_steps, N_burnin, θ0, model_builder, prior, ys, bf, rng, proposer; log_proposal_ratio)

Generic PMMH routine where the caller supplies a `model_builder` that maps a
parameter value `θ` to a model compatible with `GeneralisedFilters.filter`, a
`prior` (either a callable returning log-density or a `Distribution`), and a
`proposer` function `(rng, θ) -> θ_prop`. The optional `log_proposal_ratio`
keyword allows non-symmetric proposals by returning
`log q(θ | θ_prop) - log q(θ_prop | θ)`.
"""
function pmmh(
    N_steps, N_burnin,
    θ0,
    model_builder::Function,
    prior,
    ys, bf, rng::AbstractRNG,
    proposer::Function = (rng, θ) -> θ .+ randn(rng, length(θ));
    log_proposal_ratio::Function = (_, _) -> 0.0,
)
    samples = Vector{typeof(θ0)}(undef, N_steps - N_burnin)
    logprior(θ) = prior isa Function ? prior(θ) : logpdf(prior, θ)

    θ = deepcopy(θ0)

    # Initial likelihood at starting θ
    model_init = model_builder(θ)
    _, loglik_curr = GeneralisedFilters.filter(model_init, bf, ys)

    accepted = 0

    @showprogress for i in 1:N_steps
        ### θ | y (PMMH)

        # Propose new parameter θ'
        θ_prop = proposer(rng, θ)

        # Run PF to get log p̂(y | θ')
        model_prop = model_builder(θ_prop)
        _, loglik_prop = GeneralisedFilters.filter(model_prop, bf, ys)

        # MH acceptance ratio
        log_alpha = (loglik_prop + logprior(θ_prop)) - (loglik_curr + logprior(θ))
        log_alpha += log_proposal_ratio(θ, θ_prop)  # zero for symmetric proposals

        if log(rand(rng)) < log_alpha
            θ = θ_prop
            loglik_curr = loglik_prop
            accepted += 1
        end

        if i > N_burnin
            samples[i - N_burnin] = deepcopy(θ)
        end
    end

    acceptance_rate = accepted / N_steps
    println("Acceptance rate: ", acceptance_rate)

    return samples
end

"""
    pgibbs(N_steps, N_burnin, θ0, model_builder, θ_sampler, ys, bf, rng; ref_traj=nothing)

Generic Particle Gibbs routine. The caller provides:
- `model_builder(θ)` returning a model for `GeneralisedFilters.filter`
- `θ_sampler(ref_traj, rng, θ_curr)` returning a new draw of θ | x, y
"""
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
        # Create model
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
