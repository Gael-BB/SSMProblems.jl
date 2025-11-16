function run_gibbs_drift(
    N_steps, N_burnin,
    μ0, Σ0, A, b, Q, H, c, R,
    prior, ys, bf, ref_traj, rng::AbstractRNG
)
    samples = Vector{typeof(b)}(undef, N_steps - N_burnin)

    @showprogress for i in 1:N_steps
        ### x | θ, y (CSMC)
        # Create model
        model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)

        cb = GeneralisedFilters.DenseAncestorCallback(Vector{T})
        bf_state, _ = GeneralisedFilters.filter(model, bf, ys; ref_state=ref_traj, callback=cb)
        weights = softmax(getfield.(bf_state.particles, :log_w))
        sampled_idx = sample(1:length(weights), Weights(weights))
        ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)

        ### θ | x, y
        μ_prior = only(prior.μ)
        σ2_prior = only(prior.Σ)
        σ2 = only(Q)

        K_start = firstindex(ref_traj)
        K_end   = lastindex(ref_traj)

        xs = [only(ref_traj[t] - A * ref_traj[t - 1]) for t in (K_start + 1):K_end]
        n = length(xs)

        μ_post = (sum(xs) / σ2 + μ_prior / σ2_prior) / (n / σ2 + 1 / σ2_prior)
        σ2_post = 1 / (n / σ2 + 1 / σ2_prior)

        b = [rand(rng, Normal(μ_post, sqrt(σ2_post)))]

        if i > N_burnin
            samples[i - N_burnin] = deepcopy(b)
        end
    end

    return samples
end

function run_gibbs_observation(
    N_steps, N_burnin,
    μ0, Σ0, A, b, Q, H, c, R,
    prior, ys, bf, ref_traj, rng::AbstractRNG
)
    # We now collect samples of Q instead of b
    samples = Vector{typeof(Q)}(undef, N_steps - N_burnin)

    # Fixed observation model pieces
    # H = I                      # Identity observation matrix
    # c = zero(μ0)               # Zero observation offset
    # R = zeros(eltype(Σ0), size(H, 1), size(H, 1))  # Zero observation noise

    @showprogress for i in 1:N_steps
        ### x | θ, y (CSMC)
        # Create model with current Q (and fixed b)
        model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Matrix(Diagonal(Q)), H, c, R)

        cb = GeneralisedFilters.DenseAncestorCallback(Vector{T})
        bf_state, _ = GeneralisedFilters.filter(model, bf, ys; ref_state=ref_traj, callback=cb)
        weights = softmax(getfield.(bf_state.particles, :log_w))
        sampled_idx = sample(1:length(weights), Weights(weights))
        ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)

        ### θ = Q | x, y
        # Prior: Q ~ InverseGamma(α_prior, β_prior)
        α_prior = only(prior.invd.α)
        β_prior = only(prior.θ)

        K_start = firstindex(ref_traj)
        K_end   = lastindex(ref_traj)

        # State innovation residuals: x_t - A x_{t-1} - b
        xs = [only(ref_traj[t] - A * ref_traj[t - 1] .- b) for t in (K_start + 1):K_end]
        n = length(xs)

        ss = sum(abs2, xs)  # sum of squared residuals

        # Conjugate IG posterior
        α_post = α_prior + n / 2
        β_post = β_prior + ss / 2

        σ2 = rand(rng, InverseGamma(α_post, β_post))

        # Keep Q in the same "shape" as originally (1-element container)
        Q = [σ2]

        if i > N_burnin
            samples[i - N_burnin] = deepcopy(Q)
        end
    end

    return samples
end