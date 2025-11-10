function run_gibbs(
    N_steps, N_burnin,
    μ0, Σ0, A, c, Q, H, R,
    b_prior, ys, bf, ref_traj, b_curr, rng::AbstractRNG
)
    b_samples = Vector{typeof(b_curr)}(undef, N_steps - N_burnin)

    @showprogress for i in 1:N_steps
        ### x | θ, y (CSMC)
        # Create model
        model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b_curr, Q, H, c, R)

        cb = GeneralisedFilters.DenseAncestorCallback(Vector{T})
        bf_state, _ = GeneralisedFilters.filter(model, bf, ys; ref_state=ref_traj, callback=cb)
        weights = softmax(getfield.(bf_state.particles, :log_w))
        sampled_idx = sample(1:length(weights), Weights(weights))
        ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)

        ### θ | x, y
        μ_prior = only(b_prior.μ)
        σ2_prior = only(b_prior.Σ)
        σ2 = only(Q)

        K_start = firstindex(ref_traj)
        K_end   = lastindex(ref_traj)

        xs = [only(ref_traj[t] - A * ref_traj[t - 1]) for t in (K_start + 1):K_end]
        n = length(xs)

        μ_post = (sum(xs) / σ2 + μ_prior / σ2_prior) / (n / σ2 + 1 / σ2_prior)
        σ2_post = 1 / (n / σ2 + 1 / σ2_prior)

        b_curr = [rand(rng, Normal(μ_post, sqrt(σ2_post)))]
        if i > N_burnin
            b_samples[i - N_burnin] = deepcopy(b_curr)
        end
    end

    return b_samples
end