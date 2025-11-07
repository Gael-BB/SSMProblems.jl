@showprogress for i in 1:N_steps
    ### x | θ, y (CSMC)

    # Create model
    model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b_curr, Q, H, c, R)

    cb = GeneralisedFilters.DenseAncestorCallback(Vector{T})
    bf_state, _ = GeneralisedFilters.filter(model, bf, ys; ref_state=ref_traj, callback=cb)
    weights = softmax(getfield.(bf_state.particles, :log_w))
    sampled_idx = sample(1:length(weights), Weights(weights))
    global ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)

    μ_prior = only(b_prior.μ)
    σ2_prior = only(b_prior.Σ)
    σ2 = only(Q)

    xs = [only(ref_traj[t] - A * ref_traj[t - 1]) for t in 1:K]
    n = K

    μ_post = (sum(xs) / σ2 + μ_prior / σ2_prior) / (n / σ2 + 1 / σ2_prior)
    σ2_post = 1 / (n / σ2 + 1 / σ2_prior)

    global b_curr = [rand(rng, Normal(μ_post, sqrt(σ2_post)))]
    if i > N_burnin
        b_samples[i - N_burnin] = deepcopy(b_curr)
    end
end