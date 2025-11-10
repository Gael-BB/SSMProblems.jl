function run_pmmh(
    N_steps, N_burnin,
    μ0, Σ0, A, c, Q, H, R,
    b_prior, ys, bf, ref_traj, b_curr, rng::AbstractRNG;
    proposal_std::Real = 0.5
)
    # Storage for posterior samples of b after burn-in
    b_samples = Vector{typeof(b_curr)}(undef, N_steps - N_burnin)

    # Initial likelihood at starting b_curr
    model_init = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b_curr, Q, H, c, R)
    _, loglik_curr = GeneralisedFilters.filter(model_init, bf, ys)

    accepted = 0

    @showprogress for i in 1:N_steps
        ### θ | y (PMMH)

        # ------------------------
        # Propose new parameter b'
        # ------------------------
        b_prop = [rand(rng, Normal(only(b_curr), proposal_std))]

        # ------------------------
        # Run PF to get log p̂(y | b')
        # ------------------------
        model_prop = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b_prop, Q, H, c, R)
        _, loglik_prop = GeneralisedFilters.filter(model_prop, bf, ys)

        # ------------------------
        # MH acceptance ratio (symmetric proposal)
        # ------------------------
        log_prior_curr = logpdf(b_prior, b_curr)
        log_prior_prop = logpdf(b_prior, b_prop)

        log_alpha = (loglik_prop + log_prior_prop) - (loglik_curr + log_prior_curr)

        if log(rand(rng)) < log_alpha
            # accept
            b_curr = b_prop
            loglik_curr = loglik_prop
            accepted += 1
        end

        if i > N_burnin
            b_samples[i - N_burnin] = deepcopy(b_curr)
        end
    end

    acceptance_rate = accepted / N_steps
    println("Acceptance rate: ", acceptance_rate)

    return b_samples
end