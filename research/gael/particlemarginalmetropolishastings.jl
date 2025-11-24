function run_pmmh_drift(
    N_steps, N_burnin,
    μ0, Σ0, A, b, Q, H, c, R,
    prior, ys, bf, ref_traj, rng::AbstractRNG,
    proposal_std::Real = 2.2
)
    # Storage for posterior samples of b after burn-in
    samples = Vector{typeof(b)}(undef, N_steps - N_burnin)

    # Initial likelihood at starting b
    model_init = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    _, loglik_curr = GeneralisedFilters.filter(model_init, bf, ys)

    accepted = 0

    @showprogress for i in 1:N_steps
        ### θ | y (PMMH)

        # ------------------------
        # Propose new parameter b'
        # ------------------------
        b_prop = [rand(rng, Normal(only(b), proposal_std))]

        # ------------------------
        # Run PF to get log p̂(y | b')
        # ------------------------
        model_prop = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b_prop, Q, H, c, R)
        _, loglik_prop = GeneralisedFilters.filter(model_prop, bf, ys)

        # ------------------------
        # MH acceptance ratio (symmetric proposal)
        # ------------------------
        log_prior_curr = logpdf(prior, b)
        log_prior_prop = logpdf(prior, b_prop)

        log_alpha = (loglik_prop + log_prior_prop) - (loglik_curr + log_prior_curr)

        if log(rand(rng)) < log_alpha
            # accept
            b = b_prop
            loglik_curr = loglik_prop
            accepted += 1
        end

        if i > N_burnin
            b_samples[i - N_burnin] = deepcopy(b)
        end
    end

    acceptance_rate = accepted / N_steps
    println("Acceptance rate: ", acceptance_rate)

    return b_samples
end

function run_pmmh_observation(
    N_steps, N_burnin,
    μ0, Σ0, A, b, Q_init, H, c, R,
    prior, ys, bf, ref_traj, rng::AbstractRNG,
    proposal_std::Real = 0.00001
)
    # State dimension
    d = length(μ0)

    # Coerce Q into a scalar (allow passing length-1 vectors)
    Q = Q_init isa AbstractArray ? only(Q_init) : Q_init
    Q = float(Q)  # make sure it's a floating point

    # Storage for posterior samples of scalar Q after burn-in
    Q_samples = Vector{Float64}(undef, N_steps - N_burnin)

    # Helper: covariance matrix from scalar Q
    Qmat(q::Real) = Matrix(q * I(d))

    # Initial likelihood at starting Q
    model_init = create_homogeneous_linear_gaussian_model(
        μ0, Σ0, A, b, Qmat(Q), H, c, R
    )
    _, loglik_curr = GeneralisedFilters.filter(model_init, bf, ys)

    accepted = 0

    @showprogress for i in 1:N_steps
        ### θ | y (PMMH) for Q (scalar variance)

        # Propose new scalar Q'
        Q_prop = rand(rng, Normal(Q, proposal_std))

        # If Q is a variance, you may want to enforce positivity:
        # if Q_prop <= 0
        #     continue
        # end

        # Run filter to get log p̂(y | Q')

        # Check positivity of Q_prop (variance must be > 0)
        if Q_prop > 0
            model_prop = create_homogeneous_linear_gaussian_model(
                μ0, Σ0, A, b, Qmat(Q_prop), H, c, R
            )
            _, loglik_prop = GeneralisedFilters.filter(model_prop, bf, ys)

            # MH acceptance ratio (symmetric proposal)
            log_prior_curr = logpdf(prior, Q)
            log_prior_prop = logpdf(prior, Q_prop)

            log_alpha = (loglik_prop + log_prior_prop) - (loglik_curr + log_prior_curr)

            if log(rand(rng)) < log_alpha
                Q = Q_prop
                loglik_curr = loglik_prop
                accepted += 1
            end
        end

        if i > N_burnin
            Q_samples[i - N_burnin] = Q
        end
    end

    acceptance_rate = accepted / N_steps
    println("Acceptance rate: ", acceptance_rate)

    return Q_samples
end