using GeneralisedFilters
using Distributions
using LinearAlgebra
using LogExpFunctions
using ProgressMeter
using OffsetArrays
using Random
using SSMProblems
using StatsBase
using Plots

function rand_cov(rng::AbstractRNG, T::Type{<:Real}, d::Int)
    Σ = rand(rng, T, d, d)
    return Σ * Σ'
end

SEED = 69420
Dx = 1
Dy = 1
K = 3
T = Float64
N_particles = 100
N_burnin = 1
N_sample = 1000

rng = MersenneTwister(SEED)

b_prior = MvNormal(zeros(T, Dx), 1I)
b_true = rand(rng, b_prior)
println("True b: ", b_true)

# Generate model matrices/vectors
μ0 = rand(rng, T, Dx)
Σ0 = rand_cov(rng, T, Dx)
A = rand(rng, T, Dx, Dx)
Q = rand_cov(rng, T, Dx)
H = rand(rng, T, Dy, Dx)
c = rand(rng, T, Dy)
R = rand_cov(rng, T, Dy)

# Define full model and sample observations
full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b_true, Q, H, c, R)
_, ys = sample(rng, full_model, K)

# Define augemented dynamics
μ0_aug = [μ0; b_prior.μ]
Σ0_aug = [
    Σ0 zeros(T, Dx, Dx)
    zeros(T, Dx, Dx) b_prior.Σ
]
A_aug = [
    A I
    zeros(T, Dx, Dx) I
]
b_aug = zeros(T, 2 * Dx)
Q_aug = [
    Q zeros(T, Dx, Dx)
    zeros(T, Dx, Dx) zeros(T, Dx, Dx)
]
H_aug = [H zeros(T, Dy, Dx)]

# Calculate ground truth distribution
# TODO: Ask Tim if the kalman filter can only be applied to augmented models (ie no b vector)
aug_model = create_homogeneous_linear_gaussian_model(
    μ0_aug, Σ0_aug, A_aug, b_aug, Q_aug, H_aug, c, R
)
state, _ = GeneralisedFilters.filter(rng, aug_model, KalmanFilter(), ys)
println("Ground truth posterior mean: ", state.μ[(Dx+1):end])

function log_density(b, ref_traj, A, Q)
    log_prior = logpdf(b_prior, b)
    log_likelihood = 0.0
    for t in 0:(K-1)
        log_likelihood += logpdf(MvNormal(A * ref_traj[t] + b, Q), ref_traj[t+1])
    end
    return log_prior + log_likelihood
end

N_steps = N_burnin + N_sample
bf = BF(N_particles; threshold=1.0)

# kept for structural similarity, but unused in PMMH
ref_traj = nothing

b_samples = Vector{Vector{T}}(undef, N_sample)
b_curr = [only(b_prior.μ)]
println("Initial b: ", [only(b_prior.μ)])

# PMMH-specific bits
proposal_std = 0.5               # tune this
loglik_curr = 0.0
accepted = 0

# Initial likelihood at starting b_curr
model_init = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b_curr, Q, H, c, R)
_, loglik_init = GeneralisedFilters.filter(model_init, bf, ys)
loglik_curr = loglik_init

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
        global b_curr = b_prop
        global loglik_curr = loglik_prop
        global accepted += 1
    end

    if i > N_burnin
        b_samples[i - N_burnin] = deepcopy(b_curr)
    end
end

println("Acceptance rate: ", accepted / N_steps)
println("Posterior mean: ", mean(b_samples))

display(plot(
    only.(b_samples);
    label="Chain",
    xlabel="Iteration",
    ylabel="b_outer",
    legend=:topleft,
))