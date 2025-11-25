SEED = 1234
Dx = 1
Dy = 1
K = 10
T = Float64
N_particles = 100
N_burnin = 100
N_sample = 1000

@enum samplers PMMH PGIBBS EHMM
sampler_type::samplers = EHMM

function rand_cov(rng::AbstractRNG, T::Type{<:Real}, d::Int)
    Σ = rand(rng, T, d, d)
    return Σ * Σ'
end

function log_density(b, ref_traj, A, Q)
    log_prior = logpdf(b_prior, b)
    log_likelihood = 0.0
    for t in 0:(K-1)
        log_likelihood += logpdf(MvNormal(A * ref_traj[t] + b, Q), ref_traj[t+1])
    end
    return log_prior + log_likelihood
end