using Distributions
using LinearAlgebra
using Random
using SSMProblems
import GeneralisedFilters: calc_initial, calc_params

# Ensure covariance is positive definite for sampling; add jitter if needed.
function _ensure_posdef(cov::AbstractMatrix{T}) where {T}
    cov_sym = Symmetric((cov + cov') / 2)
    I_mat = Matrix{T}(I, size(cov_sym)...)
    jitter = eps(real(T))^(0.5)
    for _ in 1:6
        try
            cholesky(cov_sym)
            return cov_sym
        catch
            cov_sym = Symmetric(cov_sym + jitter * I_mat)
            jitter *= 10
        end
    end
    return cov_sym
end

"""
    sample_latent_trajectory(rng, model, ys)

Draw a latent state trajectory `x_{1:K}` from `p(x_{1:K} | y_{1:K}, model)` for a
linear Gaussian state space model using Kalman forward filtering followed by
backward simulation (FFBS).
"""
function sample_latent_trajectory(rng::AbstractRNG, model, ys)
    Tsteps = length(ys)
    prior = SSMProblems.prior(model)
    dyn = SSMProblems.dyn(model)
    obs = SSMProblems.obs(model)

    μ0, Σ0 = calc_initial(prior)
    As = Vector{AbstractMatrix}(undef, Tsteps)
    bs = Vector{AbstractVector}(undef, Tsteps)
    Qs = Vector{AbstractMatrix}(undef, Tsteps)
    Hs = Vector{AbstractMatrix}(undef, Tsteps)
    cs = Vector{AbstractVector}(undef, Tsteps)
    Rs = Vector{AbstractMatrix}(undef, Tsteps)
    for t in 1:Tsteps
        As[t], bs[t], Qs[t] = calc_params(dyn, t)
        Hs[t], cs[t], Rs[t] = calc_params(obs, t)
    end

    m_pred = μ0
    P_pred = _ensure_posdef(Σ0)

    m_filt = Vector{Vector{eltype(μ0)}}(undef, Tsteps)
    P_filt = Vector{Matrix{eltype(μ0)}}(undef, Tsteps)
    P_pred_store = Vector{Matrix{eltype(μ0)}}(undef, Tsteps)

    for t in 1:Tsteps
        A_t, b_t = As[t], bs[t]
        H_t, c_t, R_t = Hs[t], cs[t], Rs[t]

        P_pred_store[t] = P_pred
        S = H_t * P_pred * H_t' + R_t
        K_gain = P_pred * H_t' * inv(S)
        innovation = ys[t] - (H_t * m_pred + c_t)
        m_f = m_pred + K_gain * innovation
        P_f = (I - K_gain * H_t) * P_pred * (I - K_gain * H_t)' + K_gain * R_t * K_gain'
        m_filt[t] = m_f
        P_filt[t] = _ensure_posdef(P_f)

        if t < Tsteps
            m_pred = A_t * m_f + b_t
            P_pred = A_t * P_filt[t] * A_t' + Qs[t]
            P_pred = _ensure_posdef(P_pred)
        end
    end

    xs = Vector{Vector{eltype(μ0)}}(undef, Tsteps)
    xs[Tsteps] = rand(rng, MvNormal(m_filt[Tsteps], P_filt[Tsteps]))
    for t in (Tsteps - 1):-1:1
        A_t, b_t = As[t], bs[t]
        P_f = P_filt[t]
        m_f = m_filt[t]
        P_pred_next = P_pred_store[t + 1]
        C = P_f * A_t' * inv(P_pred_next)
        mean = m_f + C * (xs[t + 1] - (A_t * m_f + b_t))
        cov = _ensure_posdef(P_f - C * P_pred_next * C')
        xs[t] = rand(rng, MvNormal(mean, cov))
    end

    return xs
end

"""
    ehmm(N_steps, N_burnin, θ0, model_builder, θ_sampler, ys, rng; ref_traj=nothing)

Run an Embedded HMM-style sampler that alternates between drawing a latent trajectory
via FFBS and updating parameters through `θ_sampler`. Returns a vector of parameter samples
after burn-in.
"""
function ehmm(
    N_steps, N_burnin,
    θ0,
    model_builder::Function,
    θ_sampler::Function,
    ys, rng::AbstractRNG;
    ref_traj=nothing,
)
    samples = Vector{typeof(θ0)}(undef, N_steps - N_burnin)
    θ = deepcopy(θ0)
    traj = ref_traj

    @showprogress for i in 1:N_steps
        model = model_builder(θ)
        traj = sample_latent_trajectory(rng, model, ys)
        θ = θ_sampler(traj, rng, θ)

        if i > N_burnin
            samples[i - N_burnin] = deepcopy(θ)
        end
    end

    return samples
end
