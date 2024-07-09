using SparseArrays
using Distributions
using Random
using StatsBase
using Base.Threads
using JLD
using LinearAlgebra
using IterTools
using DelimitedFiles
seed = parse(Int, ARGS[1])
Random.seed!(seed)


function update_lambda(constant, count, a, b)
    return rand(Gamma(a + count, 1/(b + constant)))
end

function update_binomial(beta, constant, rho, n)
    probability = beta*(1 - rho)/(beta + rho*constant)
    return rand(Binomial(n, probability))
end

function resample_positive_latent_counts(rates) #vector of positive rates
    l_counts = zeros(length(rates))
    @views for i in eachindex(rates)
        rate = rates[i]
        l = 0
        u = 1
        if rate >= 1
            while l == 0
                l = rand(Poisson(rate))
            end
        else
            while u >= 1/(l+1)
            l = rand(Poisson(rate))
            u = rand(Uniform(0, 1))
            end
            l += 1
        end
        l_counts[i] = l
    end
    if length(l_counts) > 0
        @assert minimum(l_counts) > 0
    end
    return(l_counts)
end


function filter_rows_not_in_Y(X::Matrix, Y::Matrix) #filter out rows that are already accounted for by positive counts
    X_set = Set(eachrow(X))
    Y_set = Set(eachrow(Y))
    not_in_Y = Set(x for x in X_set if !(x in Y_set))
    result = Matrix(hcat(not_in_Y...)')
    return result
end


function update_b(p, factor_matrices, y_indices)
    M = length(factor_matrices)
    l_positives = zeros(Int, 0, M)
    Q = size(factor_matrices[1], 2)
    nonzero_L = Array{Any}(undef, M)
    for m in 1:M 
        nonzero_L[m] = []
    end
     for q in 1:Q
         for m in 1:M
            push!(nonzero_L[m], findall(!iszero, factor_matrices[m][:, q]))
        end
    end
    total = 0
    @views for q in 1:Q
        product = 1 
        @views for m in 1:M
            product *= length(nonzero_L[m][q])
        end
        total += product
    end 

    lambdas_CDKR = zeros(Int, total, M)
    rates = zeros(0)
    counter = 1
@views for q in 1:Q
    list = []
    nz = 0
    for m in 1:M
        if length(nonzero_L[m][q]) == 0
            nz = 1
        end
        push!(list, nonzero_L[m][q])
    end
    if (nz == 0)
    l = collect(IterTools.product(list...))
    l = reshape(l, prod(size(l)))
    length_list = length(l)
    matrix = vcat(map(t -> collect(t)', l)...)
    lambdas_CDKR[counter:(counter + length_list - 1), :] .= matrix
    counter += length_list
    end
end

@assert counter == (total+1)

    lambdas_CDKR = unique(lambdas_CDKR, dims=1)
    lambdas_CDKR = filter_rows_not_in_Y(lambdas_CDKR, y_indices)
    @views for i in axes(y_indices, 1)
        idx = y_indices[i, :]
        rate_vector_Q = ones(Q)
        @views for m in 1:M
            rate_vector_Q .*= factor_matrices[m][idx[m], :]
        end
        rate = sum(rate_vector_Q)
        @assert rate > 0
        l_positives = vcat(l_positives, idx')
        rates = vcat(rates, rate)
    end


    if size(lambdas_CDKR, 2) > 0
    @views for i in axes(lambdas_CDKR, 1)
        idx = lambdas_CDKR[i,:]
        rate_vector_Q = ones(Q)
        @views for m in 1:M
            rate_vector_Q .*= factor_matrices[m][idx[m], :]
        end
        rate = sum(rate_vector_Q)
        t = exp(-rate)
        @assert rate > 0
        numerator = (2*p/(1+p))*(1 - t)
        b = rand(Bernoulli(numerator/(numerator +  t)))
        if b == 1
            l_positives = vcat(l_positives, idx')
            rates = vcat(rates, rate)
        end
    end
end
    return(l_positives, rates)
end

function update_latent_cp(beta, constant, y, p)
    q = beta/(beta + constant)
    post_prob = q*(1-p)
    y_dot = sum(y)
    n = length(y)
    nonzero_y = findall(!iszero, y)
    num_nonzero = length(nonzero_y)
    zero_y = findall(iszero, y)
    free = length(zero_y)
    lv = zeros(n)
    b = rand(Binomial(free, post_prob))
    active_zero = sample(zero_y, b, replace=false)
    gamma_const = 1/(p*beta + constant)
        @views for v in active_zero
            lv[v] = rand(Gamma(1, gamma_const))
        end
        @views for v in nonzero_y
           lv[v] = rand(Gamma(1 + y[v], gamma_const))
        end
    return(lv)
end

function update_factor_cp(factor_matrices, y, p, m, c, iter)
    anneal = 0#maximum([-(iter + 1)/burn_in, 0])
    factor = factor_matrices[m]
    M = length(factor_matrices)
    K = size(factor, 2)
    V = size(factor, 1)
    @assert size(y)==(V, K)
    @views for k in 1:K
        y_vector = y[:, k]
        scaling_constant = 1
        for m_prime in 1:M
            if m_prime != m
                scaling_constant *= sum(factor_matrices[m_prime][:, k])
            end
        end
        fm = update_latent_cp(c, scaling_constant*(1-anneal), y_vector*(1-anneal), p)
        @assert length(fm)==V
        factor[:, k] = fm
    end
    return(factor)
end

function allocate_cp(nonzero_indices, nonzero_counts, factor_matrices, obs_dims) #allocate counts to latent classes
    K = size(factor_matrices[1], 2)
    M = length(factor_matrices)
    y_M = Array{Matrix{Float64}}(undef, M)
    @views for m in 1:M
        y_M[m] = zeros(obs_dims[m], K)
    end
    nonzero_counts = copy(Int.(round.(nonzero_counts)))
    @assert length(nonzero_counts) == size(nonzero_indices, 1)
    y_K = zeros(K)
    @views for j in axes(nonzero_indices, 1)
        counts = sample_count_cp(nonzero_indices[j,:], ones(K), factor_matrices, nonzero_counts[j])
        @assert length(counts)==K
        @views for m in 1:M
          y_M[m][nonzero_indices[j,m], :].+= counts
        end 
        y_K .+= counts
    end
    @assert round(sum(y_K)) == round(sum(nonzero_counts))
    @assert round(sum(y_M[2])) == round(sum(nonzero_counts))
    return (y_M)
end

function sample_count_cp(nz_ind, probs, om, nzc)
    @views for m in eachindex(om)
                theta = om[m]
                ind = nz_ind[m]
                probs .*= theta[ind, :]
            end
            s = sum(probs)
            if s == 0
                probs = ones(length(probs))/length(probs)
            else
                probs./=s
            end
        return(rand(Multinomial(nzc, probs)))
end


function update_l_indices(l_indices, l_counts, factor_matrices, obs_dims, y_indices, p_core, p_l, iter, init=false)
    M = length(factor_matrices)
    if init==false
    l_indices, rates = update_b(p_core, factor_matrices, y_indices)
    l_counts = resample_positive_latent_counts(rates)
    else
        l_indices = y_indices
        l_counts = ones(size(y_indices, 1))
    end
    l_M = allocate_cp(l_indices, l_counts, factor_matrices, obs_dims)
    for m in 1:M
        factor_matrices[m] = update_factor_cp(factor_matrices, l_M[m], p_l, m, 1, iter)
    end
    return(l_indices, l_counts, factor_matrices)
end

function update_epsilon(y)
    epsilon = zeros(size(y))
    for i in eachindex(y)
    epsilon[i] = rand(Gamma(1 + y[i], 1/(10 + prod(size(Y))/prod(size(y)))))
    end
    return(epsilon)
end

function pp_from_list(f, b, samples)
    fsample = zeros(samples)
backward = zeros(samples)
println("calculating means")
for i in 1:samples
    if length(f[i]) == 1
        fsample[i] = f[i]
        backward[i] = b[i]
    else
    fsample[i] = mean(f[i])
    backward[i] = mean(b[i])
    end
end
plot_pp(fsample, backward)
end

function adjust_Ycore(y_Q, indices_QM, lambdas_Q, factor_matrices, m_Q, gamma, constant_core)
    Y_adj = copy(y_Q)
    y_aux = copy(y_Q)
    @assert minimum(m_Q) > 0
    @views while length(Y_adj) < length(lambdas_Q)
        Y_adj = vcat(Y_adj, 0)
    end
    lambda_dot = 1
    @views for m in 1:M
        lambda_dot *= maximum(sum(factor_matrices[m], dims=1))
    end
    nz = copy(indices_QM)
    s_M = Array{Any}(undef, M)
    @views for m in 1:M
        s_M[m] = dropdims(sum(factor_matrices[m], dims=1),dims=1)
    end
    @views for q in axes(nz,1)
        ind = nz[q, :]
        mu = lambdas_Q[q]
        lambda = 1
        for m in 1:M
            lambda *= s_M[m][ind[m]]
        end
        if (lambda < lambda_dot)
        y_adj = rand(Poisson((lambda_dot - lambda)*mu))
        Y_adj[q] += y_adj
        y_aux[q] += y_adj
        end
    end
    y_indices = indices_QM[findall(!iszero, Y_adj),:]
return(Y_adj, lambda_dot, y_indices, y_aux)
end



function init_core(latent_dims, Q, dense=false)
    M = length(latent_dims)
indices = zeros(Int, Q, M)
values = zeros(Q)
q = 1
while q <= minimum([maximum(latent_dims), Q])
for m in 1:M
        indices[q, m] = mod(q, latent_dims[m]) + 1
end
        values[q] = rand(Gamma())
        q += 1
end
while q <= Q
    for m in 1:M
    indices[q, m] = rand(1:(latent_dims[m]))
    end
    values[q] = rand(Gamma())
    q += 1
end
if dense==true
    list = Array{Any}(undef, M)
    for m in 1:M
        list[m] = 1:latent_dims[m]
    end
    indices = collect(IterTools.product(list...))
    indices = reshape(indices, prod(size(indices)))
    indices = vcat(map(t -> collect(t)', indices)...)
    #indices = reshape(collect(indices), prod(latent_dims), 3)
    values = rand(Gamma(), size(indices, 1))
end
return(values, indices)
end

function init_p(K, alpha, beta)
    p = rand(Beta(alpha, beta), K)
    return(p)
end

function init_factor(V, K, constant, p, pos=true)
factor = zeros(V, K)
shape = (sum(Y)/(K*V))^0.25
for v in 1:V
    for k in 1:K
        if pos == true 
            factor[v, k] = rand(Gamma(shape))#1, 1/(p[k]*constant[k])))
        else
        m = rand(NegativeBinomial(1, p[k]))
        if (m > 0)
            factor[v, k] = rand(Gamma(m, 1/constant[k]))
        end
    end
    end
end
return(factor)
end


function update_latent(c, constant, y, p, alpha0, beta0, n, iter, core=false)
    q = c/(c + constant)
    post_prob = q*(1-p)
    y_dot = sum(y)
    anneal = 0#minimum([maximum([-iter/burn_in, 0]), 1])#.*0.01
    if (core==true)
        anneal_c = anneal
        post_prob = q*(1-p)*(1-anneal_c) + anneal_c*(1-p)*c/(c + 1)
        #println("sampling probability is $post_prob")
        nonzero_y = findall(!iszero, y)
        num_nonzero = length(nonzero_y)
        free = n - num_nonzero
    else
        post_prob = q*(1-p)*(1-anneal) + anneal*(1-p)*c/(c + 1)
        n = length(y)
        nonzero_y = findall(!iszero, y)
        num_nonzero = length(nonzero_y)
        zero_y = findall(iszero, y)
        free = length(zero_y)
        lv = zeros(n)
    end
    if free > 0
        b = rand(Binomial(free, post_prob))
    else
        b = 0
    end
    lnz = b + num_nonzero
    r = r = b + n - free + y_dot
    pi2_k = 1 - post_prob
    if r == 0
        m = 0
        lambda = 0
    else
        m = rand(NegativeBinomial(r, pi2_k)) + lnz
        if (core==true)
            length_lv = length(y) + b
            active_zero = (length(y) + 1):(length_lv)
            lv = zeros(length_lv)
        else
            active_zero = sample(zero_y, b, replace=false)
        end
        gamma_const_prior = 1/(p*c)
        gamma_const = 1/(p*c + constant*(1 - anneal))
        if anneal == 0
            gamma_const_prior = gamma_const
        end
        @views for v in active_zero
            lv[v] = rand(Gamma(1, gamma_const_prior))
        end
        @views for v in nonzero_y
           lv[v] = rand(Gamma(1 + y[v]*(1-anneal), gamma_const))#*(1 - anneal/2) + anneal/2)))
        end
        lambda = sum(lv)
    end
    active_total = lnz
    p = update_p(alpha0, beta0, n, m)
    return(c, p, q, lv, active_total, lambda)
end



function sample_data(factor_C, factor_D, factor_K, core)
    C = size(factor_C, 2)
    D = size(factor_D, 2)
    K = size(factor_K, 2)
    V = size(factor_C, 1)
    A = size(factor_K, 1)
    data = zeros(V, V, A, C, D, K)
    @views for i in 1:V
        @views for j in 1:V
            @views for a in 1:A
                    @views for c in 1:C
                        @views for d in 1:D
                            @views for k in 1:K
                                data[i,j,a, c, d, k] = rand(Poisson(factor_C[i,c]*factor_D[j,d]*factor_K[a,k]*core[c, d, k]))
                            end
                        end
                    end
            end
        end
    end
    return(data)
end


function update_p(alpha, beta, n,m)
    return rand(Beta(alpha + n, beta + m))
end



function update_factor(factor_matrices, y_factor, p_factor, lambdas_Q, indices_QM, constant_factor, m, gamma)
    factor = factor_matrices[m]
    K = size(factor, 2)
    V = size(factor, 1)
    @views @threads for k in 1:K
            s_current = dropdims(sum(factor, dims=1),dims=1)
            s_M = Array{Any}(undef, M)
            @views for m_prime in 1:M
                s_M[m_prime] = dropdims(sum(factor_matrices[m_prime], dims=1),dims=1)
            end
            nz = findall(indices_QM[:, m] .== k)
            core_sum = 0
                            @views for q in nz
                                ind = indices_QM[q,:]
                                lambda = lambdas_Q[q]
                                @views for m_prime in 1:M
                                    if m_prime != m
                                    lambda *= s_M[m_prime][ind[m_prime]]
                                end
                            end
                                core_sum += lambda
                        end
                
        if (core_sum > 0) 
        c, p, q, lv, phi, lambda = update_latent(constant_factor[k], core_sum, y_factor[:, k], p_factor[k], alpha, beta, 0, 0)
        factor[:,k] .= lv
        p_factor[k] = p
        #constant_factor[k] = c
    end
end
    return(factor, p_factor, constant_factor)
end

function update_core(y_core_adj, indices_QM, p_core, constant_core, scalar, y_Q, y_MQ, l_indices, i)
    Q = length(y_Q)
    n = size(l_indices, 1)
    M = size(indices_QM, 2)
    c, p, q, lv, phi, lambda = update_latent(constant_core, scalar, y_core_adj, p_core, 1, 1, n, i, true)
    lambdas_Q = lv
    nzl = findall(!iszero, lambdas_Q)
    nzl_original = findall(!iszero, lambdas_Q[1:Q])
    lambdas_Q = lambdas_Q[nzl]
    indices_QM = indices_QM[nzl_original, :]
    y_Q = y_Q[nzl_original]
    @views for m in 1:M
        y_MQ[m] = y_MQ[m][:, nzl_original]
    end
    num_new_indices = length(nzl) - length(nzl_original)
    if num_new_indices > 0
        while length(lambdas_Q) > size(indices_QM, 1)
            indices_QM = vcat(indices_QM, zeros(Int, 1, size(indices_QM, 2)))
            y_Q = vcat(y_Q, 0)
            y_MQ = [hcat(y_MQ[m], zeros(Int, obs_dims[m])) for m in 1:size(indices_QM, 2)]
        end
        Q_p = length(nzl_original)
        possible_new = n - Q_p
        new_indices = sample(1:possible_new, num_new_indices, replace=false)
        for i in 1:num_new_indices
            replacement_row = l_indices[Q_p + new_indices[i], :]
            indices_QM[Q_p + i, :] = replacement_row
        end
    end
    y_indices = indices_QM[findall(!iszero, y_Q),:]
    @assert size(indices_QM, 1) == length(lambdas_Q)
    @assert size(indices_QM, 1) == length(y_Q)
    @assert size(indices_QM, 1) == size(y_MQ[1], 2)
    return(lambdas_Q, indices_QM, y_Q, y_MQ, y_indices)
end


function init_allocate(obs_dims, Q) #initialize allocation matrices for allocation step
    M = length(obs_dims)
    count_matrices = Array{Matrix{Float64}}(undef, M)
    for i in 1:M
        matrix = spzeros(obs_dims[i], Q)
        count_matrices[i] = matrix
    end
    return (count_matrices)
end

function allocate(nonzero_indices, nonzero_counts, factor_matrices, lambdas_Q, indices_QM, y_M, epsilon, iter, epsilon_M) #allocate counts to latent classes
    M = length(factor_matrices)
    indices = copy(indices_QM)
    ind_M = Array{Any}(undef, M)
    for m in 1:M
        ind_M[m] = findall(!iszero, dropdims(sum(factor_matrices[m], dims=1), dims=1))
    end
    Q = length(lambdas_Q)
    lambdas_Q = vcat(lambdas_Q, zeros(M+1))
    M = length(obs_dims)
    y_Q = zeros(Q + 1 + M)
    ordered_factor_matrices_M = Array{Matrix{Float64}}(undef, M)
    @views for m in 1:M
        y_M[m] = spzeros(obs_dims[m], Q + 1 +M)
        ordered_factor_matrices_M[m] = ones(obs_dims[m], Q + 1 +M)
        ordered_factor_matrices_M[m][:, 1:Q] .= copy(factor_matrices[m][:, indices_QM[:,m]])
    end

    y_epsilon = zeros(Int, obs_dims[1], obs_dims[2])
    y_epsilon_M = Array{Any}(undef, M)
    for m in 1:M
        y_epsilon_M[m] = zeros(Int, obs_dims[m])
    end
    nonzero_counts = copy(Int.(round.(nonzero_counts)))
    @assert length(nonzero_counts) == length(nonzero_indices)
    locker = Threads.SpinLock()
    nz_inds = Array{Any}(undef, M)
    for m in 1:M
        nz_inds[m] = getindex.(nonzero_indices, m)
    end
    nonzero_indices = hcat(nz_inds...)
    if (iter > -burn_in) 
    @views @threads for j in axes(nonzero_indices, 1)
        @views counts = sample_count(nonzero_indices[j,:], copy(lambdas_Q), ordered_factor_matrices_M, nonzero_counts[j], epsilon[nonzero_indices[j,1], nonzero_indices[j,2]], epsilon_M)
        lock(locker)
        @views for m in 1:M
         y_epsilon_M[m][nonzero_indices[j,m]] += counts[end - m]
          y_M[m][nonzero_indices[j,m], :].+= counts
        end 
        y_epsilon[nonzero_indices[j,2], nonzero_indices[j,3]] += counts[end]
        unlock(locker)
    end
else
    probs = ones(Q + 1 + M)/(Q + 1 + M)
    @views @threads for j in axes(nonzero_indices, 1)
        @views counts = rand(Multinomial(nonzero_counts[j], probs))#sample_count(nonzero_indices[j,:], copy(lambdas_Q), ordered_factor_matrices_M, nonzero_counts[j])
        lock(locker)
        @views for m in 1:M
          @views y_M[m][nonzero_indices[j,m], :].+= counts
          @views y_epsilon_M[m][nonzero_indices[j,m]] += counts[end - m]
        end 
        unlock(locker)
    end
end
    y_Q = dropdims(sum(y_M[1], dims=1), dims=1)
    @assert round(sum(y_Q)) == round(sum(nonzero_counts))
    y_Q = y_Q[1:(end-1 - M)]
    lambdas_Q = lambdas_Q[1:(end-1 - M)]
    y_indices = indices_QM[findall(!iszero, y_Q),:]
    @assert length(lambdas_Q) == size(indices_QM, 1)
    return (y_M, y_Q, indices_QM, lambdas_Q, y_epsilon, y_indices, y_epsilon_M)
end

function sample_count(nz_ind, probs, om, nzc, eps, epsilon_M)
    probs[end] = eps
    @views for m in eachindex(om)
                probs[end - m] = epsilon_M[m][nz_ind[m]]
                theta = om[m]
                ind = nz_ind[m]
                @views(probs .*= theta[ind, :])
            end
                @views(normalize!(probs, 1))#./=sum(probs))
        return(rand(Multinomial(nzc, probs)))
end

function gen_mask(MCAR, p, obs_dims, mask_diag, missing_vals=false) #generate mask for heldout data
    M = length(obs_dims)
    if MCAR == true
        heldout = rand(Binomial(1, p), obs_dims)
    else
        partial_heldout = rand(Binomial(1, p), obs_dims[1:(M-1)]) #holdout p proportion of data: 1 = heldout
        heldout = repeat(partial_heldout, outer=tuple(ones(Int, M-1)..., obs_dims[M]))
        @assert size(heldout)==obs_dims
    end

    mask = copy(heldout)
    if mask_diag == true
        for i in 1:(obs_dims[1])
            inds = ntuple(d -> d <= 2 ? i : 1:obs_dims[d], M) 
            if (M > 2)
            heldout[inds...] .= 0 #make sure that the data for heldout likelihood does not include country interactions with itself: 1: in held out evaluation set
            mask[inds...] .= 1 #1: masked 
            else
            heldout[inds...] = 0 #make sure that the data for heldout likelihood does not include country interactions with itself: 1: in held out evaluation set
            mask[inds...] = 1 #1: masked 
            end
        end
    end
    if missing_vals == true
        inds = findall(==(-1), Y)
        mask[inds].=1
        heldout[inds].=0
    end
    return(heldout, mask)
end

function impute(Y, factor_matrices, test_indices, lambdas_Q, indices_QM, true_counts, barY, epsilon, gamma, init=false)   #impute heldout data
    Q = length(lambdas_Q)
    M = length(factor_matrices)
    rates = zeros(length(test_indices))
    imputed = zeros(length(test_indices))
    likelihood = zeros(length(test_indices))
    bar_y = barY
    locker = SpinLock()
    if init == false
        @views @threads for i in eachindex(test_indices)
            temp = ones(Q)
            inds = test_indices[i]
            @views for j in 1:M
                theta = factor_matrices[j]
                temp .*= theta[inds[j], indices_QM[:,j]]
            end
            rate = (temp ⋅lambdas_Q)*gamma + epsilon[inds[1], inds[2]] #+ epsilon_I[inds[1]] + epsilon_J[inds[2]] + epsilon_A[inds[3]] + epsilon_T[inds[4]]
            imputed[i] = rand(Poisson(rate), 1)[1]
            rates[i] = rate
            likelihood[i] = pdf(Poisson(rate), true_counts[i])
        end 
    elseif init==true
        @views @threads for i in eachindex(test_indices)
            inds = test_indices[i]
            mean = sum(Y[inds...])
            imputed[i] = rand(Poisson(bar_y), 1)[1]
            rates[i] = bar_y
            likelihood[i] = pdf(Poisson(bar_y), true_counts[i])
        end
    end  
    return(imputed, likelihood, rates)
end


a0 = 1
b0 = 1
a0_core = 1
b0_core = 1

C = parse(Int, ARGS[2])#25
D = 3
K = 3
R = 3
Q = maximum([C, D, K, R])

beta_core = 1
alpha_core = 1


#data = load("/Users/johnhood/Research/Schein/Dynamic Allocore/icews_full.jld")["tensor"]
#data = load("terrier.jld")["tensor"]
data = load("AllocaDA/FARMM/Y.jld")["Y"]
#data = load("/net/projects/schein-lab/hood/allocore/TERRIER/data.jld")["tensor"]
#data = load("/Users/johnhood/uber_data.jld")["tensor"]
Y = data[:,:,2:16]
#Y = load("DIABIMMUNE/Y.jld")["Y"]
#data = reshape(data, (size(data)..., 1))
#data = data["tensor"]#[:, :, :, 1:12]
#last mode of tensor is binned by month, I want to bin by year
#Y = dropdims(sum(data, dims=3), dims=3)
println(sum(Y))
#Y = zeros(size(data,1), size(data,2), size(data,3), 19)
#for i in 1:19
  #  Y[:,:,:,i] = dropdims(sum(data[:,:,:,((i-1)*12 + 1):(i*12)], dims=4), dims=4)
#end
println(mean(Y.==0))
println(mean(Y.==-1))




obs_dims = size(Y)
V = obs_dims[1]
#A = obs_dims[2]
#T = obs_dims[3]
A = obs_dims[3]
if (length(obs_dims) == 4)
    latent_dims = [C, D, K, R]
else
    latent_dims = [C, D, K]
end


gamma = 1#0.2#mean(Y)
epsilon = ones(obs_dims[1], obs_dims[2]).*0.0
epsilon_I = ones(obs_dims[1]).*0.0#01
epsilon_J = ones(obs_dims[2]).*0.0#01
epsilon_A = ones(obs_dims[3]).*0.0#01
epsilon_T = ones(obs_dims[1]).*0.0
M = length(obs_dims)
epsilon_M = Array{Any}(undef, M)
for m in 1:M
    epsilon_M[m] = ones(obs_dims[m]).*0.0#01
end
if (length(obs_dims) > 3)
    T = obs_dims[4]
    epsilon_T = ones(obs_dims[4]).*0.0#01
end

num = parse(Int, ARGS[3])
heldouts = [0, 0.1, 0.21, 0.32, 0.43]
heldout_proportion = heldouts[num + 1]

heldout, mask = gen_mask(true, heldout_proportion, obs_dims, false, true)
println(mean(mask.==1))
Y_train = Y.*abs.(mask.-1)
Y_test = Y.*heldout
test_indices = findall(!iszero, heldout)
diag_indices = findall(!iszero, mask)
diag_indices = setdiff(diag_indices, test_indices)
nonzero_test = findall(!iszero, Y_test)
nonzero_test_counts = Y_test[nonzero_test]
nonzero_train = findall(!iszero, Y_train)
nonzero_train_counts = Y_train[nonzero_train]

#imputed_counts = rand(Poisson(mean(Y)), length(test_indices))
#nonzero_imputed = findall(!iszero, imputed_counts)
#imputed_nonzero_counts = imputed_counts[nonzero_imputed]
true_counts = Y_test[test_indices]
diag_counts = Y[diag_indices]
imputed_diag_counts = rand(Poisson(mean(Y)), length(diag_indices))
nonzero_indices = findall(!iszero, Y_train)
nonzero_counts = Y_train[nonzero_indices]

#nonzero_indices = findall(!iszero, Y)
#nonzero_counts = Y[nonzero_indices]


p_core = 0.99#init_p(1, alpha_core, beta_core)
alpha = 1
beta = 1

M = length(size(Y))
num_latent_factors = 2
p_l = 0.5 #very sparse latent cp factorization, geometric prior
constant_core = 1#/p_core
constant_f = [10 .*ones(latent_dims[m]) for m in 1:M]
p_m = [init_p(latent_dims[m], alpha, beta) for m in 1:M]
lambdas_Q, indices_QM = init_core(latent_dims, Q, true)
m_Q = ones(length(lambdas_Q))
factor_matrices_M = Array{Matrix{Float64}}(undef, M)
for m in 1:M
    factor_matrices_M[m] = init_factor(obs_dims[m], latent_dims[m], constant_f[m], p_m[m], true)
end
fm_l = Array{Matrix{Float64}}(undef, M)
for m in 1:M
    fm_l[m] = zeros(latent_dims[m], num_latent_factors)
end

l_indices = Int.(indices_QM)
l_dims = latent_dims
l_counts = ones(Int, size(l_indices, 1))
burn_in = 500
l_indices, l_counts, fm_l = update_l_indices(l_indices, l_counts, fm_l, l_dims, Int.(indices_QM), p_core, p_l, -burn_in, true)
y_M = init_allocate(obs_dims, Q)


#println("allocating")
b_llks = zeros(length(test_indices))
b_rates = copy(b_llks)
n_iter = 1000
test = true


for i in -burn_in:n_iter
    start = time()
    if (mod(i, 5)==0)
    nzi = copy(nonzero_indices)
    nzc = copy(nonzero_counts)
    y_m = copy(y_M)
        y_m, y_q, indices_qm, lq, y_epsilon, y_indices, y_epsilon_M = allocate(nzi, nzc, factor_matrices_M, lambdas_Q, indices_QM, y_m, epsilon, i, epsilon_M)
        global y_MQ, y_Q, indices_QM, lambdas_Q = y_m, y_q, indices_qm, lq
        global y_epsilon = y_epsilon
        if (i > -burn_in)
            #global epsilon = update_epsilon(y_epsilon)
            for m in 1:M
                #global epsilon_M[m] = update_epsilon(y_epsilon_M[m])
            end
            end
    end 

        #println("maximum core entry: $(maximum(lambdas_Q))")
        #println("number of nonzero core entries: $(length(lambdas_Q))")
        #println("number of nonzero sender communities: $(length(unique(indices_QM[:,1])))")
        #println("number of nonzero receiver communities: $(length(unique(indices_QM[:,2])))")
        #println("number of nonzero action topics: $(length(unique(indices_QM[:,3])))")
        #println("number of nonzero regimes: $(length(unique(indices_QM[:,4])))")
        if length(lambdas_Q) > Q
            #println("max new lambda: $(maximum(lambdas_Q[(Q+1):end]))")
            #println("max new y: $(maximum(y_Q[(Q+1):end]))")
            #println("min old y: $(minimum(y_Q[1:Q]))")
        end
        #println("proportion of C factor at 0: $(mean(factor_C .== 0))")
        #println("proportion of D factor at 0: $(mean(factor_D .== 0))")
        #println("proportion of K factor at 0: $(mean(factor_K .== 0))")
        #println("proportion of R factor at 0: $(mean(factor_R .== 0))")
        #println("number of core elements with allocated counts: $(size(y_indices, 1))")
        y_core_adj, scalar, y_indices, y_aux = adjust_Ycore(y_Q, indices_QM, lambdas_Q, factor_matrices_M, m_Q, gamma, constant_core)
        global l_indices, l_counts, fm_l = update_l_indices(l_indices, l_counts, fm_l, l_dims, y_indices, p_core, p_l, i, false)
        #println("maximum latent count: $(maximum(l_counts))")
        @assert length(y_Q) == size(indices_QM, 1)
        global lambdas_Q, indices_QM, y_Q, y_MQ, y_indices = update_core(y_core_adj, indices_QM, p_core, constant_core, scalar, y_Q, y_MQ, l_indices, i)
        Q_loc = length(lambdas_Q)
        @views for m in 1:M
            matrix = zeros(obs_dims[m], latent_dims[m])
            @views for q in 1:Q_loc
                matrix[:, indices_QM[q,m]] .+= y_MQ[m][:,q]
            end
            global y_M[m] = matrix
        end 
        if (i > -burn_in)
            for m in 1:M 
                global factor_matrices_M[m], p_m[m], constant_f[m] = update_factor(factor_matrices_M, y_M[m], p_m[m], lambdas_Q, indices_QM, constant_f[m], m, 1)
            end
        else
        end
    if (mod(i, 20) == 0) && test == true && i > -burn_in + 100
        #println("number of core elements with allocated counts: $(size(y_indices, 1))")
        global test_counts, likelihood, i_rate = impute(Y, factor_matrices_M, test_indices, lambdas_Q, indices_QM, true_counts, 0, epsilon, 1, false)
        if (i > 0)
        global b_rates .+= i_rate
        mae = mean(abs.(b_rates./((i)/20) .- true_counts))
        #println(mae)
        global b_llks .+= likelihood
        avg_likelihood = b_llks./((i)/20)
        avg_likelihood[avg_likelihood.==0] .= 1e-5
        avg_likelihood[isnan.(avg_likelihood)].=1e-5
        heldout_ppd = exp(mean(log.(avg_likelihood)))
        nonzero_heldout_ppd = exp(mean(log.(avg_likelihood)[findall(!iszero, true_counts)]))
        #println("heldout PPD: $heldout_ppd")
        #println("nonzero heldout PPD: $nonzero_heldout_ppd")
        end
        global diag_counts, _, _ = impute(Y, factor_matrices_M, diag_indices, lambdas_Q, indices_QM, diag_counts, 0, epsilon, 1, false)
        global nonzero_test_indices = findall(!iszero, test_counts)
        global nonzero_test_counts = test_counts[nonzero_test_indices]
        global nonzero_test_indices = test_indices[nonzero_test_indices]
        global nonzero_diag_indices = findall(!iszero, diag_counts)
        global nonzero_diag_counts = diag_counts[nonzero_diag_indices]
        global nonzero_diag_indices = diag_indices[nonzero_diag_indices]
        global nonzero_indices = vcat(nonzero_train, nonzero_test_indices, nonzero_diag_indices)
        global nonzero_counts = vcat(nonzero_train_counts, nonzero_test_counts, nonzero_diag_counts)
        end_time = time()
        println("iteration $i took $(end_time - start) seconds")
        println("")
        if (i > 0)
        #save("sample_$i.jld", "lambdas_Q", lambdas_Q, "indices_QM", indices_QM, "factor_matrices_M", factor_matrices_M, "y_M", y_M, "y_Q", y_Q, "p_C", p_C, "p_D", p_D, "p_K", p_K, "p_R", p_R, "b_llks", b_llks)
        end
    end

    end
    #if (heldout_proportion == 0)
        writedlm("AllocaDA/FARMM/A$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", factor_matrices_M[2], ",")
        if (num == 0)
        writedlm("AllocaDA/FARMM/T$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", factor_matrices_M[3], ",")
        writedlm("AllocaDA/FARMM/G$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", factor_matrices_M[1], ",")
        writedlm("AllocaDA/FARMM/core$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", lambdas_Q, ",")
        writedlm("AllocaDA/FARMM/indices$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", indices_QM, ",")
        end
        #writedlm("FARMM/l1.csv", fm_l[1], ",")
        #writedlm("FARMM/l2.csv", fm_l[2], ",")
        #writedlm("FARMM/l3.csv", fm_l[3], ",")
    #end
    #save("priorseed$(seed)sample_4000$(heldout_proportion).jld", "lambdas_Q", lambdas_Q, "indices_QM", indices_QM, "factor_matrices_M", factor_matrices_M, "y_M", y_M, "y_Q", y_Q)