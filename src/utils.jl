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

function allocate(nonzero_indices, nonzero_counts, factor_matrices, lambdas_Q, indices_QM, y_M, iter) #allocate counts to latent classes
    M = length(factor_matrices)
    ind_M = Array{Any}(undef, M)
    for m in 1:M
        ind_M[m] = findall(!iszero, dropdims(sum(factor_matrices[m], dims=1), dims=1))
    end
    Q = length(lambdas_Q)
    lambdas_Q = vcat(lambdas_Q)
    M = length(obs_dims)
    y_Q = zeros(Q)
    ordered_factor_matrices_M = Array{Matrix{Float64}}(undef, M)
    @views for m in 1:M
        y_M[m] = spzeros(obs_dims[m], Q)
        ordered_factor_matrices_M[m] = ones(obs_dims[m], Q)
        ordered_factor_matrices_M[m] .= copy(factor_matrices[m][:, indices_QM[:,m]])
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
        @views counts = sample_count(nonzero_indices[j,:], copy(lambdas_Q), ordered_factor_matrices_M, nonzero_counts[j])
        lock(locker)
        @views for m in 1:M
          y_M[m][nonzero_indices[j,m], :].+= counts
        end 
        unlock(locker)
    end
else
    probs = ones(Q)/Q
    @views @threads for j in axes(nonzero_indices, 1)
        @views counts = rand(Multinomial(nonzero_counts[j], probs))#sample_count(nonzero_indices[j,:], copy(lambdas_Q), ordered_factor_matrices_M, nonzero_counts[j])
        lock(locker)
        @views for m in 1:M
          @views y_M[m][nonzero_indices[j,m], :].+= counts
        end 
        unlock(locker)
    end
end
    y_Q = dropdims(sum(y_M[1], dims=1), dims=1)
    @assert round(sum(y_Q)) == round(sum(nonzero_counts))
    y_indices = indices_QM[findall(!iszero, y_Q),:]
    @assert length(lambdas_Q) == size(indices_QM, 1)
    return (y_M, y_Q, indices_QM, lambdas_Q, y_indices)
end

function sample_count(nz_ind, probs, om, nzc)
    @views for m in eachindex(om)
                theta = om[m]
                ind = nz_ind[m]
                @views(probs .*= theta[ind, :])
            end
            if sum(probs) == 0
                probs = ones(length(probs))/length(probs)
            else
                @views(normalize!(probs, 1))#./=sum(probs))
            end
        return(rand(Multinomial(nzc, probs)))
end

function gen_mask(MCAR, p, obs_dims, mask_diag, missing_vals=false) #generate mask for heldout data
    M = length(obs_dims)
    if MCAR == true
        heldout = rand(Binomial(1, p), obs_dims)
    else
        partial_heldout = rand(Binomial(1, p), obs_dims[2:M]) #holdout p proportion of data: 1 = heldout
        println(size(partial_heldout))
        partial_heldout = reshape(partial_heldout, (1, 30, 15))
        heldout = repeat(partial_heldout, outer=tuple(obs_dims[1], ones(Int, M-1)...))
        print(size(heldout))
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