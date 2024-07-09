using SparseArrays
using Distributions
using Random
using StatsBase
using Base.Threads
using JLD
using LinearAlgebra
using IterTools
using DelimitedFiles

include("utils.jl")
seed = 123#parse(Int, ARGS[1])
Random.seed!(seed)

a0 = 1
b0 = 1
a0_core = 1
b0_core = 1

C = 12#parse(Int, ARGS[2])
D = 3
K = 3
R = 3
Q = maximum([C, D, K, R])

beta_core = 1
alpha_core = 1



data = load("/Users/johnhood/Research/Schein/Allocation/FARMM/Y.jld")["Y"]
Y = data[:,:,2:16]
#Y = load("DIABIMMUNE/Y.jld")["Y"]
#data = reshape(data, (size(data)..., 1))
#data = data["tensor"]#[:, :, :, 1:12]
#last mode of tensor is binned by month, I want to bin by year
#Y = dropdims(sum(data, dims=3), dims=3)

println(sum(Y))
println(mean(Y.==0))
println(mean(Y.==-1))




obs_dims = size(Y)
V = obs_dims[1]
A = obs_dims[2]
T = obs_dims[3]

latent_dims = [C, D, K]



gamma = 1
epsilon = ones(obs_dims[1], obs_dims[2]).*0.0
epsilon_I = ones(obs_dims[1]).*0.0
epsilon_J = ones(obs_dims[2]).*0.0
epsilon_A = ones(obs_dims[3]).*0.0
epsilon_T = ones(obs_dims[1]).*0.0
M = length(obs_dims)
epsilon_M = Array{Any}(undef, M)
for m in 1:M
    epsilon_M[m] = ones(obs_dims[m]).*0.0
end
if (length(obs_dims) > 3)
    T = obs_dims[4]
    epsilon_T = ones(obs_dims[4]).*0.0
end

num = 1#parse(Int, ARGS[3])
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
        y_m, y_q, indices_qm, lq, _, y_indices, _ = allocate(nzi, nzc, factor_matrices_M, lambdas_Q, indices_QM, y_m, epsilon, i, epsilon_M)
        global y_MQ, y_Q, indices_QM, lambdas_Q = y_m, y_q, indices_qm, lq
    end 

        y_core_adj, scalar, y_indices, y_aux = adjust_Ycore(y_Q, indices_QM, lambdas_Q, factor_matrices_M, m_Q, gamma, constant_core)
        global l_indices, l_counts, fm_l = update_l_indices(l_indices, l_counts, fm_l, l_dims, y_indices, p_core, p_l, i, false)
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
    end

    end
    #if (heldout_proportion == 0)
        #writedlm("AllocaDA/FARMM/A$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", factor_matrices_M[2], ",")
        #if (num == 0)
        #writedlm("AllocaDA/FARMM/T$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", factor_matrices_M[3], ",")
        #writedlm("AllocaDA/FARMM/G$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", factor_matrices_M[1], ",")
        #writedlm("AllocaDA/FARMM/core$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", lambdas_Q, ",")
        #writedlm("AllocaDA/FARMM/indices$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", indices_QM, ",")
        #end
        #writedlm("FARMM/l1.csv", fm_l[1], ",")
        #writedlm("FARMM/l2.csv", fm_l[2], ",")
        #writedlm("FARMM/l3.csv", fm_l[3], ",")
    #end
    #save("priorseed$(seed)sample_4000$(heldout_proportion).jld", "lambdas_Q", lambdas_Q, "indices_QM", indices_QM, "factor_matrices_M", factor_matrices_M, "y_M", y_M, "y_Q", y_Q)