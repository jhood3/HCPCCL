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
seed = parse(Int, ARGS[1])
Random.seed!(seed)

a0 = 1
b0 = 1
a0_core = 1
b0_core = 1

C = 15
D = 3
K = 3
R = 3
Q = maximum([C, D, K, R])

beta_core = 1
alpha_core = 1



data = load("data/FARMM/Y.jld")["Y"]
Y = data[:,:,2:16]





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

num = parse(Int, ARGS[2])

 
heldouts = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
heldout_proportion = heldouts[num + 1]

heldout, mask = gen_mask(false, heldout_proportion, obs_dims, false, true)
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

true_counts = Y_test[test_indices]
diag_counts = Y[diag_indices]
imputed_diag_counts = rand(Poisson(mean(Y)), length(diag_indices))
nonzero_indices = findall(!iszero, Y_train)
nonzero_counts = Y_train[nonzero_indices]



p_core = 0.9
alpha = 1
beta = 1

M = length(size(Y))
constant_core = 1
constant_f = [10 .*ones(latent_dims[m]) for m in 1:M]
p_m = [init_p(latent_dims[m], alpha, beta) for m in 1:M]
lambdas_Q, indices_QM = init_core(latent_dims, Q, true)
m_Q = ones(length(lambdas_Q))
factor_matrices_M = Array{Matrix{Float64}}(undef, M)
for m in 1:M
    factor_matrices_M[m] = init_factor(obs_dims[m], latent_dims[m], constant_f[m], p_m[m], true)
end


l_indices = Int.(indices_QM)
burn_in = 500
y_M = init_allocate(obs_dims, Q)



b_llks = zeros(length(test_indices))
b_rates = copy(b_llks)
n_iter = 1000
test = true


for i in -burn_in:n_iter
    start = time()
    if (mod(i, 20) == 0)  && i < 0 && i > -burn_in + 100 #thresholding during burn-in
        global y_Q = y_Q[findall(lambdas_Q .> 0.003)]
        global indices_QM = indices_QM[findall(lambdas_Q .> 0.003), :]
        global lambdas_Q = lambdas_Q[lambdas_Q .> 0.003]
    end
    if (mod(i, 5)==0)
    nzi = copy(nonzero_indices)
    nzc = copy(nonzero_counts)
    y_m = copy(y_M)
        y_m, y_q, indices_qm, lq, y_indices = allocate(nzi, nzc, factor_matrices_M, lambdas_Q, indices_QM, y_m, i)
        global y_MQ, y_Q, indices_QM, lambdas_Q = y_m, y_q, indices_qm, lq
    end 
        y_core_adj, scalar, y_indices, y_aux = adjust_Ycore(y_Q, indices_QM, lambdas_Q, factor_matrices_M, m_Q, gamma, constant_core)
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
        end
    if (mod(i, 20) == 0) && test == true && i > -burn_in + 100
        global test_counts, likelihood, i_rate = impute(Y, factor_matrices_M, test_indices, lambdas_Q, indices_QM, true_counts, 0, epsilon, 1, false)
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
    end
end
    
    #if (heldout_proportion == 0)
        writedlm("results/thresholding/A$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", factor_matrices_M[2], ",")
        if (num == 0)
        writedlm("results/thresholding/T$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", factor_matrices_M[3], ",")
        writedlm("results/thresholding/G$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", factor_matrices_M[1], ",")
        writedlm("results/thresholding/core$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", lambdas_Q, ",")
        writedlm("results/thresholding/indices$(num)_C$(C)_D$(D)_K$(K)_$seed.csv", indices_QM, ",")
        end
    #end
    #save("priorseed$(seed)sample_4000$(heldout_proportion).jld", "lambdas_Q", lambdas_Q, "indices_QM", indices_QM, "factor_matrices_M", factor_matrices_M, "y_M", y_M, "y_Q", y_Q)