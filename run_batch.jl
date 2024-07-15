seeds = 380:389
#a0s = [0.01]#, 0.1, 1, 10]
#b0s = [0.01]#, 0.1, 1, 10]
#alpha0s = [1000]#[0.01, 0.1, 1, 10]
#beta0s = [1]
#theta_rates = [0.1, 1.0]
Hs = 1:4

for seed in seeds
    for H in Hs
                cmd = `./jcp.sh $seed $H`
                run(cmd)
     end
 end


        