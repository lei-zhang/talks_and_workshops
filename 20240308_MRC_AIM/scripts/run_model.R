# =============================================================================
#### Info #### 
# =============================================================================
# simple Q-Learning model
# single true parameters, true lr = 0.6, tau = 1.5, pRew = 0.7
#
# Lei Zhang
# l.zhang.13@bham.ac.uk

# =============================================================================
#### Construct Data #### 
# =============================================================================
library(rstan)
library(ggplot2)

load('data/rl_sp_ss.RData')
sz = dim(rl_ss)
nTrials = sz[1]

dataList = list(nTrials=nTrials, 
                 choice=rl_ss[,1], 
                 reward=rl_ss[,2])

# =============================================================================
#### Running Stan #### 
# =============================================================================
rstan_options(auto_write = TRUE)
options(mc.cores = 4)

modelFile = 'scripts/q_learning_single_subj.stan'

nIter     = 2000
nChains   = 4 
nWarmup   = floor(nIter/2)
nThin     = 1

cat("Estimating", modelFile, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_rw = stan(modelFile, 
               data    = dataList, 
               chains  = nChains,
               iter    = nIter,
               warmup  = nWarmup,
               thin    = nThin,
               init    = "random")

cat("Finishing", modelFile, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")

saveRDS('outputs/stanfit.RData', fit_rw)

# =============================================================================
#### Model Summary and Diagnostics #### 
# =============================================================================
print(fit_rw)

plot_trace_excl_warm_up = stan_trace(fit_rw, pars = c('lr','tau'), inc_warmup = F)
ggsave(plot = plot_trace_excl_warm_up, "_plots/lr_ss_trace.png", width = 6, height = 4, type = "cairo-png", units = "in")

plot_dens = stan_plot(fit_rw, pars=c('lr','tau'), show_density=T, fill_color = 'skyblue')
ggsave(plot = plot_dens, "_plots/lr_ss_dens.png", width = 6, height = 4, type = "cairo-png", units = "in")

## stan_plot(fit_reg, pars = 'p', show_density = T)
return(fit_rw)

