# benchmark_analysis.R
# PURPOSE: code for issue brief benchmarking model results against ACM model and actual data in CA/NJ/RI
# This analyzes the results of the simulations run in benchmark_sim.R and outputs the results to "output/benchmark nums 9_11.csv"

library('stringr')
library('ggplot2')
source("5_ABF_functions.R")
source("6_output_analysis_functions.R")

# load simulated csvs for each state - FOR REPLICATION, THESE WILL NEED TO BE POINTED TO THE OUTPUT FILES CREATED BY SIMULATIONS RAN IN benchmark_sim.R
ri <- read.csv('C:/Users/lpatterson/AnacondaProjects/microsim/output/output_20201118_120358/RI_Logistic Regression GLMissue_brief_1_111020.csv')
ca <- read.csv('C:/Users/lpatterson/AnacondaProjects/microsim/output/output_20201117_174910/CA_Logistic Regression GLMissue_brief_1_111020.csv')
nj <- read.csv('C:/Users/lpatterson/AnacondaProjects/microsim/output/output_20201117_174503/NJ_Logistic Regression GLMissue_brief_1_111020.csv')

# start a data frame to store all the summary statistics we need
states <- c('CA','NJ','RI')
leave_types <- c("own","illspouse","illchild","illparent","matdis","bond")
vars <- append(c(),values=c(paste0('takeup_', leave_types), paste0('cpl_', leave_types), 
                            c('annual_benefit_all','eligworker','DI_plen')))
results <- data.frame(row.names = vars)

# set cpl = 0 if takeup_ = 0
for (i in leave_types) {
  cpl_var <- paste0('cpl_',i)
  takeup_var <- paste0('takeup_',i)
  ri[cpl_var] <- with(ri, ifelse(get(takeup_var)==0, 0, get(cpl_var)))
  ca[cpl_var] <- with(ca, ifelse(get(takeup_var)==0, 0, get(cpl_var)))
  nj[cpl_var] <- with(nj, ifelse(get(takeup_var)==0, 0, get(cpl_var)))
}

#for (i in vars)
results['CA'] <- NA
results['NJ'] <- NA
results['RI'] <- NA
results['CA_SE'] <- NA
results['NJ_SE'] <- NA
results['RI_SE'] <- NA

# calculate values for each var and state

# CA 
d <- ca
for (v in vars) {
  stats <- replicate_weights_SE(d, var=v, place_of_work = TRUE)
  if (str_detect(v,'cpl')==FALSE & str_detect(v,'DI_plen')==FALSE) {
    stats <- replicate_weights_SE(d, var=v, filt=d[v]>0, place_of_work = TRUE)
    results[v,'CA'] <- stats['total']
    results[v,'CA_SE'] <- stats['total_SE']  
  } else {
    stats <- replicate_weights_SE(d, var=v, filt=d[v]>0, place_of_work = TRUE)
    results[v,'CA'] <- stats[['estimate']]/5
    results[v,'CA_SE'] <- stats[['std_error']]/5
  }
}

# NJ
d <- nj
for (v in vars) {
  stats <- replicate_weights_SE(d, var=v, place_of_work = TRUE)
  if (str_detect(v,'cpl')==FALSE & str_detect(v,'DI_plen')==FALSE) {
    stats <- replicate_weights_SE(d, var=v, filt=d[v]>0, place_of_work = TRUE)
    results[v,'NJ'] <- stats['total']
    results[v,'NJ_SE'] <- stats['total_SE']  
  } else {
    stats <- replicate_weights_SE(d, var=v, filt=d[v]>0, place_of_work = TRUE)
    results[v,'NJ'] <- stats[['estimate']]/5
    results[v,'NJ_SE'] <- stats[['std_error']]/5
  }
}

# RI 
d <- ri
for (v in vars) {
  if (str_detect(v,'cpl')==FALSE & str_detect(v,'DI_plen')==FALSE) {
    stats <- replicate_weights_SE(d, var=v, filt=d[v]>0, place_of_work = TRUE)
    results[v,'RI'] <- stats['total']
    results[v,'RI_SE'] <- stats['total_SE']  
  } else {
    stats <- replicate_weights_SE(d, var=v, filt=d[v]>0, place_of_work = TRUE)
    results[v,'RI'] <- stats[['estimate']]/5
    results[v,'RI_SE'] <- stats[['std_error']]/5
  }
}
results['source'] <- 'Worker PLUS'
write.csv(results,file='output/benchmark nums 9_11.csv')


