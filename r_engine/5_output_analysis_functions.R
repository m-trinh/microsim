
# """
# File: output_analysis_functions
#
# Functions to analyze simulation output 
# 
# Luke
# 
# """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Table of Contents
#~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. replicate_weights_SE
# standard_summary_stats


# ============================ #
# 1.replicate_weights_SE
# ============================ #

# function to generate SE for an ACS variable using replicate weights
# following method specified by this document from Census:
# https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2012_2016AccuracyPUMS.pdf
replicate_weights_SE <- function(d, var) {
  # base estimate of population mean, total
  x= weighted.mean(d[,var], d[,'PWGTP'], na.rm=TRUE)
  tot=sum(d[,var]* d[,'PWGTP'], na.rm=TRUE)
  
  # Estimates from replicate weights
  replicate_weights <- paste0('PWGTP',seq(1,80))
  count=0
  for (i in replicate_weights) {
    count=count+1
    assign(paste("x", count, sep = ""), weighted.mean(d[,var], d[,i], na.rm=TRUE))   
    assign(paste("tot", count, sep = ""), sum(d[,var]* d[,i], na.rm=TRUE))
  }
  replicate_means <- paste0('x',seq(1,80))
  
  # calculate standard error, confidence interval
  SE= sqrt(4/80*sum(sapply(mget(paste0('x',seq(1,80))), function(y) {(y-x)^2})))
  CI_low=x-1.96*SE
  CI_high=x+1.96*SE
  CI= paste("[",format(x-1.96*SE, digits=2, scientific=FALSE, big.mark=","),",", format(x+1.96*SE, digits=2, scientific=FALSE, big.mark=","),"]")
  total=sum(d[,var]*d[,'PWGTP'], na.rm=TRUE)
  total_SE= sqrt(4/80*sum(sapply(mget(paste0('tot',seq(1,80))), function(y) {(y-tot)^2})))
  total_CI_low= total-total_SE*1.96
  total_CI_high= total+total_SE*1.96
  total_CI=paste("[",format(total_CI_low, digits=2, scientific=FALSE, big.mark=","),",", format(total_CI_high, digits=2, scientific=FALSE, big.mark=","),"]")

  # return statistics
  stats= list(var, estimate=x, std_error=SE,confidence_int=CI,CI_low=CI_low,CI_high=CI_high, 
              total=total, total_SE=total_SE,total_CI_low=total_CI_low,total_CI_high=total_CI_high, total_CI=total_CI)
  for (i in stats[c(2:3,7:8,11)]) {
    i <- format(i, nsmall=3)
  }
  return(stats)
}

# ============================ #
# 2. standard_summary_stats
# ============================ #
# function to produce some standard summary statistics of relevant leave taking and other vars in csv format
standard_summary_stats <-function(d, output) {
  
  ptake_vars=c()
  ptake_names=c()
  for (i in leave_types) {
    ptake_vars=c(ptake_vars,paste("ptake_",i,sep=""))
    ptake_names=c(ptake_names, paste("Took & got benefits for",i,'leave'))
  }
  
  # define columns of csv
  vars=c('eligworker', 'particip', 'particip_length',ptake_vars, 'actual_benefits')
  mean=c()
  SE=c()
  CI=c()
  total=c()
  total_SE=c()
  total_CI=c()
  for (i in vars) {
    temp=replicate_weights_SE(d, i)
    mean=c(mean, temp[2])
    SE=c(SE, temp[3])
    CI=c(CI, temp[4])
    total=c(total, temp[7])
    total_SE=c(total_SE, temp[8])
    total_CI=c(total_CI, temp[11])
  }
  
  mean=unname(unlist(mean))
  SE=unname(unlist(SE))
  CI=unname(unlist(CI))
  total=unname(unlist(total))
  total_SE=unname(unlist(total_SE))
  total_CI=unname(unlist(total_CI))
  
  var_names=c('Eligible for leave program', 'Participated in leave program', 'Length of Participation in Days', ptake_names,'Amount of Benefits Received ($)')
  d_out=data.frame(var_names,mean,SE,CI,total, total_SE, total_CI)
  colnames(d_out) <- c("Variable","Mean", "Standard Error of Mean", "Confidence Interval","Population Total", "Pop Total Standard Error", "Pop Total CI")
  write.csv(d_out,file=paste0('./output/',output,"_stats.csv"), row.names= FALSE)
}

# ============================ #
# 2. state_compar_stats
# ============================ #

# function to produce summary statistics to compare with actual state of relevant leave taking and other vars in csv format
state_compar_stats <-function(d, output) {
  
  ptake_vars=c()
  ptake_names=c()
  for (i in leave_types) {
    ptake_vars=c(ptake_vars,paste("ptake_",i,sep=""))
    ptake_names=c(ptake_names, paste("Participated for",i,'leave'))
  }
  
  # define columns of csv
  vars=c(ptake_vars, 'ptake_DI','ptake_PFL', 'particip','bene_DI','bene_PFL', 'actual_benefits')
  mean=c()
  SE=c()
  CI=c()
  total=c()
  total_SE=c()
  total_CI=c()
  for (i in vars) {
    temp=replicate_weights_SE(d, i)
    mean=c(mean, temp[2])
    SE=c(SE, temp[3])
    CI=c(CI, temp[4])
    total=c(total, temp[7])
    total_SE=c(total_SE, temp[8])
    total_CI=c(total_CI, temp[11])
  }
  
  mean=unname(unlist(mean))
  SE=unname(unlist(SE))
  CI=unname(unlist(CI))
  total=unname(unlist(total))
  total_SE=unname(unlist(total_SE))
  total_CI=unname(unlist(total_CI))
  
  var_names=c(ptake_names, 'Participated for  for own illness or maternal disability leave', 'Participated for ill relative or child bonding leave', 'Participated for any reason',
              'Benefits Received ($), for own illness or maternal disability leave','Benefits Received ($), for for ill relative or child bonding leave',
              'Benefits Received ($), total')
  d_out=data.frame(var_names,mean,SE,CI,total, total_SE, total_CI)
  colnames(d_out) <- c("Variable","Mean", "Standard Error of Mean", "Confidence Interval","Population Total", "Pop Total Standard Error", "Pop Total CI")
  write.csv(d_out,file=paste0('./output/',output,"_rawstats.csv"), row.names= FALSE)
  
  
  # create rounded results with cleaned up names
  round_mean=format(mean, digits=2, scientific=FALSE, big.mark=",")
  round_SE=format(SE,  digits=2, scientific=FALSE, big.mark=",")
  round_CI=CI
  round_total=format(total, digits=2, scientific=FALSE, big.mark=",")
  round_total_SE=format(total_SE,  digits=2, scientific=FALSE, big.mark=",")
  round_total_CI=total_CI
  
  d_out=data.frame(var_names,round_mean,round_SE,round_CI,round_total, round_total_SE, round_total_CI)
  colnames(d_out) <- c("Variable","Mean", "Standard Error of Mean", "Confidence Interval","Population Total", "Pop Total Standard Error", "Pop Total CI")
  write.csv(d_out,file=paste0('./output/',output,"_roundstats.csv"), row.names= FALSE)
  
  if (makelog==TRUE) {
    
    # add results to log file
    cat("------------------------------", file = log_name, sep="\n", append = TRUE)
    cat("Results", file = log_name, sep="\n", append = TRUE)
    cat("------------------------------", file = log_name, sep="\n", append = TRUE)
    cat(pandoc.table.return(d_out[,c(1,5:7)], split.tables = Inf, justify = 'left'), 
        file = log_name, sep="\n", append = TRUE)
    
    cat("------------------------------", file = log_name, sep="\n", append = TRUE)
    cat("Runtime", file = log_name, sep="\n", append = TRUE)
    cat("------------------------------", file = log_name, sep="\n", append = TRUE)
    cat(Sys.time() - timestart, file = log_name, sep="\n", append = TRUE)
    print(Sys.time() - timestart)
  }
}

# ============================ #
# 3. length_compar
# ============================ #
# function to compare status quo and leave taking variables 
take_compar <- function(d, output) {
  length_vars=paste0("length_",leave_types)
  length_names=paste("Counterfactual",leave_types,'leave')
  squo_vars = paste0("squo_", length_vars)
  squo_names=paste("Status Quo",leave_types,'leave')
  vars=c(length_vars, squo_vars)
  mean=c()
  SE=c()
  CI=c()
  total=c()
  total_SE=c()
  total_CI=c()
  for (i in vars) {
    temp=replicate_weights_SE(d %>% filter(get(i)>0), i)
    mean=c(mean, temp[2])
    SE=c(SE, temp[3])
    CI=c(CI, temp[4])
    total=c(total, temp[7])
    total_SE=c(total_SE, temp[8])
    total_CI=c(total_CI, temp[11])
  }
  
  mean=unname(unlist(mean))
  SE=unname(unlist(SE))
  CI=unname(unlist(CI))
  total=unname(unlist(total))
  total_SE=unname(unlist(total_SE))
  total_CI=unname(unlist(total_CI))
  
  var_names=c(length_names, squo_names)
  d_out=data.frame(var_names,mean,SE,CI,total, total_SE, total_CI)
  colnames(d_out) <- c("Variable","Mean", "Standard Error of Mean", "Confidence Interval","Population Total", "Pop Total Standard Error", "Pop Total CI")
  write.csv(d_out,file=paste0('./output/',output,"_takestats.csv"), row.names= FALSE)
  
  # graph leave length distributions 
  for (i in length_vars) {
    png(paste0('./output/',i , '_cfact.png'))
    if (nrow(d %>% filter(get(i)>0))> 0 ){
      hist(d %>% filter(get(i)>0) %>% pull(get(i)), main = paste('Counterfactual leave distribution', i), xlab = "Length in Days", ylab = "Frequency", breaks=20)  
    }
    dev.off()
  }
  for (i in squo_vars) {
    png(paste0('./output/',i , '_squo.png'))
    if (nrow(d %>% filter(get(i)>0))> 0 ){
      hist(d %>% filter(get(i)>0) %>% pull(get(i)), main = paste('Status quo leave distribution', i), xlab = "Length in Days", ylab = "Frequency", breaks=20)  
    }
    dev.off()
  }
  
  # combine graphs into a single pic file 
  cfact_img <- image_read(paste0('./output/',length_vars[1] , '_cfact.png'))
  for (i in length_vars) {
    img <- image_read(paste0('./output/',i , '_cfact.png'))
    cfact_img <- image_append(c(cfact_img, img), stack = TRUE)
    file.remove(paste0('./output/',i , '_cfact.png'))
  }
  image_write(cfact_img, path = paste0("./output/",output,"_cfact_lengths.png"), format = "png")

  squo_img <- image_read(paste0('./output/',squo_vars[1] , '_squo.png'))  
  for (i in squo_vars) {
    img <- image_read(paste0('./output/',i , '_squo.png'))
    squo_img <- image_append(c(squo_img, img), stack = TRUE)
    file.remove(paste0('./output/',i , '_squo.png'))
  }
  image_write(squo_img, path = paste0("./output/",output,"_squo_lengths.png"), format = "png")
}