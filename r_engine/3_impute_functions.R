
# """
# 3_impute_functions
#
# These functions impute the FMLA data set into the ACS.
#
# 9 Sept 2018
# Luke
# 
# TESTING TODO: what happens when filtered test data sets of 0 obs are fed into imputation functions
#               currently is handled properly by runOrdinalImpute and runRandDraw. 
#               Should make sure others are as well.
#
# """



#~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Table of Contents
#~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 1. impute_fmla_to_acs
# Modular imputation methods - can be swaped out for one another for FMLA to ACS imputation of:
# take_* vars, resp_len, prop_pay variables
  # 1A. KNN1_scratch
  # 1B. logit_leave_method
      # 1Ba. runLogitImpute - used in hardcoded methods found elsewhere as well
      # 1Bb. runOrdinalImpute - used in hardcoded methods found elsewhere as well
      # 1Bc. runRandDraw - used in hardcoded methods found elsewhere as well
  # 1C. KNN_multi
  # 1D. Naive_Bayes
  # 1E. ridge_class
  # 1F. random_forest
  # 1G. svm_impute - Gets bad results right now
  # 1H. xG Boost

# ============================ #
# 1. impute_fmla_to_acs
# ============================ #
# master fmla to acs imputation function
# Based on user-specified method, impute leave taking behavior in fmla to ACS
# default is KNN1

impute_fmla_to_acs <- function(d_fmla, d_acs, impute_method,xvars,kval,xvar_wgts) {
  # d_fmla - modified fmla data set
  # d_acs - ACS data set
  # impute_method - method to use for imputation
  # xvars - dependent variables used by imputation method. Must be present and have same name in 
  #         both fmla and acs data sets.
  
  # ---------------------------------------------------------------------------------------------------------
  # A. Leave characteristics needed: leave taking behavior, proportion of income paid by employer,
  #     whether leave was affordable or not
  # ---------------------------------------------------------------------------------------------------------
  
  # -----------------Hard coded objects all methods must use-----------------------------------------------
  # yvars: the dependent vars that must be imputed by the selected method
  yvars <- c(own = "take_own", 
             illspouse = "take_illspouse",
             illchild = "take_illchild",
             illparent = "take_illparent",
             matdis = "take_matdis",
             bond = "take_bond",
             need_own = "need_own", 
             need_illspouse = "need_illspouse",
             need_illchild = "need_illchild",
             need_illparent = "need_illparent",
             need_matdis = "need_matdis",
             need_bond = "need_bond",
              prop_pay = "prop_pay",
              resp_len= "resp_len",
             unaffordable = 'unaffordable')
  
  # filters: logical conditionals always applied to filter vraiable imputation 
  filts <- c(own = "TRUE",
                   illspouse = "nevermarried == 0 & divorced == 0",
                   illchild = "TRUE",
                   illparent = "TRUE",
                   matdis = "female == 1 & nochildren == 0",
                   bond = "nochildren == 0",
                   need_own = "TRUE",
                   need_illspouse = "nevermarried == 0 & divorced == 0",
                   need_illchild = "TRUE",
                   need_illparent = "TRUE",
                   need_matdis = "female == 1 & nochildren == 0",
                   need_bond = "nochildren == 0",
                   prop_pay="TRUE",
                   resp_len="TRUE",
                  unaffordable = 'TRUE')
  
  # weight: if method uses FMLA weights, the weight variable to use
  weights <- c(own = "~ fixed_weight",
              illspouse = "~ fixed_weight",
              illchild = "~ fixed_weight",
              illparent = "~ weight",
              matdis = "~ fixed_weight",
              bond = "~ fixed_weight",
              need_own = "~ fixed_weight",
              need_illspouse = "~ fixed_weight",
              need_illchild = "~ fixed_weight",
              need_illparent = "~ weight",
              need_matdis = "~ fixed_weight",
              need_bond = "~ fixed_weight",
              prop_pay = '~ fixed_weight',
              resp_len = "~ fixed_weight",
              unaffordable = "~ fixed_weight")
  
  # Save ACS and FMLA Dataframes at this point to document format that 
  # alternative imputation methods will need to expect
  saveRDS(d_fmla, file="./R_dataframes/d_fmla_impute_input.rds") # TODO: Remove from final version
  saveRDS(d_acs, file="./R_dataframes/d_acs_impute_input.rds") # TODO: Remove from final version
  
  # KNN1 imputation method
  if (impute_method=="KNN1") {
    # separate KNN1 calls for each unique conditional doesn't work because of differing missing values
    # INPUTS: variable to be imputed, conditionals to filter training and test data on, FMLA data (training), and
    #         ACS data (test), id variable, and dependent variables to use in imputation
    impute <- mapply(KNN1_scratch, imp_var=yvars,train_filt=filts, test_filt=filts,
                        MoreArgs=list(d_train=d_fmla,d_test=d_acs,xvars=xvars, xvar_wgts = xvar_wgts), SIMPLIFY = FALSE)
    # OUTPUTS: list of data sets for each leave taking/other variables requiring imputation. 
   # merge imputed values with acs data
    for (i in impute) {
      # old merge code, caused memory issues. using match instead
      #d_test <- merge(d_filt, d_test, by='id', all.y=TRUE)
      for (j in names(i)) {
        if (j %in% names(d_acs)==FALSE){
          d_acs[j] <- i[match(d_acs$id, i$id), j]    
        }
      }
    }  
    
    # save output for reference when making other methods
    saveRDS(d_acs, file="./R_dataframes/d_acs_impute_output.rds") # TODO: Remove from final version
  }
  
  # Logit estimation of leave taking to compare with Chris' results in Python
  if (impute_method=="logit") {
    d_acs <- logit_leave_method(d_test=d_acs, d_train=d_fmla, xvars=xvars, 
                                yvars=yvars, test_filts=filts, train_filts=filts, 
                                weights=weights, create_dummies=TRUE)
  }
  
  if (impute_method=="KNN_multi") {
    # INPUTS: variable to be imputed, conditionals to filter training and test data on, FMLA data (training), and
    #         ACS data (test), id variable, and dependent variables to use in imputation, number of nbors
    impute <- mapply(KNN_multi, imp_var=yvars,train_filt=filts, test_filt=filts,
                     MoreArgs=list(d_train=d_fmla,d_test=d_acs,xvars=xvars, kval=kval), SIMPLIFY = FALSE)
    
    # OUTPUTS: list of data sets for each leave taking/other variables requiring imputation. 
    # merge imputed values with acs data
    for (i in impute) {
      # old merge code, caused memory issues. using match instead
      #d_acs <- merge(i, d_acs, by="id",all.y=TRUE)
      for (j in names(i)) {
        if (j %in% names(d_acs)==FALSE){
          d_acs[j] <- i[match(d_acs$id, i$id), j]    
        }
      }
    }
    
  }
  if (impute_method=="Naive_Bayes") {
    options(warn=-1)
    d_acs <- Naive_Bayes(d_test=d_acs, d_train=d_fmla, xvars=xvars, 
                         yvars=yvars, test_filts=filts, train_filts=filts, 
                         weights=weights)
    options(warn=0)
  }
  
  if (impute_method=="ridge_class") {
    d_acs <- ridge_class(d_test=d_acs, d_train=d_fmla, xvars=xvars, 
                         yvars=yvars, test_filts=filts, train_filts=filts, 
                         weights=weights)
  }
  
  if (impute_method=="random_forest") {
    d_acs <- random_forest(d_test=d_acs, d_train=d_fmla, xvars=xvars, 
                         yvars=yvars, test_filts=filts, train_filts=filts, 
                         weights=weights)
  }
  
  if (impute_method=="svm") {
    d_acs <- svm_impute(d_test=d_acs, d_train=d_fmla, xvars=xvars, 
                           yvars=yvars, test_filts=filts, train_filts=filts, 
                           weights=weights)
  }
  if (impute_method=="xg_boost") {
    d_acs <- xg_boost_impute(d_test=d_acs, d_train=d_fmla, xvars=xvars, 
                        yvars=yvars, test_filts=filts, train_filts=filts, 
                        weights=weights)
  }
  
  # This is a random leave assignment for comparison purposes.
  # This is assigns leaves randomly based on the population mean for each var
  # Done by running logit with no xvars
  if (impute_method=="random") {
    d_acs <- logit_leave_method(d_test=d_acs, d_train=d_fmla, xvars = "",
                                yvars=yvars, test_filts=filts, train_filts=filts, 
                                weights=weights, create_dummies=TRUE)
  }
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # alternate imputation methods will go here
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # for example:
  
  if (impute_method=="Hocus Pocus") {
    # Hocus pocus function calls here
  }
  
  return(d_acs)
}

# ============================ #
# 1A. KNN1_scratch
# ============================ #
# Define KNN1 matching method

KNN1_scratch <- function(d_train, d_test, imp_var, train_filt, test_filt, xvars, xvar_wgts) { 
  
  # This returns a dataframe of length equal to acs with the employee id and a column for each leave type
  # that indicates whether or not they took the leave.

  # create training data
  
  # filter dataset and keep just the variables of interest
  train <-  d_train %>% filter(complete.cases(select(d_train, 'id', imp_var,xvars))) %>% 
    filter_(train_filt) %>%
    select(imp_var, xvars) %>%
    mutate(id = NULL)
  train ['nbor_id'] <- as.numeric(rownames(train))
  
  # create test data set 
  # This is a dataframe just with the variables in the acs that will be used to compute distance
  
  test <- d_test %>% filter_(test_filt) %>%
    select(id, xvars) %>%
    filter(complete.cases(.))

  # Initial checks
  
  # check for data frames
  if ((!is.data.frame(train)) | (!is.data.frame(test))) {
    stop("train_set and test_set must be data frames")
  }  
  
  # check for missing data
  if (anyNA(train) | anyNA(test)) {
    stop("missing values not allowed in train_test or test_set")
  }
  
  
  # normalize training data to weight differences between variables
  names(xvar_wgts) <- xvars
  for (i in colnames(train)) {
    if (i != 'nbor_id' & i != imp_var & sum(train[i])!=0 ){
      train[i] <- scale(train[i],center=0,scale=max(train[,i])/xvar_wgts[[i]])
    }
  } 
  
  for (i in colnames(test)) {
    if (i != 'id' & sum(test[i])!=0 ){
      test[i] <- scale(test[i],center=0,scale=max(test[,i])/xvar_wgts[[i]])
    }
  } 
  
  # id var must be first variable of data
    # find distance
  
  m_test <- as.matrix(test)
  m_train <-as.matrix(train)
  
  nest_test <- list()
  nest_train <- list()
  # nested lists of vectors for apply functions
  
  nest_test <- lapply(seq(1,nrow(m_test)) , function(y){ 
    m_test[y,colnames(test)!='id']
  })
  nest_train <- lapply(seq(1,nrow(m_train)) , function(y){ 
    m_train[y,colnames(train)!='nbor_id' & colnames(train)!=imp_var]
  })
  
  # mark minimium distance
  min_start <- ncol(train)-2
  
  where_min <- function(j) {
    min_dist <- min_start
    d <- mapply(find_dist, x=nest_train, MoreArgs=list(y=j))
    return(which.min(d))
  }
  
  find_dist <- function(x,y) {
    return((sum((x - y) ^ 2))^(0.5))
  } 
  
  temp <- lapply(nest_test, where_min)
  temp <- unlist(temp)
  temp <- cbind(test["id"],as.data.frame(unlist(temp)))
  colnames(temp)[colnames(temp)=="unlist(temp)"] <- "nbor_id"
  temp <- join(temp[c("id","nbor_id")], train[c("nbor_id",imp_var)], by=c("nbor_id"), type="left")
  temp <- temp[c("id",imp_var)]
  return(temp)
}
# ============================ #
# 1B. logit_leave_method
# ============================ #
# logit imputation of leave characteristics

logit_leave_method <- function(d_test, d_train, xvars=NULL, yvars, test_filts, train_filts, 
                               weights, create_dummies) {
  
  # placeholder modification of xvars to follow Chris' specification in python
  # should be removed in final version
  # xvars <- c('age', 'agesq', 'male', 'wkhours', 'ltHS', 'BA', 'GradSch', 
  #            'empgov_fed', 'empgov_st', 'empgov_loc',
  #            'lnfaminc', 'black', 'asian', 'hisp', 'other',
  #            'ndep_kid', 'ndep_old', 'nevermarried', 'partner',
  #            'widowed', 'divorced', 'separated')
  
  # population mean imputation for missing xvars in logit regression
  if (xvars[1]!="") {
    for (i in xvars) {
      d_train[is.na(d_train[,i]), i] <- 0
      d_test[is.na(d_test[,i]), i] <- mean(d_test[,i], na.rm = TRUE)
    }  
  }
  
  # remove prop_pay from lists as we need to use ordinal regression for it
  train_filts <- list.remove(train_filts, 'prop_pay')
  test_filts <- list.remove(test_filts, 'prop_pay')
  yvars <- list.remove(yvars, 'prop_pay')
  weights <- list.remove(weights, 'prop_pay')
  
  # generate formulas for logistic regression
  # need formula strings to look something like "take_own ~ age + agesq + male + ..." 
  
  if (xvars[1]!="") {
    formulas=c()
    for (i in yvars) { 
      formulas= c(formulas, 
                  paste(i, "~",  paste(xvars[1],'+', paste(xvars[2:length(xvars)] , collapse=" + "))))
    }
  }
  else {
    formulas=c()
    for (i in yvars) { 
      formulas= c(formulas, paste(i, "~ 1"))
    }
  }
  
  # create columns based on logit estimates  
  sets <-  mapply(runLogitEstimate, formula = formulas, train_filt = train_filts,
                       test_filt=test_filts, weight = weights, varname=yvars,
                       MoreArgs=list(d_train=d_train, d_test=d_test, create_dummies=TRUE), 
                       SIMPLIFY = FALSE)

  # merge imputed values into single data set
  
  for (i in sets) {
    # old merge code, caused memory issues. using match instead
    # d_test <- merge(i, d_test, by="id",all.y=TRUE)
    for (j in names(i)) {
      if (j %in% names(d_test)==FALSE){
        d_test[j] <- i[match(d_test$id, i$id), j]    
      }
    }
    # set missing probability = 0
    d_test[is.na(d_test[colnames(i[2])]), colnames(i[2])] <- 0
  } 
  
  # set formula
  if (xvars[1]!="") {
    formula <- paste("factor(prop_pay) ~", paste(xvars[1],'+', paste(xvars[2:length(xvars)], collapse=" + ")))
  }
  else {
    formula <- paste("factor(prop_pay) ~ 1")
  }
    
  # Do an ordinal logit imputation for prop_pay
  d_filt <- runOrdinalEstimate(d_train=d_train,d_test=d_test, formula=formula,
                               test_filt="TRUE", train_filt="TRUE", varname='prop_pay')
  
  # old merge code caused memory issues. Using match instead.
  #d_test <- merge(d_filt, d_test, by='id', all.y=TRUE)
  for (i in names(d_filt)) {
    if ((i %in% names(d_test))==FALSE){
      d_test[i] <- d_filt[match(d_test$id, d_filt$id), i]    
    }
  }

  # replace factor levels with prop_pay proportions
  d_test <- d_test %>% mutate(prop_pay = ifelse(prop_pay == 1, 0, prop_pay))
  d_test <- d_test %>% mutate(prop_pay = ifelse(prop_pay == 2, .125, prop_pay))
  d_test <- d_test %>% mutate(prop_pay = ifelse(prop_pay == 3, .375, prop_pay))
  d_test <- d_test %>% mutate(prop_pay = ifelse(prop_pay == 4, .5, prop_pay))
  d_test <- d_test %>% mutate(prop_pay = ifelse(prop_pay == 5, .625, prop_pay))
  d_test <- d_test %>% mutate(prop_pay = ifelse(prop_pay == 6, .875, prop_pay))
  d_test <- d_test %>% mutate(prop_pay = ifelse(prop_pay == 7, 1, prop_pay))
  
  return(d_test)
}

# ============================ #
# 1Ba. runLogitEstimate
# ============================ #
# function to construct logit estimation model from training data set, 
# then create imputed columns for valid observations in test data set
# returns a separate copy of only those observations with valid imputed values

runLogitEstimate <- function(d_train,d_test, formula, test_filt,train_filt, weight, 
                             varname, create_dummies){
  des <- svydesign(id = ~1,  weights = as.formula(weight), data = d_train %>% filter_(train_filt))
  complete <- svyglm(as.formula(formula), data = d_train %>% filter_(train_filt),
                     family = "quasibinomial",design = des)
  estimate <- complete$coefficients 
  
  # if making a log, record sample size of filtered data set
  if (exists('makelog')) {
    if ( makelog == TRUE) {
      temp_filt = d_train %>% filter_(train_filt)
      cat("", file = log_name, sep="\n", append = TRUE)
      cat("------------------------------", file = log_name, sep="\n", append = TRUE)
      cat(paste("Filtered FMLA Sample Size:", nrow(temp_filt)), file = log_name, sep="\n", append = TRUE)
      cat(paste("Formula:", formula), file = log_name, sep="\n", append = TRUE)
      cat(paste("Filter condition:", train_filt), file = log_name, sep="\n", append = TRUE)
      cat("------------------------------", file = log_name, sep="\n", append = TRUE)
      cat("", file = log_name, sep="\n", append = TRUE)
    }
  }
  var_prob= paste0(varname,"_prob")
  d_filt <- d_test %>% filter_(test_filt)
  d_filt[var_prob]=estimate['(Intercept)']
  for (dem in names(estimate)) {
    if (dem !='(Intercept)' & !is.na(estimate[dem])) { 
      d_filt[is.na(d_filt[,dem]),dem]=0
      d_filt[var_prob]= d_filt[,var_prob] + d_filt[,dem]*estimate[dem]
    }
  }
  
  d_filt[var_prob] <- with(d_filt, exp(get(var_prob))/(1+exp(get(var_prob))))
  d_filt <- d_filt[,c(var_prob, 'id')]
  
  # option to create dummy variables in addition to probabilities
  if (create_dummies==TRUE) {
    d_filt [is.na(d_filt[var_prob]), var_prob] <- 0
    d_filt['rand']=runif(nrow(d_filt))
    d_filt[varname] <- with(d_filt, ifelse(rand>get(var_prob),0,1))    
    d_filt <- d_filt[,c(varname, 'id')]
  }
  
  return(d_filt)
}

# ============================ #
# 1Bb. runOrdinalEstimate
# ============================ #
# MASS implementation, polr function
# biggest problem with ordered logit currently is it is unweighted; can't use CPS weight without getting a non-convergence error
runOrdinalEstimate <- function(d_train,d_test, formula, test_filt,train_filt, varname){
  
  # 
  #   # OGLMX ordinal implementation - gives pretty non sensical results from my efforts
  #   runOrdinal <- function(x,y,z){
  #      results.ologit <- oglmx(as.formula(x), data = d_cps %>% filter_(y), weights=marsupwt)
  #      pause()
  #      return(estimate)
  #   }
  
  # get estimates from training data
  estimate <- polr(as.formula(formula), data = d_train %>% filter_(train_filt))
  
  #filter test data
  d_filt <- d_test %>% filter_(test_filt)
  
  # ensure there is at least one row in test data set that needs imputing
  if (!is.null(rownames(d_filt))) {
    
    # calculate score from ordinal model
    model=estimate$coefficients
    d_filt['var_score']=0
    for (dem in names(model)) {
      if (dem !='(Intercept)') { 
        d_filt[is.na(d_filt[,dem]),dem]=0
        d_filt[,'var_score']= d_filt[,'var_score'] + d_filt[,dem]*model[dem]
      }
    }
    
    # assign categorical variable based on ordinal cuts
    cuts= estimate$zeta
    cat_num= length(cuts)+1
    d_filt[varname] <- 0
    d_filt['rand']=runif(nrow(d_filt))
    for (i in seq(cat_num)) {
      if (i!=cat_num) {
        d_filt <- d_filt %>% mutate(cumprob= var_score-cuts[i])
        d_filt <- d_filt %>% mutate(cumprob2= exp(cumprob)/(1+exp(cumprob)))
        d_filt[varname] <- with (d_filt, ifelse(get(varname)==0 & rand>=cumprob2,i,get(varname)))
      }
      else {
        d_filt[varname] <- with (d_filt, ifelse(get(varname)==0,i,get(varname)))
      }
    }
    d_filt <- d_filt[,c(varname, 'id')]
    return(d_filt)
  }
}

# ============================ #
# 1Bc. runRandDraw
# ============================ #
# run a random draw
runRandDraw <- function(d_train,d_test,yvar,train_filt,test_filt, ext_resp_len, len_method) {
  # filter training cases
  d_temp <- d_train %>% filter_(train_filt)
  train <- d_temp %>% filter(complete.cases(yvar))
  
  # log FMLA sample sizes
  if (makelog == TRUE) {
    cat("", file = log_name, sep="\n", append = TRUE)
    cat("------------------------------", file = log_name, sep="\n", append = TRUE)
    cat(paste("Filtered FMLA Sample Size:", nrow(train)), file = log_name, sep="\n", append = TRUE)
    cat(paste("Formula: Random draw from variable",yvar), file = log_name, sep="\n", append = TRUE)
    cat(paste("Filter condition:", train_filt), file = log_name, sep="\n", append = TRUE)
    cat(paste("Weighted mean of var:",weighted.mean(train[,yvar], 
                                                    train[,'weight'], na.rm = TRUE)), file = log_name, sep="\n", append = TRUE)
    cat("------------------------------", file = log_name, sep="\n", append = TRUE)
    cat("", file = log_name, sep="\n", append = TRUE)
  }
  
  # filter test cases
  test <- d_test %>% filter_(test_filt)
  
  # verify that mean_diffs are >1 in FMLA for all types (counterfact lengths should be greater
  # than status quo)
  # this is a check for when len_method == 'mean'
  # mean_diff = mean(train %>% filter(resp_len == 0) %>% pull(yvar), na.rm=TRUE) -
  #   mean(train %>% filter(resp_len == 1) %>% pull(yvar), na.rm=TRUE) 
  # mean_ratio = mean(train %>% filter(resp_len == 0) %>% pull(yvar), na.rm=TRUE) /
  #   mean(train %>% filter(resp_len == 1) %>% pull(yvar), na.rm=TRUE) 
  # print(c(yvar, mean(train %>% filter(resp_len == 0) %>% pull(yvar), na.rm=TRUE), 
  #         mean(train %>% filter(resp_len == 1) %>% pull(yvar), na.rm=TRUE), mean_ratio,
  #         mean_diff))
  
  
  if (nrow(test)==0) {
    return()
  }
  
  if (nrow(test)!=0) {
    # random draw
    if (!yvar %in% colnames(test)) {
      test[yvar]=NA
    }
    
    squo_var = paste0('squo_',yvar)
    est_df <- data.frame(matrix(ncol = 3, nrow = 0))  
    colnames(est_df) <- c('id', squo_var, yvar)
    # status quo length
    for (i in c('resp_len == 0', 'resp_len == 1')) {
      temp_train <- train %>% filter_(i)
      temp_test <- test %>% filter_(i)
      if (nrow(temp_test)!= 0 & nrow(temp_train)!= 0 ) {
        train_samp_restrict <- function(x) temp_train[sample(nrow(temp_train), 1), yvar]
        temp_test[squo_var] <- apply(temp_test[yvar],1, train_samp_restrict)
        est_df <- rbind(est_df, temp_test[c('id', squo_var)])
      }
    }
    
    
    # changing counterfactual length option:
    # for constrained individuals, draw length from unconstrained draws
    if (ext_resp_len==TRUE) {
      # for those with resp_len = 1, we draw leave length from 
      # the unconstrained distribution of training leave lengths
      temp_train <- train %>% filter(resp_len == 0)
      temp_test <- test %>% filter(resp_len == 1)
      
      # old merge code caused memory issues. using match instead
      #temp_test <- merge(temp_test, est_df, by='id', all.x=TRUE)
      for (i in names(est_df)) {
        if (i %in% names(temp_test)==FALSE){
          temp_test[i] <- est_df[match(temp_test$id, est_df$id), i]    
        }
      }
      
      
      # if these dataframes are not empty, we find the counterfactual lengths
      if (nrow(temp_test)!= 0 & nrow(temp_train)!= 0 ) {
        # mean method - find proportional difference of resp=1 and resp=0 mean, and multiply 
        # status quo lengths by that factor.
        if (len_method=='mean') {
          train_samp_cfact <- function(x) {
            if (nrow(temp_train %>% filter(get(yvar)>= x[squo_var]))!=0) {
              data <- temp_train %>% filter(get(yvar)>= x[squo_var]) %>% select_(yvar, 'weight')
              mean <- weighted.mean(x= data[ ,yvar], w= data[ , 'weight'])
              return(round(mean))
            }
            else {
              return(x[squo_var])
            }            
          }
        }
        
        # random draw method - take random draw from training sample of lengths less than or equal to
        # the counterfactual leave length
        if (len_method=='rand') {
          train_samp_cfact <- function(x) {
            if (nrow(temp_train %>% filter(get(yvar)>= x[squo_var]))!=0) {
              return(temp_train %>% filter(get(yvar)>= x[squo_var]) %>% sample_n(1, weight = weight) %>% select_(yvar))
            }
            else {
              return(x[squo_var])
            }
          }
        }
        # adjust squo lengths by factor to get counterfact lengths for resp_len == 1
        temp_test[yvar] <- data.frame(unlist(apply(temp_test, 1, train_samp_cfact)))
        
        # old merge code caused memory issues. using match instead
        #est_df <- merge(temp_test[c('id', yvar)], est_df, by='id', all.y=TRUE)  
        
        est_df[yvar] <- temp_test[match(est_df$id,temp_test$id), yvar]    
        
        # for the rest, resp_len = 0 and so leave length does not respond to presence or absence of program, 
        # so that variable remains the same
        est_df[is.na(est_df[yvar]),yvar] <- est_df[is.na(est_df[yvar]),squo_var]
      }
      # if these dataframes are empty, then we only have resp_len = 0 in test and we make the cfact var the same
      # as status quo: their leave length does not respond to presence or absence of program, 
      else {
        est_df[yvar] <- est_df[squo_var]
      }
    }
    
    # if option not used, status quo will start out the same as counterfactual.
    if (ext_resp_len==FALSE) {
      est_df[yvar] <- est_df[squo_var]
    }
    return(est_df) 
  }
}

# ============================ #
# 1C. KNN_multi
# ============================ #
# Define KNN matching method, but allowing multiple (k) neighbors 
KNN_multi <- function(d_train, d_test, imp_var, train_filt, test_filt, xvars, kval=5) { 
  
  # starts the same way as KNN1_scratch
  
  # This returns a dataframe of length equal to acs with the employee id and a column for each leave type
  # that indicates whether or not they took the leave.
  
  # create training data
  
  # filter dataset and keep just the variables of interest
  train <-  d_train %>% filter(complete.cases(select(d_train, 'id', imp_var,xvars))) %>% 
    filter_(train_filt) %>%
    select(imp_var, xvars) %>%
    mutate(id = NULL)
  train ['nbor_id'] <- as.numeric(rownames(train))
  
  # create test data set 
  # This is a dataframe just with the variables in the acs that will be used to compute distance
  
  test <- d_test %>% filter_(test_filt) %>%
    select(id, xvars) %>%
    filter(complete.cases(.))
  
  # Initial checks
  
  # check for data frames
  if ((!is.data.frame(train)) | (!is.data.frame(test))) {
    stop("train_set and test_set must be data frames")
  }  
  
  # check for missing data
  if (anyNA(train) | anyNA(test)) {
    stop("missing values not allowed in train_test or test_set")
  }
  
  
  # normalize training data to equally weight differences between variables
  for (i in colnames(train)) {
    if (i != 'nbor_id' & i != imp_var & sum(train[i])!=0 ){
      train[i] <- scale(train[i],center=0,scale=max(train[,i]))
    }
  } 
  
  for (i in colnames(test)) {
    if (i != 'id' & sum(test[i])!=0 ){
      test[i] <- scale(test[i],center=0,scale=max(test[,i]))
    }
  } 
  
  # id var must be first variable of data
  
  # find distance
  
  m_test <- as.matrix(test)
  m_train <-as.matrix(train)
  
  nest_test <- list()
  nest_train <- list()
  
  # nested lists of vectors for apply functions
  nest_test <- lapply(seq(1,nrow(m_test)) , function(y){ 
    m_test[y,colnames(test)!='id']
  })
  nest_train <- lapply(seq(1,nrow(m_train)) , function(y){ 
    m_train[y,colnames(train)!='nbor_id' & colnames(train)!=imp_var]
  })
  
  # mark minimium distance
  min_start <- ncol(train)-2
  
  find_dist <- function(x,y) {
    return((sum((x - y) ^ 2))^(0.5))
  } 
  
  # same as KNN1 up until here
  where_min <- function(j) {
    min_dist <- min_start
    d <- mapply(find_dist, x=nest_train, MoreArgs=list(y=j))
    nbors <- order(d)[1:kval]
    return(nbors)
  }
  
  # create nbors: data set for each test obs' nearest neighbors in train dataset with their index
  nbors <- lapply(nest_test, where_min)
  nbors <- as.data.frame(t(as.data.frame(nbors)))
  # store test id in nbors set before continuing
  rownames(nbors) <- NULL
  colnames(nbors) <- paste0(rep('nbor_id',kval), seq(1,kval))
  nbors['id'] <- test$id
  # add impute variable's values for each index from training data set
  for (i in seq(kval)) {
    nbors <- merge(nbors, train[c('nbor_id',imp_var)], by.x=paste0('nbor_id',i), by.y='nbor_id', how='left')
    colnames(nbors)[kval+i+1] <- c(paste0('nbor_val', i))
  }

  # To decide on a single value to impute:
  # we will pick the mode value among nbors, weighted by distance
  # There are other ways of picking a single value, I'm sure
  
  # throwaway function to get mode of rows
  Mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }
  nbors[imp_var] <- apply(nbors[(kval+2):(kval*2+1)], 1, Mode)
  temp <- merge(nbors[c('id',imp_var)], test['id'], by='id')
  return(temp)
}

# ============================ #
# 1D. Naive_Bayes
# ============================ #
# Naive Bayes imputation function

Naive_Bayes <- function(d_train, d_test, yvars, train_filts, test_filts, weights, xvars) {
  
  # generate formulas for Naive Bayes model
  # need formula strings to look something like "take_own ~ age + agesq + male + ..." 
  formulas <- c()
  for (i in yvars) { 
    formulas <- c(formulas, 
                paste(i, "~",  paste(xvars[1],'+', paste(xvars[2:length(xvars)] , collapse=" + "))))
  }
  names(formulas) <- names(yvars)
  
  # ---- Using WANBIA weights
  # from -> https://rdrr.io/cran/bnclassify/src/R/learn-params-wanbia.R 
  
  # predict each yvar 
  for (i in names(yvars)) {
    # # generate NB model from training data
    w_train <- as.data.frame(sapply(d_train %>% filter(complete.cases(select(d_train, yvars[[i]], xvars))) 
                                    %>% filter_(train_filts[i]) %>% select(xvars, yvars[[i]]), as.factor))
    
    #wanbia <- compute_wanbia_weights('prop_pay', as.data.frame(sapply(w_train, as.factor))) 
    wanbia <- bnc(dag_learner = 'nb',class=yvars[[i]], dataset=w_train,smooth=0,wanbia=TRUE) 
    
    
    # apply model to test data 
    w_test <- as.data.frame(sapply(d_test %>% filter(complete.cases(select(d_test, xvars))) %>% filter_(test_filts[i])%>% select(xvars),as.factor))
    
    # make sure that testing and training sets have same factor levels
    for (j in xvars) {
      if (!all(levels(w_train[,c(j)]) %in% levels(w_test[,c(j)]))) {
        levels(w_test[,c(j)]) = levels(w_train[,c(j)])
      }
    }
    
    w_test_ids <- d_test %>% filter(complete.cases(select(d_test, xvars))) %>% filter_(test_filts[i])%>% select(id)
    wanbia_imp <- as.data.frame(predict(object=wanbia, newdata = w_test, prob=TRUE)) 
    
    # old version of NB impute; not done correctly
    # ftrain <- d_train %>% filter(complete.cases(select(d_train, yvars[i], xvars))) %>% filter_(train_filts[i])  
    # model <- naiveBayes(x = ftrain[xvars], y = ftrain[yvars[i]])

    # ftest <- d_test %>% filter(complete.cases(select(d_test, xvars))) %>% filter_(test_filts[i]) 
    # impute <- as.data.frame(predict(object=model, newdata = ftest[xvars], type='raw'))
    # impute[yvars[i]] <- apply(impute, 1, FUN=which.min)
    # impute[yvars[i]] <- apply(impute[yvars[i]],1,function(x) colnames(impute)[x])
    # impute[yvars[i]] <- as.numeric(impute[,yvars[i]])
    
    # Play wheel of fortune with which prop_val to assign to each test obs
    wanbia_imp['rand'] <- runif(nrow(wanbia_imp))
    if (i == 'prop_pay') {
      wanbia_imp['prop_pay'] <- NA
      wanbia_imp['cum']=0
      var_vals <- sapply(unname(sort(unlist(c(unique(w_train['prop_pay']))))),toString)
      for (j in var_vals) {
        wanbia_imp['prop_pay'] <- with(wanbia_imp, ifelse(rand > cum & rand<(cum + get(j)), j, prop_pay))
        wanbia_imp['cum']= wanbia_imp[,'cum'] + wanbia_imp[,j]
      }
      wanbia_imp['prop_pay'] <- as.numeric(wanbia_imp[,'prop_pay'])
    }
    else {
      wanbia_imp[yvars[[i]]] <- as.data.frame(ifelse(wanbia_imp[2]>wanbia_imp['rand'],1,0)) 
      #browser()
    }
    
    # add imputed value to test data set
    wanbia_imp <- cbind(w_test_ids['id'], wanbia_imp)
    # old merge code, was causing issues. Using match instead.
    #d_test <- merge(d_test, wanbia_imp[c('id', yvars[i])], by='id', all.x = TRUE)
    d_test[yvars[[i]]] <- wanbia_imp[match(d_test$id, wanbia_imp$id), yvars[[i]]]
    
  }
  return(d_test)
}

# ============================ #
# 1E. ridge_class
# ============================ #
# Ridge Classifier imputation function
ridge_class <- function(d_train, d_test, yvars, train_filts, test_filts, weights, xvars) {
  # generate formulas for Ridge Regression model
  # need formula strings to look something like "take_own ~ age + agesq + male + ..." 
  formulas <- c()
  for (i in yvars) { 
    formulas <- c(formulas, 
                  paste(i, "~",  paste(xvars[1],'+', paste(xvars[2:length(xvars)] , collapse=" + "))))
  }
  names(formulas) <- names(yvars)
  
  # predict each yvar
  
  for (i in names(yvars)) {
    # generate Ridge Regression model from training data
    ftrain <- d_train %>% filter(complete.cases(select(d_train, yvars[i], xvars))) %>% filter_(train_filts[i])  
    # normalize training data to equally weight differences between variables
    for (j in xvars) {
      if (sum(ftrain[j])!=0 ){
        ftrain[j] <- scale(ftrain[j],center=0,scale=max(ftrain[,j]))
      }
    } 
    
    # check for xvars with all identical values. need to not use xvar if all values are the same in training 
    # will return a column of NaN for svd() in the lm.ridge function
    temp_xvars <- xvars
    for (j in xvars) {
      
      if (dim(table(ftrain[,j])) == 1) {
        temp_xvars <- temp_xvars[!temp_xvars %in% j]
        formulas[i] <- paste(yvars[i], "~",  paste(temp_xvars[1],'+', paste(temp_xvars[2:length(temp_xvars)] 
                                                                            , collapse=" + ")))
      }
    }
    
    # as prop_pay is categorical, we need to do ordinal version of ridge regression
    # rest of the vars are handled in this loop
    if (i!= 'prop_pay') {
      
      model <- lm.ridge(formula = formulas[i], data = ftrain[c(yvars[i], temp_xvars)])
      
      # TODO: getting a weighted version of ridge regression to work
      
      # apply model to test data
      ftest <- d_test %>% filter(complete.cases(select(d_test, temp_xvars))) %>% filter_(test_filts[i]) 
      
      # normalize test data to equally weight differences between variables
      for (j in temp_xvars) {
        if (sum(ftest[j])!=0 ){
          ftest[j] <- scale(ftest[j],center=0,scale=max(ftest[,j]))
        }
      }
      # calculate probabilities of test data set
      impute <- as.data.frame(as.matrix(cbind(constant=1,ftest[temp_xvars])) %*% coef(model))
      colnames(impute) <- 'prob'
      impute['rand'] <- runif(nrow(impute))
      impute[yvars[i]] <- with(impute, ifelse(prob >= rand, 1, 0))
      impute <- cbind(ftest['id'], impute)
      # old merge code, was causing issues. Using match instead.
      #d_test <- merge(d_test, impute[c('id', yvars[i])], by='id', all.x = TRUE)
      d_test[yvars[[i]]] <- impute[match(d_test$id, impute$id), yvars[[i]]]
    }
    # ordinal ridge regression for prop_pay
    # can't find a package that does an ordinal implementation, so writing one from scratch here
    # TODO: Not exactly ordinal in implementation right now, if there's time to revisit this could be done 
    else {
      # create dummies for each value of prop_pay
      dums <- dummy(ftrain$prop_pay)
      var_vals <- paste0('prop_pay_',sort(unlist(c(unique(ftrain['prop_pay'])))))
      colnames(dums) <- var_vals
      ftrain <- cbind(ftrain, dums)
      
      # prep test data
      ftest <- d_test %>% filter(complete.cases(select(d_test, temp_xvars))) %>% filter_(test_filts[i]) 
      # normalize test data to equally weight differences between variables
      for (j in temp_xvars) {
        if (sum(ftest[j])!=0 ){
          ftest[j] <- scale(ftest[j],center=0,scale=max(ftest[,j]))
        }
      }
      
      for (j in var_vals) {
        # run ridge regression on each dummy
        formula <- paste(j, "~",  paste(temp_xvars[1],'+', paste(temp_xvars[2:length(temp_xvars)] 
                                                                        , collapse=" + ")))
        model <- lm.ridge(formula = formula, data = ftrain[c(j, temp_xvars)])  
        
        # apply model to test data to get probabilities of each level
        impute <- as.data.frame(as.matrix(cbind(constant=1,ftest[temp_xvars])) %*% coef(model))
        colnames(impute) <- j
        impute <- cbind(ftest['id'], impute)
        
        # old merge code, was causing issues. Using match instead.
        #d_test <- merge(d_test, impute[c('id', j)], by='id', all.x = TRUE)
        d_test[j] <- impute[match(d_test$id, impute$id), j]
      }
      # normalize val probabilities to sum to 1
      d_test['total'] = rowSums(d_test[var_vals])
      for (j in var_vals) {
        d_test[j] =  d_test[j]/d_test['total']
      }
      
      # Play wheel of fortune with which prop_val to assign to each test obs
      d_test['rand'] <- runif(nrow(d_test))
      d_test['prop_pay'] <- NA
      d_test['cum']=0
      for (j in var_vals) {
        pay_val <- as.numeric(gsub("prop_pay_", "", j))
        d_test['prop_pay'] <- with(d_test, ifelse(rand > cum & rand<(cum + get(j)), pay_val, prop_pay))
        d_test['cum']= d_test[,'cum'] + d_test[,j]
      }
      
      # clean up columns 
      d_test <- d_test[!colnames(d_test) %in% c(var_vals, 'rand', 'cum')]
    }
  }
  return(d_test)
}

# ============================ #
# 1F. random_forest
# ============================ #


# Random Forest imputation function
random_forest <- function(d_train, d_test, yvars, train_filts, test_filts, weights, xvars) {
  # generate formulas for Random Forest model
  # need formula strings to look something like "take_own ~ age + agesq + male + ..." 
  formulas <- c()
  for (i in yvars) { 
    formulas <- c(formulas, 
                  paste(i, "~",  paste(xvars[1],'+', paste(xvars[2:length(xvars)] , collapse=" + "))))
  }
  names(formulas) <- names(yvars)
  
  # predict each yvar 
  for (i in names(yvars)) {
    # generate model from training data 
    ftrain <- d_train %>% filter(complete.cases(select(d_train, yvars[i], xvars))) %>% filter_(train_filts[i])  
    model <- randomForest(x = ftrain[xvars], y = factor(ftrain[,yvars[i]]))
    # apply model to test data 
    ftest <- d_test %>% filter(complete.cases(select(d_test, xvars))) %>% filter_(test_filts[i]) 
    impute <- as.data.frame(predict(object=model, newdata = ftest[xvars], type='prob'))
  
    # play wheel of fortune with predicted probabilities to get imputed value
    impute['rand'] <- runif(nrow(impute))
    # do this differently for prop_pay as a non-binary categorical var
    if (i=='prop_pay'){
      impute['prop_pay'] <- NA
      impute['cum']=0
      var_vals <- sapply(unname(sort(unlist(c(unique(ftrain['prop_pay']))))),toString)
      for (j in var_vals) {
        impute['prop_pay'] <- with(impute, ifelse(rand > cum & rand<(cum + get(j)), j, prop_pay))
        impute['cum']= impute[,'cum'] + impute[,j]
      }
      impute['prop_pay'] <- as.numeric(impute[,'prop_pay'])
    }
    # rest of vars are binary
    else {
      impute[yvars[i]] <- as.data.frame(ifelse(impute[2]>impute['rand'],1,0))  
    }
    
    # add imputed value to test data set
    impute <- cbind(ftest['id'], impute)
    # old merge code, was causing issues. Using match instead.
    #d_test <- merge(d_test, impute[c('id', yvars[i])], by='id', all.x = TRUE)
    d_test[yvars[[i]]] <- impute[match(d_test$id, impute$id), yvars[[i]]]
    
  }
  return(d_test) 
}


# ================================ #
# 1G. Support Vector Machine
# ================================ #
# SVM Imputation Function
# Functional, but produces pretty bizarre results. Never predicts leave 
svm_impute <- function(d_train, d_test, yvars, train_filts, test_filts, weights, xvars) {
  # generate formulas for model
  # need formula strings to look something like "take_own ~ age + agesq + male + ..." 
  formulas <- c()
  for (i in yvars) { 
    formulas <- c(formulas, 
                  paste(i, "~",  paste(xvars[1],'+', paste(xvars[2:length(xvars)] , collapse=" + "))))
  }
  names(formulas) <- names(yvars)
  
  # predict each yvar 
  for (i in names(yvars)) {
    
    # generate model from training data 
    ftrain <- d_train %>% filter(complete.cases(select(d_train, yvars[i], xvars))) %>% filter_(train_filts[i])  
    # normalize training data to equally weight differences between variables
    for (j in xvars) {
      if (sum(ftrain[j])!=0 ){
        ftrain[j] <- scale(ftrain[j],center=0,scale=max(ftplotrain[,j]))
      }
    } 
    # check for xvars with all identical values. need to not use xvar if all values are the same in training 
    # will return a column of NaN for svd() in the lm.ridge function
    temp_xvars <- xvars
    for (j in xvars) {
      if (dim(table(ftrain[,j])) == 1) {
        temp_xvars <- temp_xvars[!temp_xvars %in% j]
        formulas[i] <- paste(yvars[i], "~",  paste(temp_xvars[1],'+', paste(temp_xvars[2:length(temp_xvars)] 
                                                                            , collapse=" + ")))
      }
    }  
    model <- svm(x = ftrain[temp_xvars], y = factor(ftrain[,yvars[i]]), scale = FALSE)
    # apply model to test data 
    ftest <- d_test %>% filter(complete.cases(select(d_test, temp_xvars))) %>% filter_(test_filts[i]) 
    # normalize test data to equally weight differences between variables
    for (j in temp_xvars) {
      if (sum(ftest[j])!=0 ){
        ftest[j] <- scale(ftest[j],center=0,scale=max(ftest[,j]))
      }
    }
    impute <- as.data.frame(unfactor(predict(object=model, newdata = ftest[temp_xvars])))
    colnames(impute)[1] <- yvars[i]
    if (i=='prop_pay'){
      #browser()
    }

    # add imputed value to test data set
    impute <- cbind(ftest['id'], impute)
    # old merge code, caused memory issues. using match instead
    #d_test <- merge(d_test, impute[c('id', yvars[i])], by='id', all.x = TRUE)
    d_test[yvars[[i]]] <- impute[match(d_test$id, impute$id), yvars[[i]]]
    
  }
  return(d_test) 
}

# ================================ #
# 1H. xG Boost
# ================================ #
# xG boost imputation function
xg_boost_impute <- function(d_train, d_test, yvars, train_filts, test_filts, weights, xvars) {
  # generate formulas for model
  # need formula strings to look something like "take_own ~ age + agesq + male + ..." 
  formulas <- c()
  for (i in yvars) { 
    formulas <- c(formulas, 
                  paste(i, "~",  paste(xvars[1],'+', paste(xvars[2:length(xvars)] , collapse=" + "))))
  }
  names(formulas) <- names(yvars)
  
  # predict each yvar 
  for (i in names(yvars)) {
    
    # generate model from training data 
    ftrain <- d_train %>% filter(complete.cases(select(d_train, yvars[i], xvars))) %>% filter_(train_filts[i])  
    # normalize training data to equally weight differences between variables
    for (j in xvars) {
      if (sum(ftrain[j])!=0 ){
        ftrain[j] <- scale(ftrain[j],center=0,scale=max(ftrain[,j]))
      }
    } 
    
    # check for xvars with all identical values. need to not use xvar if all values are the same in training 
    temp_xvars <- xvars
    for (j in xvars) {
      if (dim(table(ftrain[,j])) == 1) {
        temp_xvars <- temp_xvars[!temp_xvars %in% j]
        formulas[i] <- paste(yvars[i], "~",  paste(temp_xvars[1],'+', paste(temp_xvars[2:length(temp_xvars)] 
                                                                            , collapse=" + ")))
      }
    } 
    # normalize for test xvars
    ftest <- d_test %>% filter(complete.cases(select(d_test, temp_xvars))) %>% filter_(test_filts[i]) 
    for (j in temp_xvars) {
      if (sum(ftest[j])!=0 ){
        ftest[j] <- scale(ftest[j],center=0,scale=max(ftest[,j]))
      }
    }
    
    # set up training and testing data as xgb objects
    mtrain <-as.matrix(ftrain[,c(temp_xvars,yvars[[i]])])
    xgb_train <- xgb.DMatrix(data=mtrain[,temp_xvars],label=mtrain[,yvars[i]])
    mtest <-as.matrix(ftest[,c(temp_xvars)])
    xgb_test <- xgb.DMatrix(data=mtest[,temp_xvars])
    
    # create training model
    model <- xgb.train(data = xgb_train,nrounds=2)

    # apply model to test data 
    impute <- as.data.frame(predict(object=model, newdata = xgb_test))
    colnames(impute)[1] <- yvars[i]
    
    # if a binary variable, play wheel of fortune to assign value based on estimated probability
    if (i != 'prop_pay') {
      impute['rand'] <- runif(nrow(impute))
      impute[yvars[[i]]] <- as.data.frame(ifelse(impute[1]>impute['rand'],1,0))
    }
    
    # add imputed value to test data set
    impute <- cbind(ftest['id'], impute)
    
    # old merge code, caused memory issues. using match instead
    #d_test <- merge(d_test, impute[c('id', yvars[i])], by='id', all.x = TRUE)
    d_test[yvars[[i]]] <- impute[match(d_test$id, impute$id), yvars[[i]]]
  }
  return(d_test) 
}