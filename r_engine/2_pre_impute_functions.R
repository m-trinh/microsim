
# """
# 2_pre_impute_functions
#
# These functions prepare the FMLA data set for ACS to impute leave taking behavior from it.
# and then execute the imputation from CPS, then FMLA into ACS.
#
# 9 Sept 2018
# Luke
# 
# TESTING TODO: what happens when filtered test data sets of 0 obs are fed into imputation functions
#               currently is handled properly by runOrdinalImpute and runRandDraw. 
#                KNN1_scratch and Logitestimate do not currently, should consider including handling methods.
#
# """



#~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Table of Contents
#~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. impute_intra_fmla
  # 1A. runLogitEstimate - see 3_impute_functions.R, function 1Ba
# 2. SMOTE 
# 3. acs_filtering

# ============================ #
# 1. impute_intra_fmla
# ============================ #

# Leave lengths for multiple reasons need to be imputed for multi-leave takers
# 
# We observe the number of reasons leave was taken, but observe only 1 or 2 of the actual reasons
# We will impute the other leave types taken based on logit model from ACM


impute_intra_fmla <- function(d_fmla, intra_impute) {
  # intra-fmla imputation for additional leave taking and lengths
  # This is modifying take_ and need_ vars for those with additional leaves
  
  # ---------------------------------------------------------------------------------------------------------
  # A. For those multi-leave takers who responded to longest leave for a different reason, use that reason/length 
  #    rather than imputing
  # ---------------------------------------------------------------------------------------------------------
  # count number of leaves needed

  varnames <- paste0("take_",leave_types)
  
  # leave_count is tracking the number of variables to be imputed. Sorry, probably should have used a better var name
  # Num_leaves_take and the rowsums term are not consistent, hence this check beyond just the observation of
  # only longest and most recent leaves.

  d_fmla['leave_count']=d_fmla['num_leaves_take']- rowSums(d_fmla[,varnames], na.rm=TRUE)
  d_fmla['long_flag']=0
  for (i in leave_types) {
    take_var <- paste0("take_",i)
    len_var  <- paste0("length_",i)
    long_var <- paste0("long_",i)
    longlen_var <- paste0("longlength_",i)
    
    # flag those whose longest leave is used
    d_fmla['long_flag'] <- with(d_fmla, ifelse(get(long_var)==1 & leave_count>0 & get(take_var)==0,1,long_flag))  
    
    # alter length of 2nd leave type to match longest leave
    d_fmla[len_var] <- with(d_fmla, ifelse(get(long_var)==1 & leave_count>0 & get(take_var)==0,get(longlen_var),get(len_var))) 
    
    # alter take_var of 2nd leave type of longest leave
    d_fmla[take_var] <- with(d_fmla, ifelse(get(long_var)==1 & leave_count>0 & get(take_var)==0,1,get(take_var)))
  }
  
  # ---------------------------------------------------------------------------------------------------------
  # B. Types of Leave taken for multiple leave takers
  # ---------------------------------------------------------------------------------------------------------
  # creating throwaway function to easily repeat this for leave needers
  temp_func <- function(lname){

    # specifications
    # using ACM specifications
    formulas <- c(own = paste0(lname, "_own ~ age + male + lnfaminc + black + hisp + coveligd"),
                illspouse = paste0(lname, "_illspouse ~ 1"),
                illchild = paste0(lname, "_illchild ~ 1"),
                illparent = paste0(lname, "_illparent ~ 1"),
                matdis = paste0(lname, "_matdis ~ 1"),
                bond = paste0(lname, "_bond ~ 1"))
    
    # subsetting data
    train_filts <- c(own = "TRUE",
                     illspouse = "nevermarried == 0 & divorced == 0",
                     illchild = "TRUE",
                     illparent = "TRUE",
                     matdis = "female == 1 & nochildren == 0",
                     bond = "nochildren == 0")
    
    varnames <- paste(lname,leave_types,sep="_")
    test_filts <- rep(paste0("num_leaves_", lname,">1 & long_flag==0"),length(varnames))
    
    # weights
    weights <- c(own = "~ fixed_weight",
                illspouse = "~ fixed_weight",
                illchild = "~ fixed_weight",
                illparent = "~ weight",
                matdis = "~ fixed_weight",
                bond = "~ fixed_weight")
    
    
    # Run Estimation
    # This is a candidate for modular imputation
    # INPUTS: FMLA/ACS (train/test) data set, Lists of regression specifications, conditional filters for 
    #         test/training data, and weight selection 
    sets <- mapply(runLogitEstimate, formula = formulas, train_filt = train_filts,
                        test_filt=test_filts, weight = weights, varname=varnames,
                        MoreArgs=list(d_train=d_fmla, d_test=d_fmla, create_dummies=FALSE), 
                        SIMPLIFY = FALSE)
    #OUTPUT: list of data sets of probabilities of each FMLA individual taking/needing a type of leave
    
    # Run leave type imputation on multi-leave takers based on these probabilities
    # merge imputed values with fmla data
    
    for (i in sets) {
      # set missings to 0
      d_fmla <- merge(i, d_fmla, by="id",all.y=TRUE)
      d_fmla[is.na(d_fmla[colnames(i[1])]), colnames(i[1])] <- 0
    }  
    
    # randomly select leave types for those taking multiple leaves from those types not already taken
    varnames <- paste(lname,leave_types,sep="_")
    
    d_fmla['leave_count']=d_fmla[paste0('num_leaves_',lname)]- rowSums(d_fmla[,varnames], na.rm=TRUE)
    
    # I looked for a canned package for multinomial drawing without replacement, but couldn't find one
    # So this existing loop of code is the only way I can see to do this
    for (n in seq(1, max(d_fmla['leave_count']))) {
      d_fmla['rand']=runif(nrow(d_fmla))
      
      # transform values to represent probabilities that sum to 1 of remaining possible leave types
      d_fmla['sum']=0
      for (i in leave_types) {
        var_name= paste(lname,i, sep="_")
        var_prob= paste(lname,i,"prob", sep="_")
        d_fmla[d_fmla[,var_name]==1 & !is.na(d_fmla[,var_name]), var_prob]= 0
        d_fmla['sum']= d_fmla[,'sum'] + d_fmla[,var_prob]
      }
      d_fmla['cum']=0
      for (i in leave_types) {
        var_prob= paste(lname,i,"prob", sep="_")
        var_name= paste(lname,i, sep="_")
        d_fmla[d_fmla['sum']!=0,var_prob]=  d_fmla[d_fmla['sum']!=0,var_prob]/d_fmla[d_fmla['sum']!=0,'sum']
        d_fmla[d_fmla['sum']==0,var_prob]=  0
        d_fmla[d_fmla['rand']>d_fmla['cum'] & d_fmla['rand']<(d_fmla['cum']+d_fmla[,var_prob]) & d_fmla['leave_count']>=n, var_name]=1
        d_fmla['cum']= d_fmla[,'cum'] + d_fmla[,var_prob]
      }
    }
    # Now we have FMLA data with imputed leave types for those taking/needing multiple leaves, 
    #  of which one or more are not observed in their responses to the FMLA survey 
  }
  
  # first run imputes take_* variables and updates d_fmla in place
  temp_func("take")
  
  # ---------------------------------------------------------------------------------------------------------
  # C. Types of Leave Needed for multiple leave needers
  # ---------------------------------------------------------------------------------------------------------
  # second run: perform same operations on need_* variables
  temp_func("need")

  # recalculate taker and needer vars
  d_fmla['taker']=rowSums(d_fmla[,paste('take',c("own","illspouse","illchild","illparent","matdis","bond"),sep="_")], na.rm=TRUE)
  d_fmla['needer']=rowSums(d_fmla[,paste('need',c("own","illspouse","illchild","illparent","matdis","bond"),sep="_")], na.rm=TRUE)
  d_fmla <- d_fmla %>% mutate(taker=ifelse(taker>=1, 1, 0))
  d_fmla <- d_fmla %>% mutate(needer=ifelse(needer>=1, 1, 0))
  
  return(d_fmla)
}

# ============================ #
# 2. SMOTE
# ============================ #
# function to apply SMOTE to FMLA data
apply_smote <- function(d_fmla, xvars) {
  # iterate through all need and take vars for each leave type
  smote_dfs <- list()
  for (i in leave_types) {
    for (j in c('take_', 'need_')){
      yvar= paste0(j,i)
      d_smote <- d_fmla
      d_smote[,paste0('factor_',yvar)] <- as.factor(d_smote[,yvar])
      formula <- paste(paste0('factor_',yvar), "~",  paste(xvars[1],'+', paste(xvars[2:length(xvars)], collapse=" + ")))
      smote_dfs[yvar] <- SMOTE(as.formula(formula),d_smote)
    }
  }
  return(smote_dfs)
}

# ============================ #
# 3. ACS Filtering
# ============================ #
# Apply filters and modifications to ACS data based on user inputs
acs_filtering <- function(d, weightfactor, place_of_work, state) {
  
  if (weightfactor!=1) {
    d$PWGTP=d$PWGTP*weightfactor
  }  
  
  # apply state filters
  if (state!='') {
    if (place_of_work==FALSE) {
      # merge in state abbreviations
      state_codes <- read.csv(paste0("./csv_inputs/ACS_state_codes.csv"))
      d <- merge(d,state_codes, by="ST",all.x=TRUE)  
      d <- d %>% filter(state_abbr==state)
    }
    if (place_of_work==TRUE) {
      # merge in state abbreviations
      state_codes <- read.csv(paste0("./csv_inputs/ACS_state_codes.csv"))
      state_codes["POWSP"] <- state_codes["ST"]  
      d <- merge(d,state_codes, by="POWSP",all.x=TRUE)
      d <- d %>% filter(state_abbr==state)
    }
  }
  
  # make sure there's not zero observations
  if (nrow(d)==0) {
    stop('Error: no rows in ACS dataframe')
  }
  
  return(d)
}
