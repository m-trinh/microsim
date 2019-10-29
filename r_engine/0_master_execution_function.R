
# """
# File: 0_NEW_Master_execution_function
#
# Master execution function that calls all other functions in order to execute the simulation
# 
# 9 Sept 2018
# Luke
# 
# """
# ----------master function that sets policy scenario, and calls functions to alter FMLA data to match policy scenario---------------------
# see parameter documentation for details on arguments

policy_simulation <- function(saveCSV=FALSE,
                              FEDGOV=FALSE, 
                              STATEGOV=FALSE,
                              LOCALGOV=FALSE,
                              SELFEMP=FALSE, 
                              PRIVATE=TRUE,
                              impute_method="KNN1",
                              kval= 3,
                              makelog=TRUE,
                              xvars=c("widowed", "divorced", "separated", "nevermarried", "female", 
                                         'age',"agesq", "ltHS", "someCol", "BA", "GradSch", "black", 
                                         "other", "asian",'native', "hisp","nochildren",'faminc','coveligd'),

                              # default is 1's for everything
                              xvar_wgts = rep(1,length(xvars)),
                              base_bene_level=1, topoff_rate=0, topoff_minlength=0, sample_prop=NULL, sample_num=NULL,
                              bene_effect=FALSE, dependent_allow=0, full_particip_needer=FALSE, 
                              own_uptake=.25, matdis_uptake=.25, bond_uptake=.25, 
                              illparent_uptake=.25, illspouse_uptake=.25, illchild_uptake=.25, wait_period=0,
                              own_elig_adj=1, illspouse_elig_adj=1, illchild_elig_adj=1, 
                              illparent_elig_adj=1, matdis_elig_adj=1, bond_elig_adj=1,
                              clone_factor=0, sens_var = 'resp_len', progalt_post_or_pre ='post',
                              ext_resp_len = FALSE, len_method = 'mean',
                              intra_impute = TRUE,
                              place_of_work = FALSE,
                              state = '',
                              exclusive_particip=TRUE,
                              ext_base_effect=TRUE, extend_prob=0, extend_days=0, extend_prop=1,
                              maxlen_own =60, maxlen_matdis =60, maxlen_bond =60, maxlen_illparent =60, maxlen_illspouse =60, maxlen_illchild =60,
                              maxlen_PFL=maxlen_illparent+maxlen_illspouse+maxlen_illchild+maxlen_bond, maxlen_DI=maxlen_bond+maxlen_matdis,
                              maxlen_total=maxlen_DI+maxlen_PFL, week_bene_cap=1000000, week_bene_min=0, week_bene_cap_prop=NULL,
                              fmla_protect=TRUE, earnings=NULL, weeks= NULL, ann_hours=NULL, minsize= NULL, 
                              elig_rule_logic= '(earnings & weeks & ann_hours & minsize)',
                              formula_prop_cuts=NULL, formula_value_cuts=NULL, formula_bene_levels=NULL,
                              weightfactor=1, output=NULL, output_stats=NULL, random_seed=123,
                              SMOTE=FALSE) { # SMOTE still under construction, should remain false) {
  
  
  ####################################
  # Option to create execution log
  ####################################
  makelog <<- makelog
  if (makelog==TRUE) {
    # note starting time
    timestart <<- Sys.time()
    
    # save starting parameters
    params <- lapply(ls(), function(x) {return(get(x))})
    names(params) <- ls()[!ls() %in% c('params')]
    params['params'] <- NULL
    
    # create log file and record starting parameters 
    log_name <<- paste0('./logs/', output,' ', format(Sys.time(), "%Y-%m-%d %H.%M.%S"), '.log')
    file.create(log_name)
    cat("==============================", file = log_name, sep="\n")
    cat("Microsim Log File", file = log_name, sep="\n", append = TRUE)
    cat(paste0("Run Title: ", output), file = log_name, sep="\n", append = TRUE)
    cat(paste0("Date/Time: ", Sys.time()), file = log_name, sep="\n", append = TRUE)
    cat("==============================", file = log_name, sep="\n", append = TRUE)
    cat("", file = log_name, sep="\n", append = TRUE)
    cat("------------------------------", file = log_name, sep="\n", append = TRUE)
    cat("Parameters used", file = log_name, sep="\n", append = TRUE)
    cat("------------------------------", file = log_name, sep="\n", append = TRUE)
    lapply(names(params), function(x) cat(paste(x, ":", params[x]), file = log_name, sep="\n", append = TRUE))
    cat("", file = log_name, sep="\n", append = TRUE)
    
  }
  
  ####################################
  # global libraries used everywhere #
  ####################################
  pkgTest <- function(x)
  {
    if (!require(x,character.only = TRUE))
    {
      install.packages(x,dep=TRUE)
      if(!require(x,character.only = TRUE)) stop("Package not found")
    }
    return("OK")
  }
  
  global.libraries <- c('DMwR','xgboost','bnclassify', 'randomForest','magick','stats', 'rlist', 'MASS', 'plyr', 'dplyr', 
                        'survey', 'class', 'dummies', 'varhandle', 'oglmx', 
                        'foreign', 'ggplot2', 'reshape2','e1071','pander','ridge')
  
  
  results <- sapply(as.list(global.libraries), pkgTest)
  
  # run files that define functions
  source("1_cleaning_functions.R"); source("2_pre_impute_functions.R"); source("3_impute_functions.R"); 
  source("4_post_impute_functions.R"); source("5_output_analysis_functions.R")
  
  # set random seed option
  if (!is.null(random_seed)) {
    set.seed(random_seed)
  }
  

  #========================================
  # 1. Cleaning 
  #========================================
  # load files from a dataframe
   d_fmla <- readRDS(paste0("./R_dataframes/","d_fmla.rds"))
   d_cps <- readRDS(paste0("./R_dataframes/","d_cps.rds"))
   # load from residence ACS if state is a 
   if (place_of_work==TRUE) {
     d_acs <- readRDS(paste0("./R_dataframes/work_states/",state,"_work.rds"))  
   }
   else {
     d_acs <- readRDS(paste0("./R_dataframes/resid_states/",state,"_resid.rds"))  
   }

  #-----CPS to ACS Imputation-----
   #already done on R_dataframes, don't need to run here any more
   
  # Impute hourly worker, weeks worked, and firm size variables from CPS into ACS. 
  # These are needed for leave program eligibilitly determination
  # INPUT: cleaned acs, cps files
  # d_acs <- impute_cps_to_acs(d_acs, d_cps)
  # OUTPUT: cleaned acs with imputed weeks worked, employer size, and hourly worker status
  
  # sample ACS
  # user option to sample ACS data
  if (!is.null(sample_prop) & is.null(sample_num)) {
    d_acs <- sample_acs(d_acs, sample_prop=sample_prop, sample_num=NULL)  
  }
  if (is.null(sample_prop) & !is.null(sample_num)) {
    d_acs <- sample_acs(d_acs, sample_prop=NULL, sample_num=sample_num)  
  }
  if (!is.null(sample_prop) & !is.null(sample_num)) {
    d_acs <- sample_acs(d_acs, sample_prop=sample_prop, sample_num=sample_num)  
  }
  #========================================
  # 2. Pre-imputation 
  #========================================
  
  # General philosophy behind split between pre-ACS imputation and post-ACS imputation modifications:
  # Do as much as possible post-imputation, and impute as few FMLA variables as possible.
  
  # Pre-imputation: Alterations that are specific to FMLA variables. Mostly leave taking behavior.
  # Post-imputation: Alterations specific to ACS variables, or require both FMLA and ACS variables. 
  #                  Uptake behavior and benefit calculations.
  
  # define global values for use across functions
  leave_types <<- c("own","illspouse","illchild","illparent","matdis","bond")
  
  # preserve original copy of FMLA survey
  d_fmla_orig <- d_fmla 
  
  # INPUT: FMLA data
  # intra-fmla imputation for additional leave taking and needings
  d_fmla <- impute_intra_fmla(d_fmla, intra_impute)

  # OUTPUT: FMLA data with modified take_ and need_ vars for those with additional leaves
  
  # option to apply SMOTE to d_fmla data set to correct for class imbalance of each leave type
  if (SMOTE == TRUE) {
    smote_dfs <- apply_smote(d_fmla, xvars)
    browser()
  }
  
  # adjust for program's base behavioral effect on leave taking
  # INPUT: FMLA data
  # In presence of program, apply leave-taking behavioral updates
  if (progalt_post_or_pre == 'pre') {
    d_fmla <-LEAVEPROGRAM(d_fmla, sens_var)
  }
  # OUTPUT: FMLA data with adjusted take_leave columns to include 1s
  #         for those that would have taken leave if they could afford it
  
  #-----FMLA to ACS Imputation-----
  # filter/modify ACS data based on user specifications
  # INPUT: ACS Data
  d_acs <- acs_filtering(d_acs, weightfactor, place_of_work, state)
  # OUTPUT: Filtered ACS data
  # default is just simple nearest neighbor, K=1 
  # This is the big beast of getting leave behavior into the ACS.
  # INPUT: cleaned acs/fmla data, method for imputation, dependent variables 
  #         used for imputation
  d_acs_imp <- impute_fmla_to_acs(d_fmla,d_acs, impute_method, xvars, kval, xvar_wgts)  
  # OUTPUT: ACS data with imputed values for a) leave taking and needing, b) proportion of pay received from
  #         employer while on leave, and c) whether leave needed was not taken due to unaffordability 

  # -------------Post-imputation functions-----------------
  # adjust for program's base behavioral effect on leave taking
  # INPUT: FMLA data
  # In presence of program, apply leave-taking behavioral updates
  if (progalt_post_or_pre == 'post') {
    d_acs_imp <-LEAVEPROGRAM(d_acs_imp, sens_var)
  }
  # OUTPUT: FMLA data with adjusted take_leave columns to include 1s
  #         for those that would have taken leave if they could afford it
  
  # ---------------------------------------------------------------------------------------------------------
  # Impute Days of Leave Taken
  # ---------------------------------------------------------------------------------------------------------
  # INPUTS: unmodified FMLA survey as the training data, ACS as test data,
  #         and conditionals for filter the two sets. Using unmodified survey for leave lengths,
  #         since modified FMLA survey contains no new information about leave lengths, and 
  #         the intra-FMLA imputed leave lengths a random draw from that imputed data
  #         would produced a biased 
  #         estimate of leave length
  d_acs_imp <- impute_leave_length(d_fmla_orig, d_acs_imp, conditional, test_conditional, ext_resp_len,
                                   len_method)
  # OUTPUT: ACS data with lengths for leaves imputed
  
  # function interactions description (may not be complete, just writing as they come to me):
  # ELIGIBILITYRULES: eligibility rules defined, and participation initially set 
  # UPTAKE: Uptake probability values applied
  # BENEFITS: base values for benefits to be modified by other functions
  # BENEFITEFFECT: Overrides UPTAKE participation, but there are some classes of participants not affected by this
  # TOPOFF: TOPOFF overrides participation behavior of BENEFITEFFECT
  # DEPENDENTALLOWANCE: independent of other functions
  
  # Allow for users to clone ACS individuals 
  # INPUT: ACS file
  d_acs_imp <- CLONEFACTOR(d_acs_imp, clone_factor)
  # OUTPUT: ACS file with user-specifed number of cloned records

  # Assign employer pay schedule for duration of leaves via imputation from Westat 2001 survey probabilities
  # Then, flag those who will have exhausted employer benefits with leave remaining, and will apply 
  # to leave program for remainder of their leave
  # INPUT: ACS file
  d_acs_imp <- PAY_SCHEDULE(d_acs_imp)
  # OUTPUT: ACS file with imputed pay schedule, and date of benefit exhaustion for those with partial pay
  
  # Program eligibility and uptake functions
  # INPUT: ACS file
  d_acs_imp <-ELIGIBILITYRULES(d_acs_imp, earnings, weeks, ann_hours, minsize, base_bene_level, week_bene_min,
                               formula_prop_cuts, formula_value_cuts, formula_bene_levels, elig_rule_logic,
                               FEDGOV, STATEGOV, LOCALGOV, SELFEMP,PRIVATE,exclusive_particip) 
  # OUTPUT: ACS file with program eligibility and base program take-up indicators
  
  # Option to extend leaves under leave program 
    # INPUT: ACS file
    d_acs_imp <- EXTENDLEAVES(d_fmla, d_acs_imp, wait_period, ext_base_effect, 
                              extend_prob, extend_days, extend_prop, fmla_protect)  
    # OUTPUT: ACS file with leaves extended based on user specifications

  # INPUT: ACS file
  d_acs_imp <-UPTAKE(d_acs_imp, own_uptake, matdis_uptake, bond_uptake, illparent_uptake, 
                     illspouse_uptake, illchild_uptake, full_particip_needer, wait_period,
                     maxlen_own, maxlen_matdis, maxlen_bond, maxlen_illparent, maxlen_illspouse, maxlen_illchild,
                     maxlen_total,maxlen_DI,maxlen_PFL,exclusive_particip)
  # OUTPUT: ACS file with modified leave program variables based on user-specified program restrictions
  #         on maximum participation length and user-specified take-up rate assumptions
  
  
  # benefit parameter functions
  # INPUT: ACS file
  d_acs_imp <- BENEFITS(d_acs_imp)
  # OUTPUT: ACS file with base employer pay and program benefit calculation variables
  
  if (bene_effect==TRUE) {
    # INPUT: ACS file
    d_acs_imp <- BENEFITEFFECT(d_acs_imp)
    # OUTPUT: ACS file with leave taking variables modified to account for behavioral cost of applying to program
  }
  
  if (topoff_rate>0) {
    # INPUT: ACS file
    d_acs_imp <- TOPOFF(d_acs_imp,topoff_rate, topoff_minlength)
    # OUTPUT: ACS file with leave taking variables modified to account for employer top-off effects
  }
  
  if (dependent_allow>0) {
    # INPUT: ACS file
    d_acs_imp <- DEPENDENTALLOWANCE(d_acs_imp,dependent_allow)
    # OUTPUT: ACS file with program benefit amounts including a user-specified weekly dependent allowance
  }
  # Apply type-specific elig adjustments 
  d_acs_imp <- DIFF_ELIG(d_acs_imp, own_elig_adj, illspouse_elig_adj, illchild_elig_adj,
                         illparent_elig_adj, matdis_elig_adj, bond_elig_adj)
  
  # final clean up 
  # INPUT: ACS file
  d_acs_imp <- CLEANUP(d_acs_imp, week_bene_cap,week_bene_cap_prop,week_bene_min, maxlen_own, maxlen_matdis, maxlen_bond, 
                       maxlen_illparent, maxlen_illspouse, maxlen_illchild, maxlen_total,maxlen_DI,maxlen_PFL)
  # OUTPUT: ACS file with finalized leave taking, program uptake, and benefits received variables
  
  
  # Options to output final data and summary statistics
  
  if (!is.null(output) & saveCSV==TRUE) {
    write.csv(d_acs_imp, file=paste0('./output/',output,'.csv'))
  }
  
  for (i in output_stats) {
    if (i=='standard') {
      standard_summary_stats(d_acs_imp,output) 
    }
    
    if (i=='state_compar') {
      state_compar_stats(d_acs_imp, output)
    }
    
    if (i=='take_compar') {
      take_compar(d_acs_imp, output)
    }  
  }
  
  
  return(d_acs_imp)
}

