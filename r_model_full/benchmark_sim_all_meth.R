# benchmark_sim_all_meth.R
# PURPOSE: code for issue brief benchmarking model results against ACM model and actual data in CA/NJ/RI
# this file runs all available methods on all states.

# load master execution function - this file should be placed in the same folder as this function.
  source("0_master_execution_function.R")
  
# list of methods to test
  meths <- c("Logistic Regression GLM",'Logistic Regression Regularized','K Nearest Neighbor',"Naive Bayes","Ridge Classifier","Random Forest","Support Vector Machine","XGBoost")
  
  for (meth in meths) {
    print(meth)
    #=================================
    #Rhode Island
    #=================================
    ri <- policy_simulation(
                              # Policy simulation parameters
                              state='RI',
                              base_bene_level=.6,
                              impute_method=meth,
                              place_of_work = TRUE,
                              dual_receiver = 1,
                              fmla_protect=FALSE,
                              ext_base_effect=TRUE,
                              rr_sensitive_leave_len=FALSE,
                              topoff_rate=0, 
                              topoff_minlength=20, 
                              bene_effect=FALSE, 
                              wait_period=5, 
                              week_bene_cap=795, 
                              week_bene_min=89,
                              alpha=0,
                              own_uptake= .0823, 
                              matdis_uptake=.0274, 
                              bond_uptake=.0104, 
                              illchild_uptake=.0006,
                              illspouse_uptake=.0016, 
                              illparent_uptake=.0009, 
                              dependent_allow = c(.07,.07,.07,.07,.07),
                              maxlen_PFL= 20, 
                              maxlen_DI=150, 
                              maxlen_own =150, 
                              maxlen_matdis =150, 
                              maxlen_bond =20, 
                              maxlen_illparent=20, 
                              maxlen_illspouse =20, 
                              maxlen_illchild =20, 
                              maxlen_total=150, 
                              earnings=3840, 
                              random_seed=12312,
                              # output parameters - has no effect on simulation, just how output is created
                              saveCSV=TRUE, 
                              output=paste0("RI_",meth,"_benchmark_11052020"),
                              addl_vars=c('DI_plen'), 
                              makelog=TRUE)
    
    
    #=================================
    # New Jersey
    #=================================
    
    timestart <<- Sys.time()
    nj <- policy_simulation(
                      # Policy simulation parameters
                      state='NJ',
                      base_bene_level=.66,
                      impute_method=meth,
                      place_of_work = TRUE,
                      dual_receiver = 1,
                      ext_resp_len = TRUE,
                      ext_base_effect=TRUE,
                      fmla_protect=FALSE,
                      topoff_rate=0, 
                      topoff_minlength=40, 
                      rr_sensitive_leave_len=FALSE,
                      bene_effect=FALSE, 
                      wait_period=5, 
                      week_bene_cap=594, 
                      week_bene_min=0,
                      alpha=0,
                      own_uptake=.0250, 
                      matdis_uptake=.0088, 
                      bond_uptake=.0092, 
                      illchild_uptake=.0004, 
                      illspouse_uptake=.0005, 
                      illparent_uptake=.0007,
                      maxlen_PFL= 30, 
                      maxlen_DI=130, 
                      maxlen_own =130, 
                      maxlen_matdis =130, 
                      maxlen_bond =30, 
                      maxlen_illparent=30,
                      maxlen_illspouse =30, 
                      maxlen_illchild =30, 
                      maxlen_total=130, 
                      earnings=8400,  
                      random_seed=12312,
                      # output parameters - has no effect on simulation, just how output is created
                      saveCSV=TRUE,
                      makelog=TRUE,
                      output=paste0("NJ_",meth,"_benchmark_11052020"),
                      addl_vars=c('DI_plen'))
    
    #=================================
    # California
    #=================================
    ca <- policy_simulation(
                            # Policy simulation parameters
                            state='CA',
                            FEDGOV=TRUE,
                            STATEGOV=TRUE,
                            LOCALGOV=TRUE,
                            base_bene_level=.55,
                            impute_method=meth,
                            place_of_work = TRUE,
                            dual_receiver = 1,
                            ext_resp_len = TRUE,
                            fmla_protect=FALSE,
                            ext_base_effect=TRUE,
                            rr_sensitive_leave_len=FALSE,
                            bene_effect=FALSE, 
                            wait_period=5, 
                            week_bene_cap=1216, 
                            week_bene_min=50,
                            alpha=.75,
                            topoff_rate=0, 
                            topoff_minlength=20, 
                            own_uptake=.0308, 
                            matdis_uptake=.0108, 
                            bond_uptake=.0130, 
                            illparent_uptake=.0006,
                            illspouse_uptake=.0006, 
                            illchild_uptake=.0004,
                            maxlen_PFL= 30, 
                            maxlen_DI=260, 
                            maxlen_own =260, 
                            maxlen_matdis =260, 
                            maxlen_bond =30, 
                            maxlen_illparent=30,
                            maxlen_illspouse =30, 
                            maxlen_illchild =30, 
                            maxlen_total=260, 
                            earnings=300,
                            random_seed=12312, 
                            # output parameters - has no effect on simulation, just how output is created
                            output=paste0("CA_",meth,"_benchmark_11052020"),
                            saveCSV=TRUE,
                            makelog=TRUE,
                            addl_vars=c('DI_plen'))
  }
