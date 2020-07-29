
source("0_master_execution_function.R")
#=================================
#Rhode Island
#=================================
meth <- 'Logistic Regression GLM'
timestart <<- Sys.time()
ri <- policy_simulation(saveCSV=TRUE,
                        impute_method=meth,
                        minsize=1,
                        ann_hours=1,
                        state='RI',
                        fmla_year=2012,
                        acs_year=2016,
                        makelog=TRUE,
                        base_bene_level=.6,
                        place_of_work = TRUE,
                        dual_receiver = 1,
                        ext_resp_len = TRUE, 
                        fmla_protect=FALSE,
                        ext_base_effect=FALSE,
                        rr_sensitive_leave_len=TRUE,
                        topoff_rate=0, topoff_minlength=0, 
                        bene_effect=FALSE, wait_period=5, clone_factor=1, week_bene_cap=804, week_bene_min=0,
                        dependent_allow = c(.07,.07,.07,.07,.07),
                        alpha=0,
                        sens_var='resp_len',
                        own_uptake= .0809, matdis_uptake=.027, bond_uptake=.0102, illchild_uptake=.0006,
                        illspouse_uptake=.0015, illparent_uptake=.0009,
                        maxlen_own =150, maxlen_matdis =150, maxlen_bond =20, maxlen_illparent=20, 
                        maxlen_illspouse =20, maxlen_illchild =20,
                        earnings=3840,  random_seed=12312)
timeend <<- Sys.time()
print(timeend - timestart)
