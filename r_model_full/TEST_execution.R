
# sample master execution function for testing code
source("0_master_execution_function.R")
options(error=recover)
meth <- 'Logistic Regression GLM'
timestart <<- Sys.time()
d <- policy_simulation(
                  base_bene_level=.6,
                  impute_method = meth,
                  acs_year=2016,
		              makelog = FALSE,
                  sample_prop=1,
		              state='RI',
		              SELFEMP=FALSE,
		              place_of_work = TRUE,
		              dual_receiver = 1,
		              alpha=0,
                  ext_resp_len = TRUE, sens_var = 'resp_len', progalt_post_or_pre ='post',
		              ext_base_effect=TRUE,
                  bene_effect=FALSE, full_particip=FALSE, wait_period=5, clone_factor=1, week_bene_cap=804,
                  own_uptake=.0809, matdis_uptake=.027, bond_uptake=.0102, illparent_uptake=.0009,
                  illspouse_uptake=.0014, illchild_uptake=.0006,
                  maxlen_own =150, maxlen_matdis =150, maxlen_bond =30, maxlen_illparent =30, maxlen_illspouse=30, maxlen_illchild=30,
                  maxlen_PFL= 150, maxlen_DI=150, maxlen_total=150,earnings=3840, dependent_allow = c(.07,.07,.07,.07,.07), random_seed=12345)

timeend <<- Sys.time()
print(timeend - timestart)