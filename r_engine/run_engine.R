source("0_master_execution_function.R")
args = commandArgs(trailingOnly=TRUE)

if (length(args) != 59) {
  stop("Incorrect number of arguments supplied", call.=FALSE)
} else {
	d <- policy_simulation(
		saveCSV=TRUE,
		xvars=c("widowed", "divorced", "separated", "nevermarried", "female", 
			  'age_cat', "ltHS", "someCol", "BA", "GradSch", "black", 
			  "other", "asian",'native', "hisp","nochildren",'faminc_cat','coveligd'),
		base_bene_level=as.numeric(args[0]),
		impute_method=args[1],
		makelog = as.logical(args[2]),
		sample_prop=as.numeric(args[3]),
		state=args[4],
		FEDGOV=as.logical(args[5]),
		STATEGOV=as.logical(args[6]),
		LOCALGOV=as.logical(args[7]),
		SELFEMP=as.logical(args[8]),
		place_of_work=as.logical(args[9]),
		exclusive_particip=as.logical(args[10]),
		SMOTE=as.logical(args[11]),
		ext_resp_len=as.logical(args[12]),
		len_method=args[13],
		sens_var=args[14],
		progalt_post_or_pre=args[15],
		intra_impute=as.logical(args[16]),
		exclusive_particip=as.logical(args[17]),
		ext_base_effect=as.logical(args[18]),
		extend_prob=as.numeric(args[19]),
		extend_days=as.numeric(args[20]),
		extend_prop=as.numeric(args[21]),
		topoff_rate=as.numeric(args[22]),
		topoff_minlength=as.numeric(args[23]),
		bene_effect=as.logical(args[24]),
		dependent_allow=as.numeric(args[25]),
		full_particip_needer=as.logical(args[26]),
		wait_period=as.numeric(args[27]),
		clone_factor=as.numeric(args[28]),
		week_bene_cap=as.numeric(args[29]),
		week_bene_min=as.numeric(args[30]),
		own_uptake=as.numeric(args[31]),
		matdis_uptake=as.numeric(args[32]),
		bond_uptake=as.numeric(args[33]),
		illparent_uptake=as.numeric(args[34]),
		illspouse_uptake=as.numeric(args[35]),
		illchild_uptake=as.numeric(args[36]),
		maxlen_own=as.numeric(args[37]),
		maxlen_matdis=as.numeric(args[38]),
		maxlen_bond=as.numeric(args[39]),
		maxlen_illparent=as.numeric(args[40]),
		maxlen_illspouse=as.numeric(args[41]),
		maxlen_illchild=as.numeric(args[42]),
		maxlen_total=as.numeric(args[43]),
		maxlen_PFL=as.numeric(args[44]),
		maxlen_DI=as.numeric(args[45]),
		earnings=as.numeric(args[46]),
        weeks=as.numeric(args[47]),
        ann_hours=as.numeric(args[48]),
        minsize=as.numeric(args[49]),
		own_elig_adj=as.numeric(args[50]),
		matdis_elig_adj=as.numeric(args[51]),
		bond_elig_adj=as.numeric(args[52]),
		illspouse_elig_adj=as.numeric(args[53]),
        illparent_elig_adj=as.numeric(args[54]),
		illchild_elig_adj=as.numeric(args[55]),
		output=args[56],
		random_seed=as.numeric(args[57]),
        progress_file=args[58])
}