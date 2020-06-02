/*Read in EE dataset, assign formats*/

libname in "Insert filepath here";

options fmtsearch=(in.fmla_2018_employee_formats);

data outdata;
set in.FMLA_2018_employee_PUF;
	format LEAVE_CAT LEAVE_CATf. NUM_JOBS NUM_JOBSf. AGE_CAT AGE_CATf. GENDER_CAT GENDER_CATf. SURVEY_TYPE SURVEY_TYPEf. GOVT_EMP GOVT_EMPf. PAID_LEAVE_STATE PAID_LEAVE_STATEf.
			LOW_WAGE LOW_WAGEf. FMLA_ELIG_if15hrwk FMLA_ELIG_if30hrwk FMLA_ELIG_if30emp FMLA_Elig_if20emp FMLA_ELIGIBLE FMLA_ELIGIBLEf. A19_MR_0_10d A19_MR_11_40d A19_MR_41_60d A19_MR_61Pd YN2f. EDUC_CAT EDUC_CATf. D3abc_CAT D3abc_CATf. nD4_CAT nD4_CATf. 
			A1 A3 A5_OwnIllness A5_NewChild A5_CHC A5_AdultHealthLT65 A5_AdultHealthGT65 A5_nonFMLAPerson A19b_MR nA20 A20a A26 A28 A35 A43 a43h: A52a A52b A52c A52d nA52e nA52f A53A A53b A53c A53d A53e A53f A53G A53h A55 A60 nA62a nA62b nA62c nA62d nA62e 
			nA62f nA62g nA62h nA62i B1 B16 B17 b20a b20b b20c b20d b20e b20f b20g C1 E1a E1c   
			nE6 E10  B15a B15b B15c B15d B15e B15f B15g B15h B15i B15j B15k B15l B15m B15n B15o B15p B15q 
			E3_media E3_cowrk E3_HR E3_poster E3_FamMem E3_friend E3_union E3_other A60 YNsf. 
			E2 ne5 D3 D5 D11 YNaf.
			A4_CAT A4_CATf. A5_MR_CAT A5_MR_CATf. A5_Long_CAT A5_Long_CATf. A6_MR_CAT A6_MR_CATf. A8_MR_CAT A8_MR_CATf. A8_Long_CAT A8_Long_CATf. A10_MR A10_MRf. A10_Long_CAT A10_Long_CATf. A14_MR A14_Long A14f.
			A15_MR_CAT A15_MR_CATf. A15_Long_CAT A15_Long_CATf. A19_MR_CAT A19_MR_CATf. A19_Long_CAT A19_Long_CATf. A19c_MR_CAT A19c_MR_CATf. 
			A19e_MR_CAT A19e_MR_CATf. nA19d_MR nA19d_MRf. RACE_CAT RACE_CATf. ELIGIBILITY_CAT ELIGIBILITY_CATf.
			A23a A23b nA23c nA23d A23e nA23f a23f. a30 a30f. A33 a33f. A41_CAT A41_CATf. A42_CAT A42_CATf. A43a A43af. A43b_CAT A43b_CATf. A43c A43cf. A43d_CAT A43d_CATf. A43f_CAT A43f_CATf. A43g_CAT  A43g_CATf.  
			A43i_a_CAT A43i_a_CATf. A43i_b_CAT A43i_b_CATf. A43i_c_CAT A43i_c_CATf. A43i_d_CAT A43i_d_CATf. A43i_e_CAT A43i_e_CATf. A43i_f_CAT A43i_f_CATf.
			A44 A44f. nA54 nA54f. A59 A59f. A63 A63f. A64 A64f. B2 B2f. B4_CAT B4_CATf. B6_CAT B6_CATf. B7_CAT B7_CATf. B9_CAT B9_CATf. nB11 nB11f. E0c_CAT E0c_CATf. E1b E1bf.
			E0a_CAT E0a_CATf. E0f_CAT E0f_CATf. E0i_CAT E0i_CATf. E0b_CAT E0b_CATf. E0g_CAT E0g_CATf. E0j_CAT E0j_CATf. E0k_CAT E0k_CATf.
			E1d E1df. E4a_a E4a_b E4a_c E4a_d nE4a_e nE4a_f nE4a_g nE4a_h nE4a_i nE4a_j nE4a_k nE4a_l E4f. ne8a ne8b ne8c ne8d ne8e nE8f. E9_CAT E9_CATf. nE15_coded $nE15_codedf. ne16_coded ne16_codedf.
			D7_CAT D7_CATf. D8_CAT D8_CATf. nD9 nD9f. D10 D10f.
	;
run;
