from cython_gsl cimport *

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from libc.math cimport copysign
from libc.math cimport fmod
from libc.math cimport round
from libc.math cimport floor
from libc.math cimport int
from libc.math cimport fabs


from copy import deepcopy

#from cython.view cimport array as cvarray # Access to arrays

import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ----------- INITIALISATION ----------- #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# --------------------------------------------------------------------------- #
# ----------- Numpy array ----------- #
# --------------------------------------------------------------------------- #

cdef extern from "numpy/arrayobject.h": # To construct numpy array
    ctypedef int intp
    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef intp *dimensions
        cdef intp *strides
        cdef int flags

# --------------------------------------------------------------------------- #
# ----------- Initialisation variables ----------- #
# --------------------------------------------------------------------------- #

cdef extern from "init_variables.h":

    cdef int h_caL13 "_ind_h_caL13"
    cdef int m_caL13 "_ind_m_caL13"
    cdef int o_AMPA "_ind_o_AMPA"
    cdef int d_AMPA "_ind_d_AMPA"
    cdef int o_NMDA "_ind_o_NMDA"
    cdef int h_CICR "_ind_h_CICR"
    cdef int Ca_cyt "_ind_Ca_cyt"
    cdef int Ca_ER "_ind_Ca_ER"
    cdef int IP3 "_ind_IP3"
    cdef int DAG "_ind_DAG"
    cdef int DAGLP "_ind_DAGLP"
    cdef int twoAG "_ind_twoAG"
    cdef int AEA "_ind_AEA"
    cdef int fpre "_ind_fpre"
    cdef int I1P "_ind_I1P"
    cdef int PP1 "_ind_PP1"
    cdef int V "_ind_V"
    cdef int o_CB1R "_ind_o_CB1R"
    cdef int d_CB1R "_ind_d_CB1R"
    cdef int B1 "_ind_B1"
    cdef int B2 "_ind_B2"
    cdef int B3 "_ind_B3"
    cdef int B4 "_ind_B4"
    cdef int B5 "_ind_B5"
    cdef int B6 "_ind_B6"
    cdef int B7 "_ind_B7"
    cdef int B8 "_ind_B8"
    cdef int B9 "_ind_B9"
    cdef int B10 "_ind_B10"
    cdef int B11 "_ind_B11"
    cdef int B12 "_ind_B12"
    cdef int B13 "_ind_B13"
    cdef int Ca "_ind_Ca"
    cdef int pmca "_ind_pmca"
    cdef int pmcaCa "_ind_pmcaCa"
    cdef int CaOut "_ind_CaOut"
    cdef int ncx "_ind_ncx"
    cdef int ncxCa "_ind_ncxCa"
    cdef int Leak "_ind_Leak"
    cdef int CaOutLeak "_ind_CaOutLeak"
    cdef int Calbin "_ind_Calbin"
    cdef int CalbinC "_ind_CalbinC"
    cdef int CamCa4 "_ind_CamCa4"
    cdef int CamCa2N "_ind_CamCa2N"
    cdef int DaDbuf "_ind_DaDbuf"
    cdef int Dbuf "_ind_Dbuf"
    cdef int DaOut "_ind_DaOut"
    cdef int Da "_ind_Da"
    cdef int DaD1R "_ind_DaD1R"
    cdef int Gsabg "_ind_Gsabg"
    cdef int DaD1RGs "_ind_DaD1RGs"
    cdef int D1R "_ind_D1R"
    cdef int GsD1R "_ind_GsD1R"
    cdef int GsaGTP "_ind_GsaGTP"
    cdef int Gbg "_ind_Gbg"
    cdef int PKAc "_ind_PKAc"
    cdef int PKAcDaD1RGs "_ind_PKAcDaD1RGs"
    cdef int pDaD1RGs "_ind_pDaD1RGs"
    cdef int GluGbuf "_ind_GluGbuf"
    cdef int Gbuf "_ind_Gbuf"
    cdef int Glu_Blackwell "_ind_Glu"
    cdef int GluOut "_ind_GluOut"
    cdef int GluMgluR "_ind_GluMgluR"
    cdef int GluMgluRdesens "_ind_GluMgluRdesens"
    cdef int MgluR "_ind_MgluR"
    cdef int Gqabg "_ind_Gqabg"
    cdef int MgluRGq "_ind_MgluRGq"
    cdef int GluMgluRGq "_ind_GluMgluRGq"
    cdef int GqaGTP "_ind_GqaGTP"
    cdef int ACh "_ind_ACh"
    cdef int m1R "_ind_m1R"
    cdef int AChm1R "_ind_AChm1R"
    cdef int AChm1RGq "_ind_AChm1RGq"
    cdef int m1RGq "_ind_m1RGq"
    cdef int m4R "_ind_m4R"
    cdef int AChm4R "_ind_AChm4R"
    cdef int Giabg "_ind_Giabg"
    cdef int AChm4RGi "_ind_AChm4RGi"
    cdef int Gim4R "_ind_Gim4R"
    cdef int GsaGDP "_ind_GsaGDP"
    cdef int GqaGDP "_ind_GqaGDP"
    cdef int GiaGTP "_ind_GiaGTP"
    cdef int GiaGDP "_ind_GiaGDP"
    cdef int AC5 "_ind_AC5"
    cdef int AC5Gsa "_ind_AC5Gsa"
    cdef int ATP "_ind_ATP"
    cdef int AC5GsaATP "_ind_AC5GsaATP"
    cdef int cAMP "_ind_cAMP"
    cdef int AC5Gia "_ind_AC5Gia"
    cdef int AC5GsaGia "_ind_AC5GsaGia"
    cdef int AC5GsaGiaATP "_ind_AC5GsaGiaATP"
    cdef int PDE1 "_ind_PDE1"
    cdef int PDE1CamCa4 "_ind_PDE1CamCa4"
    cdef int PDE1CamCa4cAMP "_ind_PDE1CamCa4cAMP"
    cdef int AMP "_ind_AMP"
    cdef int PDE10 "_ind_PDE10"
    cdef int PDE10cAMP "_ind_PDE10cAMP"
    cdef int pPDE10 "_ind_pPDE10"
    cdef int pPDE10cAMP "_ind_pPDE10cAMP"
    cdef int PKAcPDE10 "_ind_PKAcPDE10"
    cdef int PKAcPDE10cAMP "_ind_PKAcPDE10cAMP"
    cdef int PP1_Blackwell "_ind_PP1_Blackwell"
    cdef int pPDE10PP1 "_ind_pPDE10PP1"
    cdef int PDE2 "_ind_PDE2"
    cdef int PDE2cAMP "_ind_PDE2cAMP"
    cdef int PDE2cAMP2 "_ind_PDE2cAMP2"
    cdef int PDE4 "_ind_PDE4"
    cdef int PDE4cAMP "_ind_PDE4cAMP"
    cdef int pPDE4 "_ind_pPDE4"
    cdef int pPDE4cAMP "_ind_pPDE4cAMP"
    cdef int PKAcPDE4 "_ind_PKAcPDE4"
    cdef int PKAcPDE4cAMP "_ind_PKAcPDE4cAMP"
    cdef int pPDE4PP1 "_ind_pPDE4PP1"
    cdef int Plc "_ind_Plc"
    cdef int PlcCa "_ind_PlcCa"
    cdef int PlcCaGqa "_ind_PlcCaGqa"
    cdef int PlcGqa "_ind_PlcGqa"
    cdef int Pip2 "_ind_Pip2"
    cdef int PlcCaPip2 "_ind_PlcCaPip2"
    cdef int Ip3 "_ind_Ip3"
    cdef int Dag "_ind_Dag"
    cdef int PlcCaGqaPip2 "_ind_PlcCaGqaPip2"
    cdef int Dgl "_ind_Dgl"
    cdef int CaDgl "_ind_CaDgl"
    cdef int DagCaDgl "_ind_DagCaDgl"
    cdef int two_ag "_ind_two_ag"
    cdef int Ip3degrad "_ind_Ip3degrad"
    cdef int PIkinase "_ind_PIkinase"
    cdef int Ip3degPIk "_ind_Ip3degPIk"
    cdef int two_agDegrad "_ind_two_agDegrad"
    cdef int DagK "_ind_DagK"
    cdef int DagKdag "_ind_DagKdag"
    cdef int PA "_ind_PA"
    cdef int Pkc "_ind_Pkc"
    cdef int PkcCa "_ind_PkcCa"
    cdef int PkcCaDag "_ind_PkcCaDag"
    cdef int PKA "_ind_PKA"
    cdef int PKAcAMP2 "_ind_PKAcAMP2"
    cdef int PKAcAMP4 "_ind_PKAcAMP4"
    cdef int PKAr "_ind_PKAr"
    cdef int Cam "_ind_Cam"
    cdef int PP2B "_ind_PP2B"
    cdef int PP2BCam "_ind_PP2BCam"
    cdef int CamCa2C "_ind_CamCa2C"
    cdef int PP2BCamCa2C "_ind_PP2BCamCa2C"
    cdef int PP2BCamCa2N "_ind_PP2BCamCa2N"
    cdef int PP2BCamCa4 "_ind_PP2BCamCa4"
    cdef int PKAcPP2AB56d "_ind_PKAcPP2AB56d"
    cdef int pPP2A "_ind_pPP2A"
    cdef int PP2ABPR72 "_ind_PP2ABPR72"
    cdef int PP2Acal "_ind_PP2Acal"
    cdef int D32 "_ind_D32"
    cdef int D32PKAc "_ind_D32PKAc"
    cdef int D32p34 "_ind_D32p34"
    cdef int D32p34PP1 "_ind_D32p34PP1"
    cdef int D32p34PP2BCamCa4 "_ind_D32p34PP2BCamCa4"
    cdef int D32p34PP1PP2BCamCa4 "_ind_D32p34PP1PP2BCamCa4"
    cdef int D32p34PP2ABPR72 "_ind_D32p34PP2ABPR72"
    cdef int D32p34PP2AB56d "_ind_D32p34PP2AB56d"
    cdef int D32p34PP1PP2ABPR72 "_ind_D32p34PP1PP2ABPR72"
    cdef int D32p34PP1PP2AB56d "_ind_D32p34PP1PP2AB56d"
    cdef int PP2AB56d "_ind_PP2AB56d"
    cdef int Cdk5 "_ind_Cdk5"
    cdef int Cdk5D32 "_ind_Cdk5D32"
    cdef int D32p75 "_ind_D32p75"
    cdef int D32p75PKAc "_ind_D32p75PKAc"
    cdef int D32p75pPP2A "_ind_D32p75pPP2A"
    cdef int D32p75PP2ABPR72 "_ind_D32p75PP2ABPR72"
    cdef int D32p75PP2AB56d "_ind_D32p75PP2AB56d"
    cdef int D32p75PP2Acal "_ind_D32p75PP2Acal"
    cdef int CK "_ind_CK"
    cdef int CKCamCa4 "_ind_CKCamCa4"
    cdef int CKpCamCa4 "_ind_CKpCamCa4"
    cdef int CKp "_ind_CKp"
    cdef int CKpPP1 "_ind_CKpPP1"
    cdef int CKpCamCa4PP1 "_ind_CKpCamCa4PP1"
    cdef int Dgl_CKCamCa4 "_ind_Dgl_CKCamCa4"
    cdef int pDgl "_ind_pDgl"
    cdef int Dgl_CKpCamCa4 "_ind_Dgl_CKpCamCa4"
    cdef int CaDgl_CKpCamCa4 "_ind_CaDgl_CKpCamCa4"
    cdef int CapDgl "_ind_CapDgl"
    cdef int Dgl_CKp "_ind_Dgl_CKp"
    cdef int DagCapDgl "_ind_DagCapDgl"
    cdef int AKAR3 "_ind_AKAR3"
    cdef int PKAcAKAR3 "_ind_PKAcAKAR3"
    cdef int pAKAR3 "_ind_pAKAR3"
    cdef int PP1pAKAR3 "_ind_PP1pAKAR3"
    cdef int AC1 "_ind_AC1"
    cdef int AC1Gsa "_ind_AC1Gsa"
    cdef int AC1GsaCamCa4 "_ind_AC1GsaCamCa4"
    cdef int AC1GsaCamCa4ATP "_ind_AC1GsaCamCa4ATP"
    cdef int AC1CamCa4 "_ind_AC1CamCa4"
    cdef int AC1CamCa4ATP "_ind_AC1CamCa4ATP"

# --------------------------------------------------------------------------- #
# ----------- Parameters ----------- #
# --------------------------------------------------------------------------- #

cdef extern from "variables.h":
    cdef int ind_on_caL13 "_ind_on_caL13"
    cdef int ind_pcaLbar "_ind_pcaLbar"
    cdef int ind_mslope_caL13 "_ind_mslope_caL13"
    cdef int ind_hshift_caL13 "_ind_hshift_caL13"
    cdef int ind_vm_caL13 "_ind_vm_caL13"
    cdef int ind_mshift_caL13 "_ind_mshift_caL13"
    cdef int ind_hslope_caL13 "_ind_hslope_caL13"
    cdef int ind_kpr_caL13 "_ind_kpr_caL13"
    cdef int ind_c_caL13 "_ind_c_caL13"
    cdef int ind_k_caL13 "_ind_k_caL13"
    cdef int ind_hvhalf_caL13 "_ind_hvhalf_caL13"
    cdef int ind_mvhalf_caL13 "_ind_mvhalf_caL13"
    cdef int ind_cpr_caL13 "_ind_cpr_caL13"
    cdef int ind_htau_caL13 "_ind_htau_caL13"
    cdef int ind_hqfact_caL13 "_ind_hqfact_caL13"
    cdef int ind_qfact_caL13 "_ind_qfact_caL13"
    cdef int ind_on_TRPV1 "_ind_on_TRPV1"
    cdef int ind_gTRPV1 "_ind_gTRPV1"
    cdef int ind_p_ca_TRPV1 "_ind_p_ca_TRPV1"
    cdef int ind_C_TRPV1 "_ind_C_TRPV1"
    cdef int ind_D_TRPV1 "_ind_D_TRPV1"
    cdef int ind_DH_TRPV1 "_ind_DH_TRPV1"
    cdef int ind_KD_TRPV1 "_ind_KD_TRPV1"
    cdef int ind_J0_TRPV1 "_ind_J0_TRPV1"
    cdef int ind_L_TRPV1 "_ind_L_TRPV1"
    cdef int ind_P_TRPV1 "_ind_P_TRPV1"
    cdef int ind_z_TRPV1 "_ind_z_TRPV1"
    cdef int ind_DS_TRPV1 "_ind_DS_TRPV1"
    cdef int ind_K_TRPV1 "_ind_K_TRPV1"
    cdef int ind_NMDA "_ind_NMDA"
    cdef int ind_VDCC "_ind_VDCC"
    cdef int ind_TRPV1 "_ind_TRPV1"
    cdef int ind_BT_CaBuff "_ind_BT_CaBuff"
    cdef int ind_Cab "_ind_Cab"
    cdef int ind_tauCab "_ind_tauCab"
    cdef int ind_KdB_CaBuff "_ind_KdB_CaBuff"
    cdef int ind_a2_CICR "_ind_a2_CICR"
    cdef int ind_ver_CICR "_ind_ver_CICR"
    cdef int ind_d3_CICR "_ind_d3_CICR"
    cdef int ind_rc_CICR "_ind_rc_CICR"
    cdef int ind_rl_CICR "_ind_rl_CICR"
    cdef int ind_rhoER_CICR "_ind_rhoER_CICR"
    cdef int ind_d5_CICR "_ind_d5_CICR"
    cdef int ind_d2_CICR "_ind_d2_CICR"
    cdef int ind_ker_CICR "_ind_ker_CICR"
    cdef int ind_d1_CICR "_ind_d1_CICR"
    cdef int ind_kappad_IP3 "_ind_kappad_IP3"
    cdef int ind_kdelta_IP3 "_ind_kdelta_IP3"
    cdef int ind_r5p_IP3 "_ind_r5p_IP3"
    cdef int ind_v3k_IP3 "_ind_v3k_IP3"
    cdef int ind_kd_IP3 "_ind_kd_IP3"
    cdef int ind_vdelta_IP3 "_ind_vdelta_IP3"
    cdef int ind_k3_IP3 "_ind_k3_IP3"
    cdef int ind_kr_IP3 "_ind_kr_IP3"
    cdef int ind_kp_IP3 "_ind_kp_IP3"
    cdef int ind_vbeta_IP3 "_ind_vbeta_IP3"
    cdef int ind_kpi_IP3 "_ind_kpi_IP3"
    cdef int ind_n3_IP3 "_ind_n3_IP3"
    cdef int ind_KDGL "_ind_KDGL"
    cdef int ind_kMAGL "_ind_kMAGL"
    cdef int ind_rDGL "_ind_rDGL"
    cdef int ind_kDAGK "_ind_kDAGK"
    cdef int ind_nK_DAGLP "_ind_nK_DAGLP"
    cdef int ind_rP_DAGLP "_ind_rP_DAGLP"
    cdef int ind_rK_DAGLP "_ind_rK_DAGLP"
    cdef int ind_on_ECb "_ind_on_ECb"
    cdef int ind_ECb_CB1R_on "_ind_ECb_CB1R_on"
    cdef int ind_kCB1R "_ind_kCB1R"
    cdef int ind_alphaAEACB1 "_ind_alphaAEACB1"
    cdef int ind_P1_ECb "_ind_P1_ECb"
    cdef int ind_P2_ECb "_ind_P2_ECb"
    cdef int ind_P3_ECb "_ind_P3_ECb"
    cdef int ind_P4_ECb "_ind_P4_ECb"
    cdef int ind_vATAEA "_ind_vATAEA"
    cdef int ind_LTDstart "_ind_LTDstart"
    cdef int ind_LTDstop "_ind_LTDstop"
    cdef int ind_LTDMax "_ind_LTDMax"
    cdef int ind_LTPstart "_ind_LTPstart"
    cdef int ind_LTPMax "_ind_LTPMax"
    cdef int ind_KFAAH "_ind_KFAAH"
    cdef int ind_vFAAH "_ind_vFAAH"
    cdef int ind_on_ECb_smooth "_ind_on_ECb_smooth"
    cdef int ind_K_ECb_smooth "_ind_K_ECb_smooth"
    cdef int ind_n_ECb_smooth "_ind_n_ECb_smooth"
    cdef int ind_kw_ECb_smooth "_ind_kw_ECb_smooth"
    cdef int ind_tau_ECb_smooth "_ind_tau_ECb_smooth"
    cdef int ind_kadd_ECb_smooth "_ind_kadd_ECb_smooth"
    cdef int ind_kn_ECb_smooth "_ind_kn_ECb_smooth"
    cdef int ind_kpkaI1 "_ind_kpkaI1"
    cdef int ind_PP10 "_ind_PP10"
    cdef int ind_KM "_ind_KM"
    cdef int ind_K5 "_ind_K5"
    cdef int ind_k12 "_ind_k12"
    cdef int ind_k11 "_ind_k11"
    cdef int ind_kcan0I1 "_ind_kcan0I1"
    cdef int ind_km11 "_ind_km11"
    cdef int ind_CaMKT "_ind_CaMKT"
    cdef int ind_k7 "_ind_k7"
    cdef int ind_k6 "_ind_k6"
    cdef int ind_ncanI1 "_ind_ncanI1"
    cdef int ind_I10 "_ind_I10"
    cdef int ind_CaMT "_ind_CaMT"
    cdef int ind_KdpkaI1 "_ind_KdpkaI1"
    cdef int ind_kcanI1 "_ind_kcanI1"
    cdef int ind_KdcanI1 "_ind_KdcanI1"
    cdef int ind_kpka0I1 "_ind_kpka0I1"
    cdef int ind_Ka3 "_ind_Ka3"
    cdef int ind_Ka2 "_ind_Ka2"
    cdef int ind_Ka1 "_ind_Ka1"
    cdef int ind_Ka4 "_ind_Ka4"
    cdef int ind_npkaI1 "_ind_npkaI1"
    cdef int ind_conc_max_sub_unit "_ind_conc_max_sub_unit"
    cdef int ind_K9 "_ind_K9"
    cdef int ind_gamma1DA "_ind_gamma1DA"
    cdef int ind_gamma2DA "_ind_gamma2DA"
    cdef int ind_DA "_ind_DA"
    cdef int ind_on_AMPA "_ind_on_AMPA"
    cdef int ind_gAMPA "_ind_gAMPA"
    cdef int ind_Epsilon_AMPA "_ind_Epsilon_AMPA"
    cdef int ind_Beta_AMPA "_ind_Beta_AMPA"
    cdef int ind_Alpha_AMPA "_ind_Alpha_AMPA"
    cdef int ind_Gamma_AMPA "_ind_Gamma_AMPA"
    cdef int ind_on_NMDA "_ind_on_NMDA"
    cdef int ind_gNMDA "_ind_gNMDA"
    cdef int ind_p_ca_NMDA "_ind_p_ca_NMDA"
    cdef int ind_Mg_NMDA "_ind_Mg_NMDA"
    cdef int ind_Alpha_NMDA "_ind_Alpha_NMDA"
    cdef int ind_Beta_NMDA "_ind_Beta_NMDA"
    cdef int ind_EL "_ind_EL"
    cdef int ind_gL "_ind_gL"
    cdef int ind_Cm "_ind_Cm"
    cdef int ind_on_CB1R "_ind_on_CB1R"
    cdef int ind_Alpha_CB1R "_ind_Alpha_CB1R"
    cdef int ind_Beta_CB1R "_ind_Beta_CB1R"
    cdef int ind_Gamma_CB1R "_ind_Gamma_CB1R"
    cdef int ind_Epsilon_CB1R "_ind_Epsilon_CB1R"
    cdef int ind_R "_ind_R"
    cdef int ind_RT "_ind_RT"
    cdef int ind_T "_ind_T"
    cdef int ind_F "_ind_F"
    cdef int ind_zS "_ind_zS"
    cdef int ind_Ca_out "_ind_Ca_out"
    cdef int ind_t_step "_ind_t_step"
    cdef int ind_t_start "_ind_t_start"
    cdef int ind_t_end "_ind_t_end"
    cdef int ind_ATOL "_ind_ATOL"
    cdef int ind_RTOL "_ind_RTOL"
    cdef int ind_MXSTEP "_ind_MXSTEP"
    cdef int ind_HMAX "_ind_HMAX"
    cdef int ind_APdur "_ind_APdur"
    cdef int ind_action_as_VDCC "_ind_action_as_VDCC"
    cdef int ind_DPmax "_ind_DPmax"
    cdef int ind_APmax "_ind_APmax"
    cdef int ind_tausbAP "_ind_tausbAP"
    cdef int ind_steadyrise_on "_ind_steadyrise_on"
    cdef int ind_Glumax "_ind_Glumax"
    cdef int ind_BaseLevel "_ind_BaseLevel"
    cdef int ind_tauGlu "_ind_tauGlu"
    cdef int ind_regular_on "_ind_regular_on"
    cdef int ind_tsdt "_ind_tsdt"
    cdef int ind_tpost "_ind_tpost"
    cdef int ind_post_on "_ind_post_on"
    cdef int ind_Delta_t_STDP "_ind_Delta_t_STDP"
    cdef int ind_num_stim "_ind_num_stim"
    cdef int ind_pre_on "_ind_pre_on"
    cdef int ind_Freq "_ind_Freq"
    cdef int ind_tables_step "_ind_tables_step"
    cdef int ind_betaAEACB1 "_ind_betaAEACB1"
    cdef int ind_alphatwoAGCB1 "_ind_alphatwoAGCB1"
    cdef int ind_alpha_DAGLP_Ca_cyt "_ind_alpha_DAGLP_Ca_cyt"
    cdef int ind_input_ampl_DA "_ind_input_ampl_DA"
    cdef int ind_t_stim_on_DA "_ind_t_stim_on_DA"
    cdef int ind_t_stim_off_DA "_ind_t_stim_off_DA"
    cdef int ind_interv_stim_ms_DA "_ind_interv_stim_ms_DA"
    cdef int ind_input_ampl_Ca "_ind_input_ampl_Ca"
    cdef int ind_t_stim_on_Ca "_ind_t_stim_on_Ca"
    cdef int ind_t_stim_off_Ca "_ind_t_stim_off_Ca"
    cdef int ind_interv_stim_ms_Ca "_ind_interv_stim_ms_Ca"

    # ---- Blackwell params model ---- #
    cdef int ind_forwardRate_Ca_pump1a '_ind_forwardRate_Ca_pump1a'
    cdef int ind_reverseRate_Ca_pump1a '_ind_reverseRate_Ca_pump1a'
    cdef int ind_forwardRate_Ca_pump1b '_ind_forwardRate_Ca_pump1b'
    cdef int ind_reverseRate_Ca_pump1b '_ind_reverseRate_Ca_pump1b'
    cdef int ind_forwardRate_Ca_pump2a '_ind_forwardRate_Ca_pump2a'
    cdef int ind_reverseRate_Ca_pump2a '_ind_reverseRate_Ca_pump2a'
    cdef int ind_forwardRate_Ca_pump2b '_ind_forwardRate_Ca_pump2b'
    cdef int ind_reverseRate_Ca_pump2b '_ind_reverseRate_Ca_pump2b'
    cdef int ind_forwardRate_Ca_leak '_ind_forwardRate_Ca_leak'
    cdef int ind_reverseRate_Ca_leak '_ind_reverseRate_Ca_leak'
    cdef int ind_forwardRate_Ca_leak_1 '_ind_forwardRate_Ca_leak_1'
    cdef int ind_reverseRate_Ca_leak_1 '_ind_reverseRate_Ca_leak_1'
    cdef int ind_forwardRate_Ca_buffer '_ind_forwardRate_Ca_buffer'
    cdef int ind_reverseRate_Ca_buffer '_ind_reverseRate_Ca_buffer'
    cdef int ind_forwardRate_CamC_bind '_ind_forwardRate_CamC_bind'
    cdef int ind_reverseRate_CamC_bind '_ind_reverseRate_CamC_bind'
    cdef int ind_forwardRate_CamCa2C_bind '_ind_forwardRate_CamCa2C_bind'
    cdef int ind_reverseRate_CamCa2C_bind '_ind_reverseRate_CamCa2C_bind'
    cdef int ind_forwardRate_CamN_bind '_ind_forwardRate_CamN_bind'
    cdef int ind_reverseRate_CamN_bind '_ind_reverseRate_CamN_bind'
    cdef int ind_forwardRate_CamCa2N_bind '_ind_forwardRate_CamCa2N_bind'
    cdef int ind_reverseRate_CamCa2N_bind '_ind_reverseRate_CamCa2N_bind'
    cdef int ind_forwardRate_DaDbuf__Da_plus_Dbuf '_ind_forwardRate_DaDbuf__Da_plus_Dbuf'
    cdef int ind_reverseRate_DaDbuf__Da_plus_Dbuf '_ind_reverseRate_DaDbuf__Da_plus_Dbuf'
    cdef int ind_forwardRate_Da__DaOut '_ind_forwardRate_Da__DaOut'
    cdef int ind_reverseRate_Da__DaOut '_ind_reverseRate_Da__DaOut'
    cdef int ind_forwardRate_DaOut_plus_Dbuf__NULL '_ind_forwardRate_DaOut_plus_Dbuf__NULL'
    cdef int ind_reverseRate_DaOut_plus_Dbuf__NULL '_ind_reverseRate_DaOut_plus_Dbuf__NULL'
    cdef int ind_forwardRate_Da_plus_D1R__DaD1R '_ind_forwardRate_Da_plus_D1R__DaD1R'
    cdef int ind_reverseRate_Da_plus_D1R__DaD1R '_ind_reverseRate_Da_plus_D1R__DaD1R'
    cdef int ind_forwardRate_DaD1R_plus_Gs__DaD1RGs '_ind_forwardRate_DaD1R_plus_Gs__DaD1RGs'
    cdef int ind_reverseRate_DaD1R_plus_Gs__DaD1RGs '_ind_reverseRate_DaD1R_plus_Gs__DaD1RGs'
    cdef int ind_forwardRate_Gsabg_plus_D1R__GsD1R '_ind_forwardRate_Gsabg_plus_D1R__GsD1R'
    cdef int ind_reverseRate_Gsabg_plus_D1R__GsD1R '_ind_reverseRate_Gsabg_plus_D1R__GsD1R'
    cdef int ind_forwardRate_GsD1R_plus_Da__DaD1RGs '_ind_forwardRate_GsD1R_plus_Da__DaD1RGs'
    cdef int ind_reverseRate_GsD1R_plus_Da__DaD1RGs '_ind_reverseRate_GsD1R_plus_Da__DaD1RGs'
    cdef int ind_forwardRate_DaD1RGs_DaD1R_plus_GsaGTP_plus_Gbg_reac '_ind_forwardRate_DaD1RGs_DaD1R_plus_GsaGTP_plus_Gbg_reac'
    cdef int ind_reverseRate_DaD1RGs_DaD1R_plus_GsaGTP_plus_Gbg_reac '_ind_reverseRate_DaD1RGs_DaD1R_plus_GsaGTP_plus_Gbg_reac'
    cdef int ind_forwardRate_DaD1RGs_phosp1 '_ind_forwardRate_DaD1RGs_phosp1'
    cdef int ind_reverseRate_DaD1RGs_phosp1 '_ind_reverseRate_DaD1RGs_phosp1'
    cdef int ind_forwardRate_DaD1RGs_phosp2 '_ind_forwardRate_DaD1RGs_phosp2'
    cdef int ind_reverseRate_DaD1RGs_phosp2 '_ind_reverseRate_DaD1RGs_phosp2'
    cdef int ind_forwardRate_dephosD1R '_ind_forwardRate_dephosD1R'
    cdef int ind_reverseRate_dephosD1R '_ind_reverseRate_dephosD1R'
    cdef int ind_forwardRate_Glubuf__Glu_plus_Gbuf '_ind_forwardRate_Glubuf__Glu_plus_Gbuf'
    cdef int ind_reverseRate_Glubuf__Glu_plus_Gbuf '_ind_reverseRate_Glubuf__Glu_plus_Gbuf'
    cdef int ind_forwardRate_Glu__GluOut '_ind_forwardRate_Glu__GluOut'
    cdef int ind_reverseRate_Glu__GluOut '_ind_reverseRate_Glu__GluOut'
    cdef int ind_forwardRate_GluOut_buf_reac '_ind_forwardRate_GluOut_buf_reac'
    cdef int ind_reverseRate_GluOut_buf_reac '_ind_reverseRate_GluOut_buf_reac'
    cdef int ind_forwardRate_Glu_plus_MgluR__GluMgluRdesens '_ind_forwardRate_Glu_plus_MgluR__GluMgluRdesens'
    cdef int ind_reverseRate_Glu_plus_MgluR__GluMgluRdesens '_ind_reverseRate_Glu_plus_MgluR__GluMgluRdesens'
    cdef int ind_forwardRate_Glu_plus_MgluR__GluMgluR '_ind_forwardRate_Glu_plus_MgluR__GluMgluR'
    cdef int ind_reverseRate_Glu_plus_MgluR__GluMgluR '_ind_reverseRate_Glu_plus_MgluR__GluMgluR'
    cdef int ind_forwardRate_Gqabg_plus_GluMgluR__GluMgluRGq '_ind_forwardRate_Gqabg_plus_GluMgluR__GluMgluRGq'
    cdef int ind_reverseRate_Gqabg_plus_GluMgluR__GluMgluRGq '_ind_reverseRate_Gqabg_plus_GluMgluR__GluMgluRGq'
    cdef int ind_forwardRate_Gqabg_plus_MgluR___MgluRGq '_ind_forwardRate_Gqabg_plus_MgluR___MgluRGq'
    cdef int ind_reverseRate_Gqabg_plus_MgluR___MgluRGq '_ind_reverseRate_Gqabg_plus_MgluR___MgluRGq'
    cdef int ind_forwardRate_Glu_plus_MgluRGq___GluMgluRGq '_ind_forwardRate_Glu_plus_MgluRGq___GluMgluRGq'
    cdef int ind_reverseRate_Glu_plus_MgluRGq___GluMgluRGq '_ind_reverseRate_Glu_plus_MgluRGq___GluMgluRGq'
    cdef int ind_forwardRate_GluMgluRGq__GqaGTP_plus_GluMgluRGbg '_ind_forwardRate_GluMgluRGq__GqaGTP_plus_GluMgluRGbg'
    cdef int ind_reverseRate_GluMgluRGq__GqaGTP_plus_GluMgluRGbg '_ind_reverseRate_GluMgluRGq__GqaGTP_plus_GluMgluRGbg'
    cdef int ind_forwardRate_ACh_plus_m1R__AChm1R '_ind_forwardRate_ACh_plus_m1R__AChm1R'
    cdef int ind_reverseRate_ACh_plus_m1R__AChm1R '_ind_reverseRate_ACh_plus_m1R__AChm1R'
    cdef int ind_forwardRate_Gqabg_plus_AChm1R__AChm1RGq '_ind_forwardRate_Gqabg_plus_AChm1R__AChm1RGq'
    cdef int ind_reverseRate_Gqabg_plus_AChm1R__AChm1RGq '_ind_reverseRate_Gqabg_plus_AChm1R__AChm1RGq'
    cdef int ind_forwardRate_Gqabg_plus_m1R___m1RGq '_ind_forwardRate_Gqabg_plus_m1R___m1RGq'
    cdef int ind_reverseRate_Gqabg_plus_m1R___m1RGq '_ind_reverseRate_Gqabg_plus_m1R___m1RGq'
    cdef int ind_forwardRate_ACh_plus_m1RGq___AChm1RGq '_ind_forwardRate_ACh_plus_m1RGq___AChm1RGq'
    cdef int ind_reverseRate_ACh_plus_m1RGq___AChm1RGq '_ind_reverseRate_ACh_plus_m1RGq___AChm1RGq'
    cdef int ind_forwardRate_AChm1RGq__GqaGTP_plus_AChm1RGbg '_ind_forwardRate_AChm1RGq__GqaGTP_plus_AChm1RGbg'
    cdef int ind_reverseRate_AChm1RGq__GqaGTP_plus_AChm1RGbg '_ind_reverseRate_AChm1RGq__GqaGTP_plus_AChm1RGbg'
    cdef int ind_forwardRate_ACh__null '_ind_forwardRate_ACh__null'
    cdef int ind_forwardRate_ACh_plus_m4R__AChm4R '_ind_forwardRate_ACh_plus_m4R__AChm4R'
    cdef int ind_reverseRate_ACh_plus_m4R__AChm4R '_ind_reverseRate_ACh_plus_m4R__AChm4R'
    cdef int ind_forwardRate_Giabg_plus_AChm4R__AChm4RGi '_ind_forwardRate_Giabg_plus_AChm4R__AChm4RGi'
    cdef int ind_reverseRate_Giabg_plus_AChm4R__AChm4RGi '_ind_reverseRate_Giabg_plus_AChm4R__AChm4RGi'
    cdef int ind_forwardRate_Giabg_plus_m4R__Gim4R '_ind_forwardRate_Giabg_plus_m4R__Gim4R'
    cdef int ind_reverseRate_Giabg_plus_m4R__Gim4R '_ind_reverseRate_Giabg_plus_m4R__Gim4R'
    cdef int ind_forwardRate_Gim4R_plus_ACh__AChm4RGi '_ind_forwardRate_Gim4R_plus_ACh__AChm4RGi'
    cdef int ind_reverseRate_Gim4R_plus_ACh__AChm4RGi '_ind_reverseRate_Gim4R_plus_ACh__AChm4RGi'
    cdef int ind_forwardRate_AChm4RGi__GiaGTP_plus_AChm4R_plus_Gbg '_ind_forwardRate_AChm4RGi__GiaGTP_plus_AChm4R_plus_Gbg'
    cdef int ind_reverseRate_AChm4RGi__GiaGTP_plus_AChm4R_plus_Gbg '_ind_reverseRate_AChm4RGi__GiaGTP_plus_AChm4R_plus_Gbg'
    cdef int ind_forwardRate_GsaGTP__GsaGDP_reac '_ind_forwardRate_GsaGTP__GsaGDP_reac'
    cdef int ind_reverseRate_GsaGTP__GsaGDP_reac '_ind_reverseRate_GsaGTP__GsaGDP_reac'
    cdef int ind_forwardRate_GsaGDP_plus_Gbg__Gs_reac '_ind_forwardRate_GsaGDP_plus_Gbg__Gs_reac'
    cdef int ind_reverseRate_GsaGDP_plus_Gbg__Gs_reac '_ind_reverseRate_GsaGDP_plus_Gbg__Gs_reac'
    cdef int ind_forwardRate_GqaGTP__GqaGDP '_ind_forwardRate_GqaGTP__GqaGDP'
    cdef int ind_reverseRate_GqaGTP__GqaGDP '_ind_reverseRate_GqaGTP__GqaGDP'
    cdef int ind_forwardRate_GqaGDP_plus_Gbg__Gqabg '_ind_forwardRate_GqaGDP_plus_Gbg__Gqabg'
    cdef int ind_reverseRate_GqaGDP_plus_Gbg__Gqabg '_ind_reverseRate_GqaGDP_plus_Gbg__Gqabg'
    cdef int ind_forwardRate_GiaGTP__GiaGDP '_ind_forwardRate_GiaGTP__GiaGDP'
    cdef int ind_reverseRate_GiaGTP__GiaGDP '_ind_reverseRate_GiaGTP__GiaGDP'
    cdef int ind_forwardRate_GiaGDP__Giabg '_ind_forwardRate_GiaGDP__Giabg'
    cdef int ind_reverseRate_GiaGDP__Giabg '_ind_reverseRate_GiaGDP__Giabg'
    cdef int ind_forwardRate_GsaGTP_plus_AC5__AC5Gsa_reac '_ind_forwardRate_GsaGTP_plus_AC5__AC5Gsa_reac'
    cdef int ind_reverseRate_GsaGTP_plus_AC5__AC5Gsa_reac '_ind_reverseRate_GsaGTP_plus_AC5__AC5Gsa_reac'
    cdef int ind_forwardRate_AC5Gsa_plus_ATP__AC5GsaATP_reac '_ind_forwardRate_AC5Gsa_plus_ATP__AC5GsaATP_reac'
    cdef int ind_reverseRate_AC5Gsa_plus_ATP__AC5GsaATP_reac '_ind_reverseRate_AC5Gsa_plus_ATP__AC5GsaATP_reac'
    cdef int ind_forwardRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac '_ind_forwardRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac'
    cdef int ind_reverseRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac '_ind_reverseRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac'
    cdef int ind_forwardRate_GiaGTP_AC5_reac '_ind_forwardRate_GiaGTP_AC5_reac'
    cdef int ind_reverseRate_GiaGTP_AC5_reac '_ind_reverseRate_GiaGTP_AC5_reac'
    cdef int ind_forwardRate_GiaGTP_AC5GsaGTP_reac '_ind_forwardRate_GiaGTP_AC5GsaGTP_reac'
    cdef int ind_reverseRate_GiaGTP_AC5GsaGTP_reac '_ind_reverseRate_GiaGTP_AC5GsaGTP_reac'
    cdef int ind_forwardRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP '_ind_forwardRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP'
    cdef int ind_reverseRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP '_ind_reverseRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP'
    cdef int ind_forwardRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP '_ind_forwardRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP'
    cdef int ind_reverseRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP '_ind_reverseRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP'
    cdef int ind_forwardRate_AC5GsaGiaATP__AC5Gia_plus_Gs_plus_cAMP '_ind_forwardRate_AC5GsaGiaATP__AC5Gia_plus_Gs_plus_cAMP'
    cdef int ind_reverseRate_AC5GsaGiaATP__AC5Gia_plus_Gs_plus_cAMP '_ind_reverseRate_AC5GsaGiaATP__AC5Gia_plus_Gs_plus_cAMP'
    cdef int ind_forwardRate_PDE1_plus_CamCa4__PDE1CamCa4 '_ind_forwardRate_PDE1_plus_CamCa4__PDE1CamCa4'
    cdef int ind_reverseRate_PDE1_plus_CamCa4__PDE1CamCa4 '_ind_reverseRate_PDE1_plus_CamCa4__PDE1CamCa4'
    cdef int ind_forwardRate_PDE1CamCa4_plus_cAMP__PDE1CamCa4cAMP '_ind_forwardRate_PDE1CamCa4_plus_cAMP__PDE1CamCa4cAMP'
    cdef int ind_reverseRate_PDE1CamCa4_plus_cAMP__PDE1CamCa4cAMP '_ind_reverseRate_PDE1CamCa4_plus_cAMP__PDE1CamCa4cAMP'
    cdef int ind_forwardRate_PDE1CamCa4cAMP__PDE1CamCa4_plus_AMP '_ind_forwardRate_PDE1CamCa4cAMP__PDE1CamCa4_plus_AMP'
    cdef int ind_reverseRate_PDE1CamCa4cAMP__PDE1CamCa4_plus_AMP '_ind_reverseRate_PDE1CamCa4cAMP__PDE1CamCa4_plus_AMP'
    cdef int ind_forwardRate_PDE10_plus_cAMP__PDE10cAMP '_ind_forwardRate_PDE10_plus_cAMP__PDE10cAMP'
    cdef int ind_reverseRate_PDE10_plus_cAMP__PDE10cAMP '_ind_reverseRate_PDE10_plus_cAMP__PDE10cAMP'
    cdef int ind_forwardRate_PDE10cAMP__PDE10_plus_AMP '_ind_forwardRate_PDE10cAMP__PDE10_plus_AMP'
    cdef int ind_reverseRate_PDE10cAMP__PDE10_plus_AMP '_ind_reverseRate_PDE10cAMP__PDE10_plus_AMP'
    cdef int ind_forwardRate_pPDE10_plus_cAMP__pPDE10cAMP '_ind_forwardRate_pPDE10_plus_cAMP__pPDE10cAMP'
    cdef int ind_reverseRate_pPDE10_plus_cAMP__pPDE10cAMP '_ind_reverseRate_pPDE10_plus_cAMP__pPDE10cAMP'
    cdef int ind_forwardRate_pPDE10cAMP__pPDE10_plus_AMP '_ind_forwardRate_pPDE10cAMP__pPDE10_plus_AMP'
    cdef int ind_reverseRate_pPDE10cAMP__pPDE10_plus_AMP '_ind_reverseRate_pPDE10cAMP__pPDE10_plus_AMP'
    cdef int ind_forwardRate_AMP__ATP '_ind_forwardRate_AMP__ATP'
    cdef int ind_reverseRate_AMP__ATP '_ind_reverseRate_AMP__ATP'
    cdef int ind_forwardRate_PDE10_phosp1 '_ind_forwardRate_PDE10_phosp1'
    cdef int ind_reverseRate_PDE10_phosp1 '_ind_reverseRate_PDE10_phosp1'
    cdef int ind_forwardRate_PDE10_phosp2 '_ind_forwardRate_PDE10_phosp2'
    cdef int ind_reverseRate_PDE10_phosp2 '_ind_reverseRate_PDE10_phosp2'
    cdef int ind_forwardRate_PDE10cAMP_phosp1 '_ind_forwardRate_PDE10cAMP_phosp1'
    cdef int ind_reverseRate_PDE10cAMP_phosp1 '_ind_reverseRate_PDE10cAMP_phosp1'
    cdef int ind_forwardRate_PDE10cAMP_phosp2 '_ind_forwardRate_PDE10cAMP_phosp2'
    cdef int ind_reverseRate_PDE10cAMP_phosp2 '_ind_reverseRate_PDE10cAMP_phosp2'
    cdef int ind_forwardRate_PDE10_d_phosp1 '_ind_forwardRate_PDE10_d_phosp1'
    cdef int ind_reverseRate_PDE10_d_phosp1 '_ind_reverseRate_PDE10_d_phosp1'
    cdef int ind_forwardRate_PDE10_d_phosp2 '_ind_forwardRate_PDE10_d_phosp2'
    cdef int ind_reverseRate_PDE10_d_phosp2 '_ind_reverseRate_PDE10_d_phosp2'
    cdef int ind_forwardRate_PDE2_plus_cAMP__PDE2cAMP '_ind_forwardRate_PDE2_plus_cAMP__PDE2cAMP'
    cdef int ind_reverseRate_PDE2_plus_cAMP__PDE2cAMP '_ind_reverseRate_PDE2_plus_cAMP__PDE2cAMP'
    cdef int ind_forwardRate_PDE2cAMP_plus_cAMP__PDE2cAMP2 '_ind_forwardRate_PDE2cAMP_plus_cAMP__PDE2cAMP2'
    cdef int ind_reverseRate_PDE2cAMP_plus_cAMP__PDE2cAMP2 '_ind_reverseRate_PDE2cAMP_plus_cAMP__PDE2cAMP2'
    cdef int ind_forwardRate_PDE2cAMP2__PDE2cAMP_plus_AMP '_ind_forwardRate_PDE2cAMP2__PDE2cAMP_plus_AMP'
    cdef int ind_reverseRate_PDE2cAMP2__PDE2cAMP_plus_AMP '_ind_reverseRate_PDE2cAMP2__PDE2cAMP_plus_AMP'
    cdef int ind_forwardRate_PDE4_enzyme1 '_ind_forwardRate_PDE4_enzyme1'
    cdef int ind_reverseRate_PDE4_enzyme1 '_ind_reverseRate_PDE4_enzyme1'
    cdef int ind_forwardRate_PDE4_actvcomp '_ind_forwardRate_PDE4_actvcomp'
    cdef int ind_reverseRate_PDE4_actvcomp '_ind_reverseRate_PDE4_actvcomp'
    cdef int ind_forwardRate_pPDE4_enzyme1 '_ind_forwardRate_pPDE4_enzyme1'
    cdef int ind_reverseRate_pPDE4_enzyme1 '_ind_reverseRate_pPDE4_enzyme1'
    cdef int ind_forwardRate_pPDE4_actvcomp '_ind_forwardRate_pPDE4_actvcomp'
    cdef int ind_reverseRate_pPDE4_actvcomp '_ind_reverseRate_pPDE4_actvcomp'
    cdef int ind_forwardRate_PDE4_phosp1 '_ind_forwardRate_PDE4_phosp1'
    cdef int ind_reverseRate_PDE4_phosp1 '_ind_reverseRate_PDE4_phosp1'
    cdef int ind_forwardRate_PDE4_phosp2 '_ind_forwardRate_PDE4_phosp2'
    cdef int ind_reverseRate_PDE4_phosp2 '_ind_reverseRate_PDE4_phosp2'
    cdef int ind_forwardRate_PDE4cAMP_phosp1 '_ind_forwardRate_PDE4cAMP_phosp1'
    cdef int ind_reverseRate_PDE4cAMP_phosp1 '_ind_reverseRate_PDE4cAMP_phosp1'
    cdef int ind_forwardRate_PDE4cAMP_phosp2 '_ind_forwardRate_PDE4cAMP_phosp2'
    cdef int ind_reverseRate_PDE4cAMP_phosp2 '_ind_reverseRate_PDE4cAMP_phosp2'
    cdef int ind_forwardRate_PDE4_d_phosp1 '_ind_forwardRate_PDE4_d_phosp1'
    cdef int ind_reverseRate_PDE4_d_phosp1 '_ind_reverseRate_PDE4_d_phosp1'
    cdef int ind_forwardRate_PDE4_d_phosp2 '_ind_forwardRate_PDE4_d_phosp2'
    cdef int ind_reverseRate_PDE4_d_phosp2 '_ind_reverseRate_PDE4_d_phosp2'
    cdef int ind_forwardRate_Plc_plus_Ca__PlcCa '_ind_forwardRate_Plc_plus_Ca__PlcCa'
    cdef int ind_reverseRate_Plc_plus_Ca__PlcCa '_ind_reverseRate_Plc_plus_Ca__PlcCa'
    cdef int ind_forwardRate_PlcCa_plus_GqaGTP__PlcCaGqa '_ind_forwardRate_PlcCa_plus_GqaGTP__PlcCaGqa'
    cdef int ind_reverseRate_PlcCa_plus_GqaGTP__PlcCaGqa '_ind_reverseRate_PlcCa_plus_GqaGTP__PlcCaGqa'
    cdef int ind_forwardRate_Plc_plus_GqaGTP__PlcGqa '_ind_forwardRate_Plc_plus_GqaGTP__PlcGqa'
    cdef int ind_reverseRate_Plc_plus_GqaGTP__PlcGqa '_ind_reverseRate_Plc_plus_GqaGTP__PlcGqa'
    cdef int ind_forwardRate_PlcGaGTP_plus_Ca__PlcCaGqa '_ind_forwardRate_PlcGaGTP_plus_Ca__PlcCaGqa'
    cdef int ind_reverseRate_PlcGaGTP_plus_Ca__PlcCaGqa '_ind_reverseRate_PlcGaGTP_plus_Ca__PlcCaGqa'
    cdef int ind_forwardRate_PlcCa_plus_Pip2__PlcCaPip2 '_ind_forwardRate_PlcCa_plus_Pip2__PlcCaPip2'
    cdef int ind_reverseRate_PlcCa_plus_Pip2__PlcCaPip2 '_ind_reverseRate_PlcCa_plus_Pip2__PlcCaPip2'
    cdef int ind_forwardRate_PlcCaPip2__PlcCa_plus_Dag_plus_Ip3 '_ind_forwardRate_PlcCaPip2__PlcCa_plus_Dag_plus_Ip3'
    cdef int ind_reverseRate_PlcCaPip2__PlcCa_plus_Dag_plus_Ip3 '_ind_reverseRate_PlcCaPip2__PlcCa_plus_Dag_plus_Ip3'
    cdef int ind_forwardRate_PlcCaGqa_plus_Pip2__PlcCaGqaPip2 '_ind_forwardRate_PlcCaGqa_plus_Pip2__PlcCaGqaPip2'
    cdef int ind_reverseRate_PlcCaGqa_plus_Pip2__PlcCaGqaPip2 '_ind_reverseRate_PlcCaGqa_plus_Pip2__PlcCaGqaPip2'
    cdef int ind_forwardRate_PlcCaGqaPip2__PlcCaGqaDag_plus_Ip3 '_ind_forwardRate_PlcCaGqaPip2__PlcCaGqaDag_plus_Ip3'
    cdef int ind_reverseRate_PlcCaGqaPip2__PlcCaGqaDag_plus_Ip3 '_ind_reverseRate_PlcCaGqaPip2__PlcCaGqaDag_plus_Ip3'
    cdef int ind_forwardRate_PlcGqa__Plc_plus_GqaGDP '_ind_forwardRate_PlcGqa__Plc_plus_GqaGDP'
    cdef int ind_reverseRate_PlcGqa__Plc_plus_GqaGDP '_ind_reverseRate_PlcGqa__Plc_plus_GqaGDP'
    cdef int ind_forwardRate_PlcCaGqa__PlcCa_plus_GaGDP '_ind_forwardRate_PlcCaGqa__PlcCa_plus_GaGDP'
    cdef int ind_reverseRate_PlcCaGqa__PlcCa_plus_GaGDP '_ind_reverseRate_PlcCaGqa__PlcCa_plus_GaGDP'
    cdef int ind_forwardRate_Ca_plus_Dlg__CaDgl '_ind_forwardRate_Ca_plus_Dlg__CaDgl'
    cdef int ind_reverseRate_Ca_plus_Dlg__CaDgl '_ind_reverseRate_Ca_plus_Dlg__CaDgl'
    cdef int ind_forwardRate_Dag_plus_CaDlg__DagCaDgl '_ind_forwardRate_Dag_plus_CaDlg__DagCaDgl'
    cdef int ind_reverseRate_Dag_plus_CaDlg__DagCaDgl '_ind_reverseRate_Dag_plus_CaDlg__DagCaDgl'
    cdef int ind_forwardRate_DagCaDgl__CaDgl_plus_2ag '_ind_forwardRate_DagCaDgl__CaDgl_plus_2ag'
    cdef int ind_reverseRate_DagCaDgl__CaDgl_plus_2ag '_ind_reverseRate_DagCaDgl__CaDgl_plus_2ag'
    cdef int ind_forwardRate_Ip3__Ip3degrad '_ind_forwardRate_Ip3__Ip3degrad'
    cdef int ind_reverseRate_Ip3__Ip3degrad '_ind_reverseRate_Ip3__Ip3degrad'
    cdef int ind_forwardRate_Ip3degrad_plus_PIkinase__Ip3degPIk '_ind_forwardRate_Ip3degrad_plus_PIkinase__Ip3degPIk'
    cdef int ind_reverseRate_Ip3degrad_plus_PIkinase__Ip3degPIk '_ind_reverseRate_Ip3degrad_plus_PIkinase__Ip3degPIk'
    cdef int ind_forwardRate_Ip3degPIk__PIP2__plus__PIkinase '_ind_forwardRate_Ip3degPIk__PIP2__plus__PIkinase'
    cdef int ind_reverseRate_Ip3degPIk__PIP2__plus__PIkinase '_ind_reverseRate_Ip3degPIk__PIP2__plus__PIkinase'
    cdef int ind_forwardRate_2ag__2agDegrad '_ind_forwardRate_2ag__2agDegrad'
    cdef int ind_reverseRate_2ag__2agDegrad '_ind_reverseRate_2ag__2agDegrad'
    cdef int ind_forwardRate_DagKdag1 '_ind_forwardRate_DagKdag1'
    cdef int ind_reverseRate_DagKdag1 '_ind_reverseRate_DagKdag1'
    cdef int ind_forwardRate_DagKdag1_1 '_ind_forwardRate_DagKdag1_1'
    cdef int ind_reverseRate_DagKdag1_1 '_ind_reverseRate_DagKdag1_1'
    cdef int ind_forwardRate_Pkc_plus_Ca__PkcCa '_ind_forwardRate_Pkc_plus_Ca__PkcCa'
    cdef int ind_reverseRate_Pkc_plus_Ca__PkcCa '_ind_reverseRate_Pkc_plus_Ca__PkcCa'
    cdef int ind_forwardRate_PkcCa_plus_Dag__PkcCaDag '_ind_forwardRate_PkcCa_plus_Dag__PkcCaDag'
    cdef int ind_reverseRate_PkcCa_plus_Dag__PkcCaDag '_ind_reverseRate_PkcCa_plus_Dag__PkcCaDag'
    cdef int ind_forwardRate_PKA_bind '_ind_forwardRate_PKA_bind'
    cdef int ind_reverseRate_PKA_bind '_ind_reverseRate_PKA_bind'
    cdef int ind_forwardRate_PKAcAMP2_bind '_ind_forwardRate_PKAcAMP2_bind'
    cdef int ind_reverseRate_PKAcAMP2_bind '_ind_reverseRate_PKAcAMP2_bind'
    cdef int ind_forwardRate_PKAcAMP4_diss '_ind_forwardRate_PKAcAMP4_diss'
    cdef int ind_reverseRate_PKAcAMP4_diss '_ind_reverseRate_PKAcAMP4_diss'
    cdef int ind_forwardRate_Cam_plus_PP2B__PP2BCam_reac '_ind_forwardRate_Cam_plus_PP2B__PP2BCam_reac'
    cdef int ind_reverseRate_Cam_plus_PP2B__PP2BCam_reac '_ind_reverseRate_Cam_plus_PP2B__PP2BCam_reac'
    cdef int ind_forwardRate_CamCa2C_plus_PP2B__PP2BCamCa2C_reac '_ind_forwardRate_CamCa2C_plus_PP2B__PP2BCamCa2C_reac'
    cdef int ind_reverseRate_CamCa2C_plus_PP2B__PP2BCamCa2C_reac '_ind_reverseRate_CamCa2C_plus_PP2B__PP2BCamCa2C_reac'
    cdef int ind_forwardRate_CamCa2N_plus_PP2B__PP2BCamCa2N_reac '_ind_forwardRate_CamCa2N_plus_PP2B__PP2BCamCa2N_reac'
    cdef int ind_reverseRate_CamCa2N_plus_PP2B__PP2BCamCa2N_reac '_ind_reverseRate_CamCa2N_plus_PP2B__PP2BCamCa2N_reac'
    cdef int ind_forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac '_ind_forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac'
    cdef int ind_reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac '_ind_reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac'
    cdef int ind_forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac '_ind_forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac'
    cdef int ind_reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac '_ind_reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac'
    cdef int ind_forwardRate_CamCa4_plus_PP2B__PP2BCamCa4_reac '_ind_forwardRate_CamCa4_plus_PP2B__PP2BCamCa4_reac'
    cdef int ind_reverseRate_CamCa4_plus_PP2B__PP2BCamCa4_reac '_ind_reverseRate_CamCa4_plus_PP2B__PP2BCamCa4_reac'
    cdef int ind_forwardRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac '_ind_forwardRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac'
    cdef int ind_reverseRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac '_ind_reverseRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac'
    cdef int ind_forwardRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac '_ind_forwardRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac'
    cdef int ind_reverseRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac '_ind_reverseRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac'
    cdef int ind_forwardRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac '_ind_forwardRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac'
    cdef int ind_reverseRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac '_ind_reverseRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac'
    cdef int ind_forwardRate_PKAcPP2AB56d__PKAc_plus_pPP2A_reac '_ind_forwardRate_PKAcPP2AB56d__PKAc_plus_pPP2A_reac'
    cdef int ind_reverseRate_PKAcPP2AB56d__PKAc_plus_pPP2A_reac '_ind_reverseRate_PKAcPP2AB56d__PKAc_plus_pPP2A_reac'
    cdef int ind_forwardRate_pPP2A__PP2A_reac '_ind_forwardRate_pPP2A__PP2A_reac'
    cdef int ind_reverseRate_pPP2A__PP2A_reac '_ind_reverseRate_pPP2A__PP2A_reac'
    cdef int ind_forwardRate_PP2ABPR72_plus_Ca__PP2Acal_reac '_ind_forwardRate_PP2ABPR72_plus_Ca__PP2Acal_reac'
    cdef int ind_reverseRate_PP2ABPR72_plus_Ca__PP2Acal_reac '_ind_reverseRate_PP2ABPR72_plus_Ca__PP2Acal_reac'
    cdef int ind_forwardRate_D32_plus_PKAc__D32PKAc_reac '_ind_forwardRate_D32_plus_PKAc__D32PKAc_reac'
    cdef int ind_reverseRate_D32_plus_PKAc__D32PKAc_reac '_ind_reverseRate_D32_plus_PKAc__D32PKAc_reac'
    cdef int ind_forwardRate_D32PKAc__PKAc_plus_D32p34_reac '_ind_forwardRate_D32PKAc__PKAc_plus_D32p34_reac'
    cdef int ind_reverseRate_D32PKAc__PKAc_plus_D32p34_reac '_ind_reverseRate_D32PKAc__PKAc_plus_D32p34_reac'
    cdef int ind_forwardRate_D32p34_plus_PP1__D32p34pp1_reac '_ind_forwardRate_D32p34_plus_PP1__D32p34pp1_reac'
    cdef int ind_reverseRate_D32p34_plus_PP1__D32p34pp1_reac '_ind_reverseRate_D32p34_plus_PP1__D32p34pp1_reac'
    cdef int ind_forwardRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac '_ind_forwardRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac'
    cdef int ind_reverseRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac '_ind_reverseRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac'
    cdef int ind_forwardRate_D32p34PP2BCamCa4__PP2BCamCa4_plus_D32_reac '_ind_forwardRate_D32p34PP2BCamCa4__PP2BCamCa4_plus_D32_reac'
    cdef int ind_reverseRate_D32p34PP2BCamCa4__PP2BCamCa4_plus_D32_reac '_ind_reverseRate_D32p34PP2BCamCa4__PP2BCamCa4_plus_D32_reac'
    cdef int ind_forwardRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac '_ind_forwardRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac'
    cdef int ind_reverseRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac '_ind_reverseRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac'
    cdef int ind_forwardRate_D32p34PP1PP2BCamCa4__PP1_plus_PP2BCamCa4_plus_D32_reac '_ind_forwardRate_D32p34PP1PP2BCamCa4__PP1_plus_PP2BCamCa4_plus_D32_reac'
    cdef int ind_reverseRate_D32p34PP1PP2BCamCa4__PP1_plus_PP2BCamCa4_plus_D32_reac '_ind_reverseRate_D32p34PP1PP2BCamCa4__PP1_plus_PP2BCamCa4_plus_D32_reac'
    cdef int ind_forwardRate_D32p34_plus_PP2ABPR72__D32p34PP2ABPR72_reac '_ind_forwardRate_D32p34_plus_PP2ABPR72__D32p34PP2ABPR72_reac'
    cdef int ind_reverseRate_D32p34_plus_PP2ABPR72__D32p34PP2ABPR72_reac '_ind_reverseRate_D32p34_plus_PP2ABPR72__D32p34PP2ABPR72_reac'
    cdef int ind_forwardRate_D32p34PP2ABPR72__PP2ABPR72_plus_D32_reac '_ind_forwardRate_D32p34PP2ABPR72__PP2ABPR72_plus_D32_reac'
    cdef int ind_reverseRate_D32p34PP2ABPR72__PP2ABPR72_plus_D32_reac '_ind_reverseRate_D32p34PP2ABPR72__PP2ABPR72_plus_D32_reac'
    cdef int ind_forwardRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac '_ind_forwardRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac'
    cdef int ind_reverseRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac '_ind_reverseRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac'
    cdef int ind_forwardRate_D32p34PP2AB56d__PP2AB56d_plus_D32_reac '_ind_forwardRate_D32p34PP2AB56d__PP2AB56d_plus_D32_reac'
    cdef int ind_reverseRate_D32p34PP2AB56d__PP2AB56d_plus_D32_reac '_ind_reverseRate_D32p34PP2AB56d__PP2AB56d_plus_D32_reac'
    cdef int ind_forwardRate_D32p34pp1_plus_PP2ABPR72__D32p34PP1PP2ABPR72_reac '_ind_forwardRate_D32p34pp1_plus_PP2ABPR72__D32p34PP1PP2ABPR72_reac'
    cdef int ind_reverseRate_D32p34pp1_plus_PP2ABPR72__D32p34PP1PP2ABPR72_reac '_ind_reverseRate_D32p34pp1_plus_PP2ABPR72__D32p34PP1PP2ABPR72_reac'
    cdef int ind_forwardRate_D32p34PP1PP2ABPR72__PP1_plus_PP2ABPR72_plus_D32_reac '_ind_forwardRate_D32p34PP1PP2ABPR72__PP1_plus_PP2ABPR72_plus_D32_reac'
    cdef int ind_reverseRate_D32p34PP1PP2ABPR72__PP1_plus_PP2ABPR72_plus_D32_reac '_ind_reverseRate_D32p34PP1PP2ABPR72__PP1_plus_PP2ABPR72_plus_D32_reac'
    cdef int ind_forwardRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac '_ind_forwardRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac'
    cdef int ind_reverseRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac '_ind_reverseRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac'
    cdef int ind_forwardRate_D32p34PP1PP2AB56d__PP1_plus_PP2AB56d_plus_D32_reac '_ind_forwardRate_D32p34PP1PP2AB56d__PP1_plus_PP2AB56d_plus_D32_reac'
    cdef int ind_reverseRate_D32p34PP1PP2AB56d__PP1_plus_PP2AB56d_plus_D32_reac '_ind_reverseRate_D32p34PP1PP2AB56d__PP1_plus_PP2AB56d_plus_D32_reac'
    cdef int ind_forwardRate_Cdk5_plus_D32__Cdk5D32_reac '_ind_forwardRate_Cdk5_plus_D32__Cdk5D32_reac'
    cdef int ind_reverseRate_Cdk5_plus_D32__Cdk5D32_reac '_ind_reverseRate_Cdk5_plus_D32__Cdk5D32_reac'
    cdef int ind_forwardRate_Cdk5D32__D32p75_plus_Cdk5_reac '_ind_forwardRate_Cdk5D32__D32p75_plus_Cdk5_reac'
    cdef int ind_reverseRate_Cdk5D32__D32p75_plus_Cdk5_reac '_ind_reverseRate_Cdk5D32__D32p75_plus_Cdk5_reac'
    cdef int ind_forwardRate_D32p75_plus_PKAc__D32p75PKAc_reac '_ind_forwardRate_D32p75_plus_PKAc__D32p75PKAc_reac'
    cdef int ind_reverseRate_D32p75_plus_PKAc__D32p75PKAc_reac '_ind_reverseRate_D32p75_plus_PKAc__D32p75PKAc_reac'
    cdef int ind_forwardRate_D32p75_plus_pPP2A__D32p75pPP2A_reac '_ind_forwardRate_D32p75_plus_pPP2A__D32p75pPP2A_reac'
    cdef int ind_reverseRate_D32p75_plus_pPP2A__D32p75pPP2A_reac '_ind_reverseRate_D32p75_plus_pPP2A__D32p75pPP2A_reac'
    cdef int ind_forwardRate_D32p75pPP2A__D32_plus_pPP2A_reac '_ind_forwardRate_D32p75pPP2A__D32_plus_pPP2A_reac'
    cdef int ind_reverseRate_D32p75pPP2A__D32_plus_pPP2A_reac '_ind_reverseRate_D32p75pPP2A__D32_plus_pPP2A_reac'
    cdef int ind_forwardRate_D32p75_plus_PP2ABPR72__D32p75PP2ABPR72_reac '_ind_forwardRate_D32p75_plus_PP2ABPR72__D32p75PP2ABPR72_reac'
    cdef int ind_reverseRate_D32p75_plus_PP2ABPR72__D32p75PP2ABPR72_reac '_ind_reverseRate_D32p75_plus_PP2ABPR72__D32p75PP2ABPR72_reac'
    cdef int ind_forwardRate_D32p75PP2ABPR72__D32_plus_PP2ABPR72_reac '_ind_forwardRate_D32p75PP2ABPR72__D32_plus_PP2ABPR72_reac'
    cdef int ind_reverseRate_D32p75PP2ABPR72__D32_plus_PP2ABPR72_reac '_ind_reverseRate_D32p75PP2ABPR72__D32_plus_PP2ABPR72_reac'
    cdef int ind_forwardRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac '_ind_forwardRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac'
    cdef int ind_reverseRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac '_ind_reverseRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac'
    cdef int ind_forwardRate_D32p75PP2AB56d__D32_plus_PP2AB56d_reac '_ind_forwardRate_D32p75PP2AB56d__D32_plus_PP2AB56d_reac'
    cdef int ind_reverseRate_D32p75PP2AB56d__D32_plus_PP2AB56d_reac '_ind_reverseRate_D32p75PP2AB56d__D32_plus_PP2AB56d_reac'
    cdef int ind_forwardRate_D32p75_plus_PP2Acal__D32p75PP2Acal_reac '_ind_forwardRate_D32p75_plus_PP2Acal__D32p75PP2Acal_reac'
    cdef int ind_reverseRate_D32p75_plus_PP2Acal__D32p75PP2Acal_reac '_ind_reverseRate_D32p75_plus_PP2Acal__D32p75PP2Acal_reac'
    cdef int ind_forwardRate_D32p75PP2Acal_D32_plus_PP2Acal_reac '_ind_forwardRate_D32p75PP2Acal_D32_plus_PP2Acal_reac'
    cdef int ind_reverseRate_D32p75PP2Acal_D32_plus_PP2Acal_reac '_ind_reverseRate_D32p75PP2Acal_D32_plus_PP2Acal_reac'
    cdef int ind_forwardRate_CK_bind '_ind_forwardRate_CK_bind'
    cdef int ind_reverseRate_CK_bind '_ind_reverseRate_CK_bind'
    cdef int ind_forwardRate_CKCam_bind '_ind_forwardRate_CKCam_bind'
    cdef int ind_reverseRate_CKCam_bind '_ind_reverseRate_CKCam_bind'
    cdef int ind_forwardRate_CKCam_bind_1 '_ind_forwardRate_CKCam_bind_1'
    cdef int ind_reverseRate_CKCam_bind_1 '_ind_reverseRate_CKCam_bind_1'
    cdef int ind_forwardRate_CKCam_bind_2 '_ind_forwardRate_CKCam_bind_2'
    cdef int ind_reverseRate_CKCam_bind_2 '_ind_reverseRate_CKCam_bind_2'
    cdef int ind_forwardRate_CKp_auto '_ind_forwardRate_CKp_auto'
    cdef int ind_reverseRate_CKp_auto '_ind_reverseRate_CKp_auto'
    cdef int ind_forwardRate_CKp_bind '_ind_forwardRate_CKp_bind'
    cdef int ind_reverseRate_CKp_bind '_ind_reverseRate_CKp_bind'
    cdef int ind_forwardRate_CKpPP1_reac '_ind_forwardRate_CKpPP1_reac'
    cdef int ind_reverseRate_CKpPP1_reac '_ind_reverseRate_CKpPP1_reac'
    cdef int ind_forwardRate_CKpCamCa4PP1_bind '_ind_forwardRate_CKpCamCa4PP1_bind'
    cdef int ind_reverseRate_CKpCamCa4PP1_bind '_ind_reverseRate_CKpCamCa4PP1_bind'
    cdef int ind_forwardRate_CKpCamCa4PP1_reac '_ind_forwardRate_CKpCamCa4PP1_reac'
    cdef int ind_reverseRate_CKpCamCa4PP1_reac '_ind_reverseRate_CKpCamCa4PP1_reac'
    cdef int ind_forwardRate_Dgl_CKCamCa4_bind '_ind_forwardRate_Dgl_CKCamCa4_bind'
    cdef int ind_reverseRate_Dgl_CKCamCa4_bind '_ind_reverseRate_Dgl_CKCamCa4_bind'
    cdef int ind_forwardRate_Dgl_CKCam_diss '_ind_forwardRate_Dgl_CKCam_diss'
    cdef int ind_reverseRate_Dgl_CKCam_diss '_ind_reverseRate_Dgl_CKCam_diss'
    cdef int ind_forwardRate_Dgl_CKpCamCa4_bind '_ind_forwardRate_Dgl_CKpCamCa4_bind'
    cdef int ind_reverseRate_Dgl_CKpCamCa4_bind '_ind_reverseRate_Dgl_CKpCamCa4_bind'
    cdef int ind_forwardRate_Dgl_CKpCamCa4_diss '_ind_forwardRate_Dgl_CKpCamCa4_diss'
    cdef int ind_reverseRate_Dgl_CKpCamCa4_diss '_ind_reverseRate_Dgl_CKpCamCa4_diss'
    cdef int ind_forwardRate_CaDgl_CKpCamCa4_bind '_ind_forwardRate_CaDgl_CKpCamCa4_bind'
    cdef int ind_reverseRate_CaDgl_CKpCamCa4_bind '_ind_reverseRate_CaDgl_CKpCamCa4_bind'
    cdef int ind_forwardRate_CaDgl_CKpCamCa4_diss '_ind_forwardRate_CaDgl_CKpCamCa4_diss'
    cdef int ind_reverseRate_CaDgl_CKpCamCa4_diss '_ind_reverseRate_CaDgl_CKpCamCa4_diss'
    cdef int ind_forwardRate_Dgl_CKp_bind '_ind_forwardRate_Dgl_CKp_bind'
    cdef int ind_reverseRate_Dgl_CKp_bind '_ind_reverseRate_Dgl_CKp_bind'
    cdef int ind_forwardRate_Dgl_CKp_diss '_ind_forwardRate_Dgl_CKp_diss'
    cdef int ind_reverseRate_Dgl_CKp_diss '_ind_reverseRate_Dgl_CKp_diss'
    cdef int ind_forwardRate_pDgl_dephos '_ind_forwardRate_pDgl_dephos'
    cdef int ind_reverseRate_pDgl_dephos '_ind_reverseRate_pDgl_dephos'
    cdef int ind_forwardRate_Ca_plus_pDlg__CapDgl '_ind_forwardRate_Ca_plus_pDlg__CapDgl'
    cdef int ind_reverseRate_Ca_plus_pDlg__CapDgl '_ind_reverseRate_Ca_plus_pDlg__CapDgl'
    cdef int ind_forwardRate_Dag_plus_CapDlg__DagCapDgl '_ind_forwardRate_Dag_plus_CapDlg__DagCapDgl'
    cdef int ind_reverseRate_Dag_plus_CapDlg__DagCapDgl '_ind_reverseRate_Dag_plus_CapDlg__DagCapDgl'
    cdef int ind_forwardRate_DagCapDgl__CapDgl_plus_2ag '_ind_forwardRate_DagCapDgl__CapDgl_plus_2ag'
    cdef int ind_reverseRate_DagCapDgl__CapDgl_plus_2ag '_ind_reverseRate_DagCapDgl__CapDgl_plus_2ag'
    cdef int ind_forwardRate_AKAR3_plus_PKAc_bind '_ind_forwardRate_AKAR3_plus_PKAc_bind'
    cdef int ind_reverseRate_AKAR3_plus_PKAc_bind '_ind_reverseRate_AKAR3_plus_PKAc_bind'
    cdef int ind_forwardRate_PKAcAKAR3_phos '_ind_forwardRate_PKAcAKAR3_phos'
    cdef int ind_reverseRate_PKAcAKAR3_phos '_ind_reverseRate_PKAcAKAR3_phos'
    cdef int ind_forwardRate_pAKAR3_plus_PP1_bind '_ind_forwardRate_pAKAR3_plus_PP1_bind'
    cdef int ind_reverseRate_pAKAR3_plus_PP1_bind '_ind_reverseRate_pAKAR3_plus_PP1_bind'
    cdef int ind_forwardRate_PP1pAKAR3_dephos '_ind_forwardRate_PP1pAKAR3_dephos'
    cdef int ind_reverseRate_PP1pAKAR3_dephos '_ind_reverseRate_PP1pAKAR3_dephos'
    cdef int ind_forwardRate_GsaGTP_AC1_reac '_ind_forwardRate_GsaGTP_AC1_reac'
    cdef int ind_reverseRate_GsaGTP_AC1_reac '_ind_reverseRate_GsaGTP_AC1_reac'
    cdef int ind_forwardRate_AC1GsaGTP_CamCa4_Reac '_ind_forwardRate_AC1GsaGTP_CamCa4_Reac'
    cdef int ind_reverseRate_AC1GsaGTP_CamCa4_Reac '_ind_reverseRate_AC1GsaGTP_CamCa4_Reac'
    cdef int ind_forwardRate_AC1GsaGTPCamCa4_ATP_Reac '_ind_forwardRate_AC1GsaGTPCamCa4_ATP_Reac'
    cdef int ind_reverseRate_AC1GsaGTPCamCa4_ATP_Reac '_ind_reverseRate_AC1GsaGTPCamCa4_ATP_Reac'
    cdef int ind_forwardRate_AC1GsaGTPCamCa4ATP_diss '_ind_forwardRate_AC1GsaGTPCamCa4ATP_diss'
    cdef int ind_reverseRate_AC1GsaGTPCamCa4ATP_diss '_ind_reverseRate_AC1GsaGTPCamCa4ATP_diss'
    cdef int ind_forwardRate_AC1CamCa4_Reac '_ind_forwardRate_AC1CamCa4_Reac'
    cdef int ind_reverseRate_AC1CamCa4_Reac '_ind_reverseRate_AC1CamCa4_Reac'
    cdef int ind_forwardRate_AC1CamCa4ATP_bind '_ind_forwardRate_AC1CamCa4ATP_bind'
    cdef int ind_reverseRate_AC1CamCa4ATP_bind '_ind_reverseRate_AC1CamCa4ATP_bind'
    cdef int ind_forwardRate_AC1CamCa4ATP_disso '_ind_forwardRate_AC1CamCa4ATP_disso'
    cdef int ind_reverseRate_AC1CamCa4ATP_disso '_ind_reverseRate_AC1CamCa4ATP_disso'

    # ---- Ilya model params ---- #
    cdef int ind_xst_LTPwin "_ind_xst_LTPwin"
    cdef int ind_xstep_LTPwin "_ind_xstep_LTPwin"
    cdef int ind_xst_caL13 "_ind_xst_caL13"
    cdef int ind_xstep_caL13 "_ind_xstep_caL13"
    cdef int ind_xst_NMDA "_ind_xst_NMDA"
    cdef int ind_xstep_NMDA "_ind_xstep_NMDA"
    cdef int ind_nb_Glu_stim "_ind_nb_Glu_stim"
    cdef int ind_nb_Iact_stim "_ind_nb_Iact_stim"
    cdef int ind_n_x_LTPwin "_ind_n_x_LTPwin"
    cdef int ind_n_x_caL13 "_ind_n_x_caL13"
    cdef int ind_n_x_NMDA "_ind_n_x_NMDA"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ___________ FUNCTIONS ___________ #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

cdef double hill_func(double x, double k, double n) nogil:
    x_n = x**n
    return x_n/(x_n +k**n)

cdef double heaviside_func(double x) nogil:
    cdef double heaviside
    if (x == 0):
        heaviside = 0.5
    elif (x < 0):
        heaviside = 0
    else:
        heaviside = 1.0
    return heaviside

cdef double rect_func(double x, double x_low, double x_high) nogil:
    cdef double rect
    if (x < x_low):
        rect = 0
    elif (x > x_high):
        rect = 0
    else:
        rect = 1.0
    return rect

cdef double lin_interpTable_brds(gsl_vector* Table, int li, double i_point) nogil:
    cdef int ip=int(floor(i_point) -1)
    cdef double res

    if ip < 0:
        res = gsl_vector_get(Table, 0)
    elif ip < li-1:
        res = gsl_vector_get(Table, ip) +(gsl_vector_get(Table,ip+1) -gsl_vector_get(Table, ip))*(i_point-1 -ip)
    else:
        res = gsl_vector_get(Table, li-1)
    return res


cdef double efun(double x) nogil:
    cdef double efun
    if (fabs(x)<1e-4):
        efun = 1-x/2
    elif (x<=85):
        efun = x/(exp(x)-1)
    else:
        efun = 0
    return efun


cdef double tauCa(double Ca, double BT_CaBuff, double KdB_CaBuff) nogil:
    return 1 +BT_CaBuff/KdB_CaBuff/(1+Ca/KdB_CaBuff)**2 # No unit


# Function which contains the system of ODE; the input parameters can't be easily modify #
# I put all the extra parameters as an array in void *params #
# Furthermore, this function is in nogil mode; we can only use C. If you want to use python, use <with gil> statement #
cdef int func (double t,  double y[], double f[], void *params) nogil:

    cdef double on_caL13 =(<double *> params)[ind_on_caL13]
    cdef double pcaLbar =(<double *> params)[ind_pcaLbar]
    cdef double mslope_caL13 =(<double *> params)[ind_mslope_caL13]
    cdef double hshift_caL13 =(<double *> params)[ind_hshift_caL13]
    cdef double vm_caL13 =(<double *> params)[ind_vm_caL13]
    cdef double mshift_caL13 =(<double *> params)[ind_mshift_caL13]
    cdef double hslope_caL13 =(<double *> params)[ind_hslope_caL13]
    cdef double kpr_caL13 =(<double *> params)[ind_kpr_caL13]
    cdef double c_caL13 =(<double *> params)[ind_c_caL13]
    cdef double k_caL13 =(<double *> params)[ind_k_caL13]
    cdef double hvhalf_caL13 =(<double *> params)[ind_hvhalf_caL13]
    cdef double mvhalf_caL13 =(<double *> params)[ind_mvhalf_caL13]
    cdef double cpr_caL13 =(<double *> params)[ind_cpr_caL13]
    cdef double htau_caL13 =(<double *> params)[ind_htau_caL13]
    cdef double hqfact_caL13 =(<double *> params)[ind_hqfact_caL13]
    cdef double qfact_caL13 =(<double *> params)[ind_qfact_caL13]
    cdef double on_TRPV1 =(<double *> params)[ind_on_TRPV1]
    cdef double gTRPV1 =(<double *> params)[ind_gTRPV1]
    cdef double p_ca_TRPV1 =(<double *> params)[ind_p_ca_TRPV1]
    cdef double C_TRPV1 =(<double *> params)[ind_C_TRPV1]
    cdef double D_TRPV1 =(<double *> params)[ind_D_TRPV1]
    cdef double DH_TRPV1 =(<double *> params)[ind_DH_TRPV1]
    cdef double KD_TRPV1 =(<double *> params)[ind_KD_TRPV1]
    cdef double J0_TRPV1 =(<double *> params)[ind_J0_TRPV1]
    cdef double L_TRPV1 =(<double *> params)[ind_L_TRPV1]
    cdef double P_TRPV1 =(<double *> params)[ind_P_TRPV1]
    cdef double z_TRPV1 =(<double *> params)[ind_z_TRPV1]
    cdef double DS_TRPV1 =(<double *> params)[ind_DS_TRPV1]
    cdef double K_TRPV1 =(<double *> params)[ind_K_TRPV1]
    cdef double NMDA =(<double *> params)[ind_NMDA]
    cdef double VDCC =(<double *> params)[ind_VDCC]
    cdef double TRPV1 =(<double *> params)[ind_TRPV1]
    cdef double BT_CaBuff =(<double *> params)[ind_BT_CaBuff]
    cdef double Cab =(<double *> params)[ind_Cab]
    cdef double tauCab =(<double *> params)[ind_tauCab]
    cdef double KdB_CaBuff =(<double *> params)[ind_KdB_CaBuff]
    cdef double a2_CICR =(<double *> params)[ind_a2_CICR]
    cdef double ver_CICR =(<double *> params)[ind_ver_CICR]
    cdef double d3_CICR =(<double *> params)[ind_d3_CICR]
    cdef double rc_CICR =(<double *> params)[ind_rc_CICR]
    cdef double rl_CICR =(<double *> params)[ind_rl_CICR]
    cdef double rhoER_CICR =(<double *> params)[ind_rhoER_CICR]
    cdef double d5_CICR =(<double *> params)[ind_d5_CICR]
    cdef double d2_CICR =(<double *> params)[ind_d2_CICR]
    cdef double ker_CICR =(<double *> params)[ind_ker_CICR]
    cdef double d1_CICR =(<double *> params)[ind_d1_CICR]
    cdef double kappad_IP3 =(<double *> params)[ind_kappad_IP3]
    cdef double kdelta_IP3 =(<double *> params)[ind_kdelta_IP3]
    cdef double r5p_IP3 =(<double *> params)[ind_r5p_IP3]
    cdef double v3k_IP3 =(<double *> params)[ind_v3k_IP3]
    cdef double kd_IP3 =(<double *> params)[ind_kd_IP3]
    cdef double vdelta_IP3 =(<double *> params)[ind_vdelta_IP3]
    cdef double k3_IP3 =(<double *> params)[ind_k3_IP3]
    cdef double kr_IP3 =(<double *> params)[ind_kr_IP3]
    cdef double kp_IP3 =(<double *> params)[ind_kp_IP3]
    cdef double vbeta_IP3 =(<double *> params)[ind_vbeta_IP3]
    cdef double kpi_IP3 =(<double *> params)[ind_kpi_IP3]
    cdef double n3_IP3 =(<double *> params)[ind_n3_IP3]
    cdef double KDGL =(<double *> params)[ind_KDGL]
    cdef double kMAGL =(<double *> params)[ind_kMAGL]
    cdef double rDGL =(<double *> params)[ind_rDGL]
    cdef double kDAGK =(<double *> params)[ind_kDAGK]
    cdef double nK_DAGLP =(<double *> params)[ind_nK_DAGLP]
    cdef double rP_DAGLP =(<double *> params)[ind_rP_DAGLP]
    cdef double rK_DAGLP =(<double *> params)[ind_rK_DAGLP]
    cdef double on_ECb =(<double *> params)[ind_on_ECb]
    cdef double ECb_CB1R_on =(<double *> params)[ind_ECb_CB1R_on]
    cdef double kCB1R =(<double *> params)[ind_kCB1R]
    cdef double alphaAEACB1 =(<double *> params)[ind_alphaAEACB1]
    cdef double P1_ECb =(<double *> params)[ind_P1_ECb]
    cdef double P2_ECb =(<double *> params)[ind_P2_ECb]
    cdef double P3_ECb =(<double *> params)[ind_P3_ECb]
    cdef double P4_ECb =(<double *> params)[ind_P4_ECb]
    cdef double vATAEA =(<double *> params)[ind_vATAEA]
    cdef double LTDstart =(<double *> params)[ind_LTDstart]
    cdef double LTDstop =(<double *> params)[ind_LTDstop]
    cdef double LTDMax =(<double *> params)[ind_LTDMax]
    cdef double LTPstart =(<double *> params)[ind_LTPstart]
    cdef double LTPMax =(<double *> params)[ind_LTPMax]
    cdef double KFAAH =(<double *> params)[ind_KFAAH]
    cdef double vFAAH =(<double *> params)[ind_vFAAH]
    cdef double on_ECb_smooth =(<double *> params)[ind_on_ECb_smooth]
    cdef double K_ECb_smooth =(<double *> params)[ind_K_ECb_smooth]
    cdef double n_ECb_smooth =(<double *> params)[ind_n_ECb_smooth]
    cdef double kw_ECb_smooth =(<double *> params)[ind_kw_ECb_smooth]
    cdef double tau_ECb_smooth =(<double *> params)[ind_tau_ECb_smooth]
    cdef double kadd_ECb_smooth =(<double *> params)[ind_kadd_ECb_smooth]
    cdef double kn_ECb_smooth =(<double *> params)[ind_kn_ECb_smooth]
    cdef double kpkaI1 =(<double *> params)[ind_kpkaI1]
    cdef double PP10 =(<double *> params)[ind_PP10]
    cdef double KM =(<double *> params)[ind_KM]
    cdef double K5 =(<double *> params)[ind_K5]
    cdef double k12 =(<double *> params)[ind_k12]
    cdef double k11 =(<double *> params)[ind_k11]
    cdef double kcan0I1 =(<double *> params)[ind_kcan0I1]
    cdef double km11 =(<double *> params)[ind_km11]
    cdef double CaMKT =(<double *> params)[ind_CaMKT]
    cdef double k7 =(<double *> params)[ind_k7]
    cdef double k6 =(<double *> params)[ind_k6]
    cdef double ncanI1 =(<double *> params)[ind_ncanI1]
    cdef double I10 =(<double *> params)[ind_I10]
    cdef double CaMT =(<double *> params)[ind_CaMT]
    cdef double KdpkaI1 =(<double *> params)[ind_KdpkaI1]
    cdef double kcanI1 =(<double *> params)[ind_kcanI1]
    cdef double KdcanI1 =(<double *> params)[ind_KdcanI1]
    cdef double kpka0I1 =(<double *> params)[ind_kpka0I1]
    cdef double Ka3 =(<double *> params)[ind_Ka3]
    cdef double Ka2 =(<double *> params)[ind_Ka2]
    cdef double Ka1 =(<double *> params)[ind_Ka1]
    cdef double Ka4 =(<double *> params)[ind_Ka4]
    cdef double npkaI1 =(<double *> params)[ind_npkaI1]
    cdef double conc_max_sub_unit =(<double *> params)[ind_conc_max_sub_unit]
    cdef double K9 =(<double *> params)[ind_K9]
    cdef double gamma1DA =(<double *> params)[ind_gamma1DA]
    cdef double gamma2DA =(<double *> params)[ind_gamma2DA]
    cdef double DA =(<double *> params)[ind_DA]
    cdef double on_AMPA =(<double *> params)[ind_on_AMPA]
    cdef double gAMPA =(<double *> params)[ind_gAMPA]
    cdef double Epsilon_AMPA =(<double *> params)[ind_Epsilon_AMPA]
    cdef double Beta_AMPA =(<double *> params)[ind_Beta_AMPA]
    cdef double Alpha_AMPA =(<double *> params)[ind_Alpha_AMPA]
    cdef double Gamma_AMPA =(<double *> params)[ind_Gamma_AMPA]
    cdef double on_NMDA =(<double *> params)[ind_on_NMDA]
    cdef double gNMDA =(<double *> params)[ind_gNMDA]
    cdef double p_ca_NMDA =(<double *> params)[ind_p_ca_NMDA]
    cdef double Mg_NMDA =(<double *> params)[ind_Mg_NMDA]
    cdef double Alpha_NMDA =(<double *> params)[ind_Alpha_NMDA]
    cdef double Beta_NMDA =(<double *> params)[ind_Beta_NMDA]
    cdef double EL =(<double *> params)[ind_EL]
    cdef double gL =(<double *> params)[ind_gL]
    cdef double Cm =(<double *> params)[ind_Cm]
    cdef double on_CB1R =(<double *> params)[ind_on_CB1R]
    cdef double Alpha_CB1R =(<double *> params)[ind_Alpha_CB1R]
    cdef double Beta_CB1R =(<double *> params)[ind_Beta_CB1R]
    cdef double Gamma_CB1R =(<double *> params)[ind_Gamma_CB1R]
    cdef double Epsilon_CB1R =(<double *> params)[ind_Epsilon_CB1R]
    cdef double R =(<double *> params)[ind_R]
    cdef double RT =(<double *> params)[ind_RT]
    cdef double T =(<double *> params)[ind_T]
    cdef double F =(<double *> params)[ind_F]
    cdef double zS =(<double *> params)[ind_zS]
    cdef double Ca_out =(<double *> params)[ind_Ca_out]
    cdef double t_step =(<double *> params)[ind_t_step]
    cdef double t_start =(<double *> params)[ind_t_start]
    cdef double t_end =(<double *> params)[ind_t_end]
    cdef double ATOL =(<double *> params)[ind_ATOL]
    cdef double RTOL =(<double *> params)[ind_RTOL]
    cdef double MXSTEP =(<double *> params)[ind_MXSTEP]
    cdef double HMAX =(<double *> params)[ind_HMAX]
    cdef double APdur =(<double *> params)[ind_APdur]
    cdef double action_as_VDCC =(<double *> params)[ind_action_as_VDCC]
    cdef double DPmax =(<double *> params)[ind_DPmax]
    cdef double APmax =(<double *> params)[ind_APmax]
    cdef double tausbAP =(<double *> params)[ind_tausbAP]
    cdef double steadyrise_on =(<double *> params)[ind_steadyrise_on]
    cdef double Glumax =(<double *> params)[ind_Glumax]
    cdef double BaseLevel =(<double *> params)[ind_BaseLevel]
    cdef double tauGlu =(<double *> params)[ind_tauGlu]
    cdef double regular_on =(<double *> params)[ind_regular_on]
    cdef double tsdt =(<double *> params)[ind_tsdt]
    cdef double tpost =(<double *> params)[ind_tpost]
    cdef double post_on =(<double *> params)[ind_post_on]
    cdef double Delta_t_STDP =(<double *> params)[ind_Delta_t_STDP]
    cdef double num_stim =(<double *> params)[ind_num_stim]
    cdef double pre_on =(<double *> params)[ind_pre_on]
    cdef double Freq =(<double *> params)[ind_Freq]
    cdef double tables_step =(<double *> params)[ind_tables_step]
    cdef double betaAEACB1 =(<double *> params)[ind_betaAEACB1]
    cdef double alphatwoAGCB1 =(<double *> params)[ind_alphatwoAGCB1]
    cdef double alpha_DAGLP_Ca_cyt =(<double *> params) [ind_alpha_DAGLP_Ca_cyt]
    cdef double input_ampl_DA =(<double *> params)[ind_input_ampl_DA]
    cdef double t_stim_on_DA =(<double *> params)[ind_t_stim_on_DA]
    cdef double t_stim_off_DA =(<double *> params)[ind_t_stim_off_DA]
    cdef double interv_stim_ms_DA =(<double *> params)[ind_interv_stim_ms_DA]
    cdef double input_ampl_Ca =(<double *> params)[ind_input_ampl_Ca]
    cdef double t_stim_on_Ca =(<double *> params)[ind_t_stim_on_Ca]
    cdef double t_stim_off_Ca =(<double *> params)[ind_t_stim_off_Ca]
    cdef double interv_stim_ms_Ca =(<double *> params)[ind_interv_stim_ms_Ca]


    # ---- Blackwell model params ---- #
    cdef double forwardRate_Ca_pump1a = (<double *> params)[ind_forwardRate_Ca_pump1a]
    cdef double reverseRate_Ca_pump1a = (<double *> params)[ind_reverseRate_Ca_pump1a]
    cdef double forwardRate_Ca_pump1b = (<double *> params)[ind_forwardRate_Ca_pump1b]
    cdef double reverseRate_Ca_pump1b = (<double *> params)[ind_reverseRate_Ca_pump1b]
    cdef double forwardRate_Ca_pump2a = (<double *> params)[ind_forwardRate_Ca_pump2a]
    cdef double reverseRate_Ca_pump2a = (<double *> params)[ind_reverseRate_Ca_pump2a]
    cdef double forwardRate_Ca_pump2b = (<double *> params)[ind_forwardRate_Ca_pump2b]
    cdef double reverseRate_Ca_pump2b = (<double *> params)[ind_reverseRate_Ca_pump2b]
    cdef double forwardRate_Ca_leak = (<double *> params)[ind_forwardRate_Ca_leak]
    cdef double reverseRate_Ca_leak = (<double *> params)[ind_reverseRate_Ca_leak]
    cdef double forwardRate_Ca_leak_1 = (<double *> params)[ind_forwardRate_Ca_leak_1]
    cdef double reverseRate_Ca_leak_1 = (<double *> params)[ind_reverseRate_Ca_leak_1]
    cdef double forwardRate_Ca_buffer = (<double *> params)[ind_forwardRate_Ca_buffer]
    cdef double reverseRate_Ca_buffer = (<double *> params)[ind_reverseRate_Ca_buffer]
    cdef double forwardRate_CamC_bind = (<double *> params)[ind_forwardRate_CamC_bind]
    cdef double reverseRate_CamC_bind = (<double *> params)[ind_reverseRate_CamC_bind]
    cdef double forwardRate_CamCa2C_bind = (<double *> params)[ind_forwardRate_CamCa2C_bind]
    cdef double reverseRate_CamCa2C_bind = (<double *> params)[ind_reverseRate_CamCa2C_bind]
    cdef double forwardRate_CamN_bind = (<double *> params)[ind_forwardRate_CamN_bind]
    cdef double reverseRate_CamN_bind = (<double *> params)[ind_reverseRate_CamN_bind]
    cdef double forwardRate_CamCa2N_bind = (<double *> params)[ind_forwardRate_CamCa2N_bind]
    cdef double reverseRate_CamCa2N_bind = (<double *> params)[ind_reverseRate_CamCa2N_bind]
    cdef double forwardRate_DaDbuf__Da_plus_Dbuf = (<double *> params)[ind_forwardRate_DaDbuf__Da_plus_Dbuf]
    cdef double reverseRate_DaDbuf__Da_plus_Dbuf = (<double *> params)[ind_reverseRate_DaDbuf__Da_plus_Dbuf]
    cdef double forwardRate_Da__DaOut = (<double *> params)[ind_forwardRate_Da__DaOut]
    cdef double reverseRate_Da__DaOut = (<double *> params)[ind_reverseRate_Da__DaOut]
    cdef double forwardRate_DaOut_plus_Dbuf__NULL = (<double *> params)[ind_forwardRate_DaOut_plus_Dbuf__NULL]
    cdef double reverseRate_DaOut_plus_Dbuf__NULL = (<double *> params)[ind_reverseRate_DaOut_plus_Dbuf__NULL]
    cdef double forwardRate_Da_plus_D1R__DaD1R = (<double *> params)[ind_forwardRate_Da_plus_D1R__DaD1R]
    cdef double reverseRate_Da_plus_D1R__DaD1R = (<double *> params)[ind_reverseRate_Da_plus_D1R__DaD1R]
    cdef double forwardRate_DaD1R_plus_Gs__DaD1RGs = (<double *> params)[ind_forwardRate_DaD1R_plus_Gs__DaD1RGs]
    cdef double reverseRate_DaD1R_plus_Gs__DaD1RGs = (<double *> params)[ind_reverseRate_DaD1R_plus_Gs__DaD1RGs]
    cdef double forwardRate_Gsabg_plus_D1R__GsD1R = (<double *> params)[ind_forwardRate_Gsabg_plus_D1R__GsD1R]
    cdef double reverseRate_Gsabg_plus_D1R__GsD1R = (<double *> params)[ind_reverseRate_Gsabg_plus_D1R__GsD1R]
    cdef double forwardRate_GsD1R_plus_Da__DaD1RGs = (<double *> params)[ind_forwardRate_GsD1R_plus_Da__DaD1RGs]
    cdef double reverseRate_GsD1R_plus_Da__DaD1RGs = (<double *> params)[ind_reverseRate_GsD1R_plus_Da__DaD1RGs]
    cdef double forwardRate_DaD1RGs_DaD1R_plus_GsaGTP_plus_Gbg_reac = (<double *> params)[ind_forwardRate_DaD1RGs_DaD1R_plus_GsaGTP_plus_Gbg_reac]
    cdef double reverseRate_DaD1RGs_DaD1R_plus_GsaGTP_plus_Gbg_reac = (<double *> params)[ind_reverseRate_DaD1RGs_DaD1R_plus_GsaGTP_plus_Gbg_reac]
    cdef double forwardRate_DaD1RGs_phosp1 = (<double *> params)[ind_forwardRate_DaD1RGs_phosp1]
    cdef double reverseRate_DaD1RGs_phosp1 = (<double *> params)[ind_reverseRate_DaD1RGs_phosp1]
    cdef double forwardRate_DaD1RGs_phosp2 = (<double *> params)[ind_forwardRate_DaD1RGs_phosp2]
    cdef double reverseRate_DaD1RGs_phosp2 = (<double *> params)[ind_reverseRate_DaD1RGs_phosp2]
    cdef double forwardRate_dephosD1R = (<double *> params)[ind_forwardRate_dephosD1R]
    cdef double reverseRate_dephosD1R = (<double *> params)[ind_reverseRate_dephosD1R]
    cdef double forwardRate_Glubuf__Glu_plus_Gbuf = (<double *> params)[ind_forwardRate_Glubuf__Glu_plus_Gbuf]
    cdef double reverseRate_Glubuf__Glu_plus_Gbuf = (<double *> params)[ind_reverseRate_Glubuf__Glu_plus_Gbuf]
    cdef double forwardRate_Glu__GluOut = (<double *> params)[ind_forwardRate_Glu__GluOut]
    cdef double reverseRate_Glu__GluOut = (<double *> params)[ind_reverseRate_Glu__GluOut]
    cdef double forwardRate_GluOut_buf_reac = (<double *> params)[ind_forwardRate_GluOut_buf_reac]
    cdef double reverseRate_GluOut_buf_reac = (<double *> params)[ind_reverseRate_GluOut_buf_reac]
    cdef double forwardRate_Glu_plus_MgluR__GluMgluRdesens = (<double *> params)[ind_forwardRate_Glu_plus_MgluR__GluMgluRdesens]
    cdef double reverseRate_Glu_plus_MgluR__GluMgluRdesens = (<double *> params)[ind_reverseRate_Glu_plus_MgluR__GluMgluRdesens]
    cdef double forwardRate_Glu_plus_MgluR__GluMgluR = (<double *> params)[ind_forwardRate_Glu_plus_MgluR__GluMgluR]
    cdef double reverseRate_Glu_plus_MgluR__GluMgluR = (<double *> params)[ind_reverseRate_Glu_plus_MgluR__GluMgluR]
    cdef double forwardRate_Gqabg_plus_GluMgluR__GluMgluRGq = (<double *> params)[ind_forwardRate_Gqabg_plus_GluMgluR__GluMgluRGq]
    cdef double reverseRate_Gqabg_plus_GluMgluR__GluMgluRGq = (<double *> params)[ind_reverseRate_Gqabg_plus_GluMgluR__GluMgluRGq]
    cdef double forwardRate_Gqabg_plus_MgluR___MgluRGq = (<double *> params)[ind_forwardRate_Gqabg_plus_MgluR___MgluRGq]
    cdef double reverseRate_Gqabg_plus_MgluR___MgluRGq = (<double *> params)[ind_reverseRate_Gqabg_plus_MgluR___MgluRGq]
    cdef double forwardRate_Glu_plus_MgluRGq___GluMgluRGq = (<double *> params)[ind_forwardRate_Glu_plus_MgluRGq___GluMgluRGq]
    cdef double reverseRate_Glu_plus_MgluRGq___GluMgluRGq = (<double *> params)[ind_reverseRate_Glu_plus_MgluRGq___GluMgluRGq]
    cdef double forwardRate_GluMgluRGq__GqaGTP_plus_GluMgluRGbg = (<double *> params)[ind_forwardRate_GluMgluRGq__GqaGTP_plus_GluMgluRGbg]
    cdef double reverseRate_GluMgluRGq__GqaGTP_plus_GluMgluRGbg = (<double *> params)[ind_reverseRate_GluMgluRGq__GqaGTP_plus_GluMgluRGbg]
    cdef double forwardRate_ACh_plus_m1R__AChm1R = (<double *> params)[ind_forwardRate_ACh_plus_m1R__AChm1R]
    cdef double reverseRate_ACh_plus_m1R__AChm1R = (<double *> params)[ind_reverseRate_ACh_plus_m1R__AChm1R]
    cdef double forwardRate_Gqabg_plus_AChm1R__AChm1RGq = (<double *> params)[ind_forwardRate_Gqabg_plus_AChm1R__AChm1RGq]
    cdef double reverseRate_Gqabg_plus_AChm1R__AChm1RGq = (<double *> params)[ind_reverseRate_Gqabg_plus_AChm1R__AChm1RGq]
    cdef double forwardRate_Gqabg_plus_m1R___m1RGq = (<double *> params)[ind_forwardRate_Gqabg_plus_m1R___m1RGq]
    cdef double reverseRate_Gqabg_plus_m1R___m1RGq = (<double *> params)[ind_reverseRate_Gqabg_plus_m1R___m1RGq]
    cdef double forwardRate_ACh_plus_m1RGq___AChm1RGq = (<double *> params)[ind_forwardRate_ACh_plus_m1RGq___AChm1RGq]
    cdef double reverseRate_ACh_plus_m1RGq___AChm1RGq = (<double *> params)[ind_reverseRate_ACh_plus_m1RGq___AChm1RGq]
    cdef double forwardRate_AChm1RGq__GqaGTP_plus_AChm1RGbg = (<double *> params)[ind_forwardRate_AChm1RGq__GqaGTP_plus_AChm1RGbg]
    cdef double reverseRate_AChm1RGq__GqaGTP_plus_AChm1RGbg = (<double *> params)[ind_reverseRate_AChm1RGq__GqaGTP_plus_AChm1RGbg]
    cdef double forwardRate_ACh__null = (<double *> params)[ind_forwardRate_ACh__null]
    cdef double forwardRate_ACh_plus_m4R__AChm4R = (<double *> params)[ind_forwardRate_ACh_plus_m4R__AChm4R]
    cdef double reverseRate_ACh_plus_m4R__AChm4R = (<double *> params)[ind_reverseRate_ACh_plus_m4R__AChm4R]
    cdef double forwardRate_Giabg_plus_AChm4R__AChm4RGi = (<double *> params)[ind_forwardRate_Giabg_plus_AChm4R__AChm4RGi]
    cdef double reverseRate_Giabg_plus_AChm4R__AChm4RGi = (<double *> params)[ind_reverseRate_Giabg_plus_AChm4R__AChm4RGi]
    cdef double forwardRate_Giabg_plus_m4R__Gim4R = (<double *> params)[ind_forwardRate_Giabg_plus_m4R__Gim4R]
    cdef double reverseRate_Giabg_plus_m4R__Gim4R = (<double *> params)[ind_reverseRate_Giabg_plus_m4R__Gim4R]
    cdef double forwardRate_Gim4R_plus_ACh__AChm4RGi = (<double *> params)[ind_forwardRate_Gim4R_plus_ACh__AChm4RGi]
    cdef double reverseRate_Gim4R_plus_ACh__AChm4RGi = (<double *> params)[ind_reverseRate_Gim4R_plus_ACh__AChm4RGi]
    cdef double forwardRate_AChm4RGi__GiaGTP_plus_AChm4R_plus_Gbg = (<double *> params)[ind_forwardRate_AChm4RGi__GiaGTP_plus_AChm4R_plus_Gbg]
    cdef double reverseRate_AChm4RGi__GiaGTP_plus_AChm4R_plus_Gbg = (<double *> params)[ind_reverseRate_AChm4RGi__GiaGTP_plus_AChm4R_plus_Gbg]
    cdef double forwardRate_GsaGTP__GsaGDP_reac = (<double *> params)[ind_forwardRate_GsaGTP__GsaGDP_reac]
    cdef double reverseRate_GsaGTP__GsaGDP_reac = (<double *> params)[ind_reverseRate_GsaGTP__GsaGDP_reac]
    cdef double forwardRate_GsaGDP_plus_Gbg__Gs_reac = (<double *> params)[ind_forwardRate_GsaGDP_plus_Gbg__Gs_reac]
    cdef double reverseRate_GsaGDP_plus_Gbg__Gs_reac = (<double *> params)[ind_reverseRate_GsaGDP_plus_Gbg__Gs_reac]
    cdef double forwardRate_GqaGTP__GqaGDP = (<double *> params)[ind_forwardRate_GqaGTP__GqaGDP]
    cdef double reverseRate_GqaGTP__GqaGDP = (<double *> params)[ind_reverseRate_GqaGTP__GqaGDP]
    cdef double forwardRate_GqaGDP_plus_Gbg__Gqabg = (<double *> params)[ind_forwardRate_GqaGDP_plus_Gbg__Gqabg]
    cdef double reverseRate_GqaGDP_plus_Gbg__Gqabg = (<double *> params)[ind_reverseRate_GqaGDP_plus_Gbg__Gqabg]
    cdef double forwardRate_GiaGTP__GiaGDP = (<double *> params)[ind_forwardRate_GiaGTP__GiaGDP]
    cdef double reverseRate_GiaGTP__GiaGDP = (<double *> params)[ind_reverseRate_GiaGTP__GiaGDP]
    cdef double forwardRate_GiaGDP__Giabg = (<double *> params)[ind_forwardRate_GiaGDP__Giabg]
    cdef double reverseRate_GiaGDP__Giabg = (<double *> params)[ind_reverseRate_GiaGDP__Giabg]
    cdef double forwardRate_GsaGTP_plus_AC5__AC5Gsa_reac = (<double *> params)[ind_forwardRate_GsaGTP_plus_AC5__AC5Gsa_reac]
    cdef double reverseRate_GsaGTP_plus_AC5__AC5Gsa_reac = (<double *> params)[ind_reverseRate_GsaGTP_plus_AC5__AC5Gsa_reac]
    cdef double forwardRate_AC5Gsa_plus_ATP__AC5GsaATP_reac = (<double *> params)[ind_forwardRate_AC5Gsa_plus_ATP__AC5GsaATP_reac]
    cdef double reverseRate_AC5Gsa_plus_ATP__AC5GsaATP_reac = (<double *> params)[ind_reverseRate_AC5Gsa_plus_ATP__AC5GsaATP_reac]
    cdef double forwardRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac = (<double *> params)[ind_forwardRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac]
    cdef double reverseRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac = (<double *> params)[ind_reverseRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac]
    cdef double forwardRate_GiaGTP_AC5_reac = (<double *> params)[ind_forwardRate_GiaGTP_AC5_reac]
    cdef double reverseRate_GiaGTP_AC5_reac = (<double *> params)[ind_reverseRate_GiaGTP_AC5_reac]
    cdef double forwardRate_GiaGTP_AC5GsaGTP_reac = (<double *> params)[ind_forwardRate_GiaGTP_AC5GsaGTP_reac]
    cdef double reverseRate_GiaGTP_AC5GsaGTP_reac = (<double *> params)[ind_reverseRate_GiaGTP_AC5GsaGTP_reac]
    cdef double forwardRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP = (<double *> params)[ind_forwardRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP]
    cdef double reverseRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP = (<double *> params)[ind_reverseRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP]
    cdef double forwardRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP = (<double *> params)[ind_forwardRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP]
    cdef double reverseRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP = (<double *> params)[ind_reverseRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP]
    cdef double forwardRate_AC5GsaGiaATP__AC5Gia_plus_Gs_plus_cAMP = (<double *> params)[ind_forwardRate_AC5GsaGiaATP__AC5Gia_plus_Gs_plus_cAMP]
    cdef double reverseRate_AC5GsaGiaATP__AC5Gia_plus_Gs_plus_cAMP = (<double *> params)[ind_reverseRate_AC5GsaGiaATP__AC5Gia_plus_Gs_plus_cAMP]
    cdef double forwardRate_PDE1_plus_CamCa4__PDE1CamCa4 = (<double *> params)[ind_forwardRate_PDE1_plus_CamCa4__PDE1CamCa4]
    cdef double reverseRate_PDE1_plus_CamCa4__PDE1CamCa4 = (<double *> params)[ind_reverseRate_PDE1_plus_CamCa4__PDE1CamCa4]
    cdef double forwardRate_PDE1CamCa4_plus_cAMP__PDE1CamCa4cAMP = (<double *> params)[ind_forwardRate_PDE1CamCa4_plus_cAMP__PDE1CamCa4cAMP]
    cdef double reverseRate_PDE1CamCa4_plus_cAMP__PDE1CamCa4cAMP = (<double *> params)[ind_reverseRate_PDE1CamCa4_plus_cAMP__PDE1CamCa4cAMP]
    cdef double forwardRate_PDE1CamCa4cAMP__PDE1CamCa4_plus_AMP = (<double *> params)[ind_forwardRate_PDE1CamCa4cAMP__PDE1CamCa4_plus_AMP]
    cdef double reverseRate_PDE1CamCa4cAMP__PDE1CamCa4_plus_AMP = (<double *> params)[ind_reverseRate_PDE1CamCa4cAMP__PDE1CamCa4_plus_AMP]
    cdef double forwardRate_PDE10_plus_cAMP__PDE10cAMP = (<double *> params)[ind_forwardRate_PDE10_plus_cAMP__PDE10cAMP]
    cdef double reverseRate_PDE10_plus_cAMP__PDE10cAMP = (<double *> params)[ind_reverseRate_PDE10_plus_cAMP__PDE10cAMP]
    cdef double forwardRate_PDE10cAMP__PDE10_plus_AMP = (<double *> params)[ind_forwardRate_PDE10cAMP__PDE10_plus_AMP]
    cdef double reverseRate_PDE10cAMP__PDE10_plus_AMP = (<double *> params)[ind_reverseRate_PDE10cAMP__PDE10_plus_AMP]
    cdef double forwardRate_pPDE10_plus_cAMP__pPDE10cAMP = (<double *> params)[ind_forwardRate_pPDE10_plus_cAMP__pPDE10cAMP]
    cdef double reverseRate_pPDE10_plus_cAMP__pPDE10cAMP = (<double *> params)[ind_reverseRate_pPDE10_plus_cAMP__pPDE10cAMP]
    cdef double forwardRate_pPDE10cAMP__pPDE10_plus_AMP = (<double *> params)[ind_forwardRate_pPDE10cAMP__pPDE10_plus_AMP]
    cdef double reverseRate_pPDE10cAMP__pPDE10_plus_AMP = (<double *> params)[ind_reverseRate_pPDE10cAMP__pPDE10_plus_AMP]
    cdef double forwardRate_AMP__ATP = (<double *> params)[ind_forwardRate_AMP__ATP]
    cdef double reverseRate_AMP__ATP = (<double *> params)[ind_reverseRate_AMP__ATP]
    cdef double forwardRate_PDE10_phosp1 = (<double *> params)[ind_forwardRate_PDE10_phosp1]
    cdef double reverseRate_PDE10_phosp1 = (<double *> params)[ind_reverseRate_PDE10_phosp1]
    cdef double forwardRate_PDE10_phosp2 = (<double *> params)[ind_forwardRate_PDE10_phosp2]
    cdef double reverseRate_PDE10_phosp2 = (<double *> params)[ind_reverseRate_PDE10_phosp2]
    cdef double forwardRate_PDE10cAMP_phosp1 = (<double *> params)[ind_forwardRate_PDE10cAMP_phosp1]
    cdef double reverseRate_PDE10cAMP_phosp1 = (<double *> params)[ind_reverseRate_PDE10cAMP_phosp1]
    cdef double forwardRate_PDE10cAMP_phosp2 = (<double *> params)[ind_forwardRate_PDE10cAMP_phosp2]
    cdef double reverseRate_PDE10cAMP_phosp2 = (<double *> params)[ind_reverseRate_PDE10cAMP_phosp2]
    cdef double forwardRate_PDE10_d_phosp1 = (<double *> params)[ind_forwardRate_PDE10_d_phosp1]
    cdef double reverseRate_PDE10_d_phosp1 = (<double *> params)[ind_reverseRate_PDE10_d_phosp1]
    cdef double forwardRate_PDE10_d_phosp2 = (<double *> params)[ind_forwardRate_PDE10_d_phosp2]
    cdef double reverseRate_PDE10_d_phosp2 = (<double *> params)[ind_reverseRate_PDE10_d_phosp2]
    cdef double forwardRate_PDE2_plus_cAMP__PDE2cAMP = (<double *> params)[ind_forwardRate_PDE2_plus_cAMP__PDE2cAMP]
    cdef double reverseRate_PDE2_plus_cAMP__PDE2cAMP = (<double *> params)[ind_reverseRate_PDE2_plus_cAMP__PDE2cAMP]
    cdef double forwardRate_PDE2cAMP_plus_cAMP__PDE2cAMP2 = (<double *> params)[ind_forwardRate_PDE2cAMP_plus_cAMP__PDE2cAMP2]
    cdef double reverseRate_PDE2cAMP_plus_cAMP__PDE2cAMP2 = (<double *> params)[ind_reverseRate_PDE2cAMP_plus_cAMP__PDE2cAMP2]
    cdef double forwardRate_PDE2cAMP2__PDE2cAMP_plus_AMP = (<double *> params)[ind_forwardRate_PDE2cAMP2__PDE2cAMP_plus_AMP]
    cdef double reverseRate_PDE2cAMP2__PDE2cAMP_plus_AMP = (<double *> params)[ind_reverseRate_PDE2cAMP2__PDE2cAMP_plus_AMP]
    cdef double forwardRate_PDE4_enzyme1 = (<double *> params)[ind_forwardRate_PDE4_enzyme1]
    cdef double reverseRate_PDE4_enzyme1 = (<double *> params)[ind_reverseRate_PDE4_enzyme1]
    cdef double forwardRate_PDE4_actvcomp = (<double *> params)[ind_forwardRate_PDE4_actvcomp]
    cdef double reverseRate_PDE4_actvcomp = (<double *> params)[ind_reverseRate_PDE4_actvcomp]
    cdef double forwardRate_pPDE4_enzyme1 = (<double *> params)[ind_forwardRate_pPDE4_enzyme1]
    cdef double reverseRate_pPDE4_enzyme1 = (<double *> params)[ind_reverseRate_pPDE4_enzyme1]
    cdef double forwardRate_pPDE4_actvcomp = (<double *> params)[ind_forwardRate_pPDE4_actvcomp]
    cdef double reverseRate_pPDE4_actvcomp = (<double *> params)[ind_reverseRate_pPDE4_actvcomp]
    cdef double forwardRate_PDE4_phosp1 = (<double *> params)[ind_forwardRate_PDE4_phosp1]
    cdef double reverseRate_PDE4_phosp1 = (<double *> params)[ind_reverseRate_PDE4_phosp1]
    cdef double forwardRate_PDE4_phosp2 = (<double *> params)[ind_forwardRate_PDE4_phosp2]
    cdef double reverseRate_PDE4_phosp2 = (<double *> params)[ind_reverseRate_PDE4_phosp2]
    cdef double forwardRate_PDE4cAMP_phosp1 = (<double *> params)[ind_forwardRate_PDE4cAMP_phosp1]
    cdef double reverseRate_PDE4cAMP_phosp1 = (<double *> params)[ind_reverseRate_PDE4cAMP_phosp1]
    cdef double forwardRate_PDE4cAMP_phosp2 = (<double *> params)[ind_forwardRate_PDE4cAMP_phosp2]
    cdef double reverseRate_PDE4cAMP_phosp2 = (<double *> params)[ind_reverseRate_PDE4cAMP_phosp2]
    cdef double forwardRate_PDE4_d_phosp1 = (<double *> params)[ind_forwardRate_PDE4_d_phosp1]
    cdef double reverseRate_PDE4_d_phosp1 = (<double *> params)[ind_reverseRate_PDE4_d_phosp1]
    cdef double forwardRate_PDE4_d_phosp2 = (<double *> params)[ind_forwardRate_PDE4_d_phosp2]
    cdef double reverseRate_PDE4_d_phosp2 = (<double *> params)[ind_reverseRate_PDE4_d_phosp2]
    cdef double forwardRate_Plc_plus_Ca__PlcCa = (<double *> params)[ind_forwardRate_Plc_plus_Ca__PlcCa]
    cdef double reverseRate_Plc_plus_Ca__PlcCa = (<double *> params)[ind_reverseRate_Plc_plus_Ca__PlcCa]
    cdef double forwardRate_PlcCa_plus_GqaGTP__PlcCaGqa = (<double *> params)[ind_forwardRate_PlcCa_plus_GqaGTP__PlcCaGqa]
    cdef double reverseRate_PlcCa_plus_GqaGTP__PlcCaGqa = (<double *> params)[ind_reverseRate_PlcCa_plus_GqaGTP__PlcCaGqa]
    cdef double forwardRate_Plc_plus_GqaGTP__PlcGqa = (<double *> params)[ind_forwardRate_Plc_plus_GqaGTP__PlcGqa]
    cdef double reverseRate_Plc_plus_GqaGTP__PlcGqa = (<double *> params)[ind_reverseRate_Plc_plus_GqaGTP__PlcGqa]
    cdef double forwardRate_PlcGaGTP_plus_Ca__PlcCaGqa = (<double *> params)[ind_forwardRate_PlcGaGTP_plus_Ca__PlcCaGqa]
    cdef double reverseRate_PlcGaGTP_plus_Ca__PlcCaGqa = (<double *> params)[ind_reverseRate_PlcGaGTP_plus_Ca__PlcCaGqa]
    cdef double forwardRate_PlcCa_plus_Pip2__PlcCaPip2 = (<double *> params)[ind_forwardRate_PlcCa_plus_Pip2__PlcCaPip2]
    cdef double reverseRate_PlcCa_plus_Pip2__PlcCaPip2 = (<double *> params)[ind_reverseRate_PlcCa_plus_Pip2__PlcCaPip2]
    cdef double forwardRate_PlcCaPip2__PlcCa_plus_Dag_plus_Ip3 = (<double *> params)[ind_forwardRate_PlcCaPip2__PlcCa_plus_Dag_plus_Ip3]
    cdef double reverseRate_PlcCaPip2__PlcCa_plus_Dag_plus_Ip3 = (<double *> params)[ind_reverseRate_PlcCaPip2__PlcCa_plus_Dag_plus_Ip3]
    cdef double forwardRate_PlcCaGqa_plus_Pip2__PlcCaGqaPip2 = (<double *> params)[ind_forwardRate_PlcCaGqa_plus_Pip2__PlcCaGqaPip2]
    cdef double reverseRate_PlcCaGqa_plus_Pip2__PlcCaGqaPip2 = (<double *> params)[ind_reverseRate_PlcCaGqa_plus_Pip2__PlcCaGqaPip2]
    cdef double forwardRate_PlcCaGqaPip2__PlcCaGqaDag_plus_Ip3 = (<double *> params)[ind_forwardRate_PlcCaGqaPip2__PlcCaGqaDag_plus_Ip3]
    cdef double reverseRate_PlcCaGqaPip2__PlcCaGqaDag_plus_Ip3 = (<double *> params)[ind_reverseRate_PlcCaGqaPip2__PlcCaGqaDag_plus_Ip3]
    cdef double forwardRate_PlcGqa__Plc_plus_GqaGDP = (<double *> params)[ind_forwardRate_PlcGqa__Plc_plus_GqaGDP]
    cdef double reverseRate_PlcGqa__Plc_plus_GqaGDP = (<double *> params)[ind_reverseRate_PlcGqa__Plc_plus_GqaGDP]
    cdef double forwardRate_PlcCaGqa__PlcCa_plus_GaGDP = (<double *> params)[ind_forwardRate_PlcCaGqa__PlcCa_plus_GaGDP]
    cdef double reverseRate_PlcCaGqa__PlcCa_plus_GaGDP = (<double *> params)[ind_reverseRate_PlcCaGqa__PlcCa_plus_GaGDP]
    cdef double forwardRate_Ca_plus_Dlg__CaDgl = (<double *> params)[ind_forwardRate_Ca_plus_Dlg__CaDgl]
    cdef double reverseRate_Ca_plus_Dlg__CaDgl = (<double *> params)[ind_reverseRate_Ca_plus_Dlg__CaDgl]
    cdef double forwardRate_Dag_plus_CaDlg__DagCaDgl = (<double *> params)[ind_forwardRate_Dag_plus_CaDlg__DagCaDgl]
    cdef double reverseRate_Dag_plus_CaDlg__DagCaDgl = (<double *> params)[ind_reverseRate_Dag_plus_CaDlg__DagCaDgl]
    cdef double forwardRate_DagCaDgl__CaDgl_plus_2ag = (<double *> params)[ind_forwardRate_DagCaDgl__CaDgl_plus_2ag]
    cdef double reverseRate_DagCaDgl__CaDgl_plus_2ag = (<double *> params)[ind_reverseRate_DagCaDgl__CaDgl_plus_2ag]
    cdef double forwardRate_Ip3__Ip3degrad = (<double *> params)[ind_forwardRate_Ip3__Ip3degrad]
    cdef double reverseRate_Ip3__Ip3degrad = (<double *> params)[ind_reverseRate_Ip3__Ip3degrad]
    cdef double forwardRate_Ip3degrad_plus_PIkinase__Ip3degPIk = (<double *> params)[ind_forwardRate_Ip3degrad_plus_PIkinase__Ip3degPIk]
    cdef double reverseRate_Ip3degrad_plus_PIkinase__Ip3degPIk = (<double *> params)[ind_reverseRate_Ip3degrad_plus_PIkinase__Ip3degPIk]
    cdef double forwardRate_Ip3degPIk__PIP2__plus__PIkinase = (<double *> params)[ind_forwardRate_Ip3degPIk__PIP2__plus__PIkinase]
    cdef double reverseRate_Ip3degPIk__PIP2__plus__PIkinase = (<double *> params)[ind_reverseRate_Ip3degPIk__PIP2__plus__PIkinase]
    cdef double forwardRate_2ag__2agDegrad = (<double *> params)[ind_forwardRate_2ag__2agDegrad]
    cdef double reverseRate_2ag__2agDegrad = (<double *> params)[ind_reverseRate_2ag__2agDegrad]
    cdef double forwardRate_DagKdag1 = (<double *> params)[ind_forwardRate_DagKdag1]
    cdef double reverseRate_DagKdag1 = (<double *> params)[ind_reverseRate_DagKdag1]
    cdef double forwardRate_DagKdag1_1 = (<double *> params)[ind_forwardRate_DagKdag1_1]
    cdef double reverseRate_DagKdag1_1 = (<double *> params)[ind_reverseRate_DagKdag1_1]
    cdef double forwardRate_Pkc_plus_Ca__PkcCa = (<double *> params)[ind_forwardRate_Pkc_plus_Ca__PkcCa]
    cdef double reverseRate_Pkc_plus_Ca__PkcCa = (<double *> params)[ind_reverseRate_Pkc_plus_Ca__PkcCa]
    cdef double forwardRate_PkcCa_plus_Dag__PkcCaDag = (<double *> params)[ind_forwardRate_PkcCa_plus_Dag__PkcCaDag]
    cdef double reverseRate_PkcCa_plus_Dag__PkcCaDag = (<double *> params)[ind_reverseRate_PkcCa_plus_Dag__PkcCaDag]
    cdef double forwardRate_PKA_bind = (<double *> params)[ind_forwardRate_PKA_bind]
    cdef double reverseRate_PKA_bind = (<double *> params)[ind_reverseRate_PKA_bind]
    cdef double forwardRate_PKAcAMP2_bind = (<double *> params)[ind_forwardRate_PKAcAMP2_bind]
    cdef double reverseRate_PKAcAMP2_bind = (<double *> params)[ind_reverseRate_PKAcAMP2_bind]
    cdef double forwardRate_PKAcAMP4_diss = (<double *> params)[ind_forwardRate_PKAcAMP4_diss]
    cdef double reverseRate_PKAcAMP4_diss = (<double *> params)[ind_reverseRate_PKAcAMP4_diss]
    cdef double forwardRate_Cam_plus_PP2B__PP2BCam_reac = (<double *> params)[ind_forwardRate_Cam_plus_PP2B__PP2BCam_reac]
    cdef double reverseRate_Cam_plus_PP2B__PP2BCam_reac = (<double *> params)[ind_reverseRate_Cam_plus_PP2B__PP2BCam_reac]
    cdef double forwardRate_CamCa2C_plus_PP2B__PP2BCamCa2C_reac = (<double *> params)[ind_forwardRate_CamCa2C_plus_PP2B__PP2BCamCa2C_reac]
    cdef double reverseRate_CamCa2C_plus_PP2B__PP2BCamCa2C_reac = (<double *> params)[ind_reverseRate_CamCa2C_plus_PP2B__PP2BCamCa2C_reac]
    cdef double forwardRate_CamCa2N_plus_PP2B__PP2BCamCa2N_reac = (<double *> params)[ind_forwardRate_CamCa2N_plus_PP2B__PP2BCamCa2N_reac]
    cdef double reverseRate_CamCa2N_plus_PP2B__PP2BCamCa2N_reac = (<double *> params)[ind_reverseRate_CamCa2N_plus_PP2B__PP2BCamCa2N_reac]
    cdef double forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac = (<double *> params)[ind_forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac]
    cdef double reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac = (<double *> params)[ind_reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac]
    cdef double forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac = (<double *> params)[ind_forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac]
    cdef double reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac = (<double *> params)[ind_reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac]
    cdef double forwardRate_CamCa4_plus_PP2B__PP2BCamCa4_reac = (<double *> params)[ind_forwardRate_CamCa4_plus_PP2B__PP2BCamCa4_reac]
    cdef double reverseRate_CamCa4_plus_PP2B__PP2BCamCa4_reac = (<double *> params)[ind_reverseRate_CamCa4_plus_PP2B__PP2BCamCa4_reac]
    cdef double forwardRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac = (<double *> params)[ind_forwardRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac]
    cdef double reverseRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac = (<double *> params)[ind_reverseRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac]
    cdef double forwardRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac = (<double *> params)[ind_forwardRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac]
    cdef double reverseRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac = (<double *> params)[ind_reverseRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac]
    cdef double forwardRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac = (<double *> params)[ind_forwardRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac]
    cdef double reverseRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac = (<double *> params)[ind_reverseRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac]
    cdef double forwardRate_PKAcPP2AB56d__PKAc_plus_pPP2A_reac = (<double *> params)[ind_forwardRate_PKAcPP2AB56d__PKAc_plus_pPP2A_reac]
    cdef double reverseRate_PKAcPP2AB56d__PKAc_plus_pPP2A_reac = (<double *> params)[ind_reverseRate_PKAcPP2AB56d__PKAc_plus_pPP2A_reac]
    cdef double forwardRate_pPP2A__PP2A_reac = (<double *> params)[ind_forwardRate_pPP2A__PP2A_reac]
    cdef double reverseRate_pPP2A__PP2A_reac = (<double *> params)[ind_reverseRate_pPP2A__PP2A_reac]
    cdef double forwardRate_PP2ABPR72_plus_Ca__PP2Acal_reac = (<double *> params)[ind_forwardRate_PP2ABPR72_plus_Ca__PP2Acal_reac]
    cdef double reverseRate_PP2ABPR72_plus_Ca__PP2Acal_reac = (<double *> params)[ind_reverseRate_PP2ABPR72_plus_Ca__PP2Acal_reac]
    cdef double forwardRate_D32_plus_PKAc__D32PKAc_reac = (<double *> params)[ind_forwardRate_D32_plus_PKAc__D32PKAc_reac]
    cdef double reverseRate_D32_plus_PKAc__D32PKAc_reac = (<double *> params)[ind_reverseRate_D32_plus_PKAc__D32PKAc_reac]
    cdef double forwardRate_D32PKAc__PKAc_plus_D32p34_reac = (<double *> params)[ind_forwardRate_D32PKAc__PKAc_plus_D32p34_reac]
    cdef double reverseRate_D32PKAc__PKAc_plus_D32p34_reac = (<double *> params)[ind_reverseRate_D32PKAc__PKAc_plus_D32p34_reac]
    cdef double forwardRate_D32p34_plus_PP1__D32p34pp1_reac = (<double *> params)[ind_forwardRate_D32p34_plus_PP1__D32p34pp1_reac]
    cdef double reverseRate_D32p34_plus_PP1__D32p34pp1_reac = (<double *> params)[ind_reverseRate_D32p34_plus_PP1__D32p34pp1_reac]
    cdef double forwardRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac = (<double *> params)[ind_forwardRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac]
    cdef double reverseRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac = (<double *> params)[ind_reverseRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac]
    cdef double forwardRate_D32p34PP2BCamCa4__PP2BCamCa4_plus_D32_reac = (<double *> params)[ind_forwardRate_D32p34PP2BCamCa4__PP2BCamCa4_plus_D32_reac]
    cdef double reverseRate_D32p34PP2BCamCa4__PP2BCamCa4_plus_D32_reac = (<double *> params)[ind_reverseRate_D32p34PP2BCamCa4__PP2BCamCa4_plus_D32_reac]
    cdef double forwardRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac = (<double *> params)[ind_forwardRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac]
    cdef double reverseRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac = (<double *> params)[ind_reverseRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac]
    cdef double forwardRate_D32p34PP1PP2BCamCa4__PP1_plus_PP2BCamCa4_plus_D32_reac = (<double *> params)[ind_forwardRate_D32p34PP1PP2BCamCa4__PP1_plus_PP2BCamCa4_plus_D32_reac]
    cdef double reverseRate_D32p34PP1PP2BCamCa4__PP1_plus_PP2BCamCa4_plus_D32_reac = (<double *> params)[ind_reverseRate_D32p34PP1PP2BCamCa4__PP1_plus_PP2BCamCa4_plus_D32_reac]
    cdef double forwardRate_D32p34_plus_PP2ABPR72__D32p34PP2ABPR72_reac = (<double *> params)[ind_forwardRate_D32p34_plus_PP2ABPR72__D32p34PP2ABPR72_reac]
    cdef double reverseRate_D32p34_plus_PP2ABPR72__D32p34PP2ABPR72_reac = (<double *> params)[ind_reverseRate_D32p34_plus_PP2ABPR72__D32p34PP2ABPR72_reac]
    cdef double forwardRate_D32p34PP2ABPR72__PP2ABPR72_plus_D32_reac = (<double *> params)[ind_forwardRate_D32p34PP2ABPR72__PP2ABPR72_plus_D32_reac]
    cdef double reverseRate_D32p34PP2ABPR72__PP2ABPR72_plus_D32_reac = (<double *> params)[ind_reverseRate_D32p34PP2ABPR72__PP2ABPR72_plus_D32_reac]
    cdef double forwardRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac = (<double *> params)[ind_forwardRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac]
    cdef double reverseRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac = (<double *> params)[ind_reverseRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac]
    cdef double forwardRate_D32p34PP2AB56d__PP2AB56d_plus_D32_reac = (<double *> params)[ind_forwardRate_D32p34PP2AB56d__PP2AB56d_plus_D32_reac]
    cdef double reverseRate_D32p34PP2AB56d__PP2AB56d_plus_D32_reac = (<double *> params)[ind_reverseRate_D32p34PP2AB56d__PP2AB56d_plus_D32_reac]
    cdef double forwardRate_D32p34pp1_plus_PP2ABPR72__D32p34PP1PP2ABPR72_reac = (<double *> params)[ind_forwardRate_D32p34pp1_plus_PP2ABPR72__D32p34PP1PP2ABPR72_reac]
    cdef double reverseRate_D32p34pp1_plus_PP2ABPR72__D32p34PP1PP2ABPR72_reac = (<double *> params)[ind_reverseRate_D32p34pp1_plus_PP2ABPR72__D32p34PP1PP2ABPR72_reac]
    cdef double forwardRate_D32p34PP1PP2ABPR72__PP1_plus_PP2ABPR72_plus_D32_reac = (<double *> params)[ind_forwardRate_D32p34PP1PP2ABPR72__PP1_plus_PP2ABPR72_plus_D32_reac]
    cdef double reverseRate_D32p34PP1PP2ABPR72__PP1_plus_PP2ABPR72_plus_D32_reac = (<double *> params)[ind_reverseRate_D32p34PP1PP2ABPR72__PP1_plus_PP2ABPR72_plus_D32_reac]
    cdef double forwardRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac = (<double *> params)[ind_forwardRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac]
    cdef double reverseRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac = (<double *> params)[ind_reverseRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac]
    cdef double forwardRate_D32p34PP1PP2AB56d__PP1_plus_PP2AB56d_plus_D32_reac = (<double *> params)[ind_forwardRate_D32p34PP1PP2AB56d__PP1_plus_PP2AB56d_plus_D32_reac]
    cdef double reverseRate_D32p34PP1PP2AB56d__PP1_plus_PP2AB56d_plus_D32_reac = (<double *> params)[ind_reverseRate_D32p34PP1PP2AB56d__PP1_plus_PP2AB56d_plus_D32_reac]
    cdef double forwardRate_Cdk5_plus_D32__Cdk5D32_reac = (<double *> params)[ind_forwardRate_Cdk5_plus_D32__Cdk5D32_reac]
    cdef double reverseRate_Cdk5_plus_D32__Cdk5D32_reac = (<double *> params)[ind_reverseRate_Cdk5_plus_D32__Cdk5D32_reac]
    cdef double forwardRate_Cdk5D32__D32p75_plus_Cdk5_reac = (<double *> params)[ind_forwardRate_Cdk5D32__D32p75_plus_Cdk5_reac]
    cdef double reverseRate_Cdk5D32__D32p75_plus_Cdk5_reac = (<double *> params)[ind_reverseRate_Cdk5D32__D32p75_plus_Cdk5_reac]
    cdef double forwardRate_D32p75_plus_PKAc__D32p75PKAc_reac = (<double *> params)[ind_forwardRate_D32p75_plus_PKAc__D32p75PKAc_reac]
    cdef double reverseRate_D32p75_plus_PKAc__D32p75PKAc_reac = (<double *> params)[ind_reverseRate_D32p75_plus_PKAc__D32p75PKAc_reac]
    cdef double forwardRate_D32p75_plus_pPP2A__D32p75pPP2A_reac = (<double *> params)[ind_forwardRate_D32p75_plus_pPP2A__D32p75pPP2A_reac]
    cdef double reverseRate_D32p75_plus_pPP2A__D32p75pPP2A_reac = (<double *> params)[ind_reverseRate_D32p75_plus_pPP2A__D32p75pPP2A_reac]
    cdef double forwardRate_D32p75pPP2A__D32_plus_pPP2A_reac = (<double *> params)[ind_forwardRate_D32p75pPP2A__D32_plus_pPP2A_reac]
    cdef double reverseRate_D32p75pPP2A__D32_plus_pPP2A_reac = (<double *> params)[ind_reverseRate_D32p75pPP2A__D32_plus_pPP2A_reac]
    cdef double forwardRate_D32p75_plus_PP2ABPR72__D32p75PP2ABPR72_reac = (<double *> params)[ind_forwardRate_D32p75_plus_PP2ABPR72__D32p75PP2ABPR72_reac]
    cdef double reverseRate_D32p75_plus_PP2ABPR72__D32p75PP2ABPR72_reac = (<double *> params)[ind_reverseRate_D32p75_plus_PP2ABPR72__D32p75PP2ABPR72_reac]
    cdef double forwardRate_D32p75PP2ABPR72__D32_plus_PP2ABPR72_reac = (<double *> params)[ind_forwardRate_D32p75PP2ABPR72__D32_plus_PP2ABPR72_reac]
    cdef double reverseRate_D32p75PP2ABPR72__D32_plus_PP2ABPR72_reac = (<double *> params)[ind_reverseRate_D32p75PP2ABPR72__D32_plus_PP2ABPR72_reac]
    cdef double forwardRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac = (<double *> params)[ind_forwardRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac]
    cdef double reverseRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac = (<double *> params)[ind_reverseRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac]
    cdef double forwardRate_D32p75PP2AB56d__D32_plus_PP2AB56d_reac = (<double *> params)[ind_forwardRate_D32p75PP2AB56d__D32_plus_PP2AB56d_reac]
    cdef double reverseRate_D32p75PP2AB56d__D32_plus_PP2AB56d_reac = (<double *> params)[ind_reverseRate_D32p75PP2AB56d__D32_plus_PP2AB56d_reac]
    cdef double forwardRate_D32p75_plus_PP2Acal__D32p75PP2Acal_reac = (<double *> params)[ind_forwardRate_D32p75_plus_PP2Acal__D32p75PP2Acal_reac]
    cdef double reverseRate_D32p75_plus_PP2Acal__D32p75PP2Acal_reac = (<double *> params)[ind_reverseRate_D32p75_plus_PP2Acal__D32p75PP2Acal_reac]
    cdef double forwardRate_D32p75PP2Acal_D32_plus_PP2Acal_reac = (<double *> params)[ind_forwardRate_D32p75PP2Acal_D32_plus_PP2Acal_reac]
    cdef double reverseRate_D32p75PP2Acal_D32_plus_PP2Acal_reac = (<double *> params)[ind_reverseRate_D32p75PP2Acal_D32_plus_PP2Acal_reac]
    cdef double forwardRate_CK_bind = (<double *> params)[ind_forwardRate_CK_bind]
    cdef double reverseRate_CK_bind = (<double *> params)[ind_reverseRate_CK_bind]
    cdef double forwardRate_CKCam_bind = (<double *> params)[ind_forwardRate_CKCam_bind]
    cdef double reverseRate_CKCam_bind = (<double *> params)[ind_reverseRate_CKCam_bind]
    cdef double forwardRate_CKCam_bind_1 = (<double *> params)[ind_forwardRate_CKCam_bind_1]
    cdef double reverseRate_CKCam_bind_1 = (<double *> params)[ind_reverseRate_CKCam_bind_1]
    cdef double forwardRate_CKCam_bind_2 = (<double *> params)[ind_forwardRate_CKCam_bind_2]
    cdef double reverseRate_CKCam_bind_2 = (<double *> params)[ind_reverseRate_CKCam_bind_2]
    cdef double forwardRate_CKp_auto = (<double *> params)[ind_forwardRate_CKp_auto]
    cdef double reverseRate_CKp_auto = (<double *> params)[ind_reverseRate_CKp_auto]
    cdef double forwardRate_CKp_bind = (<double *> params)[ind_forwardRate_CKp_bind]
    cdef double reverseRate_CKp_bind = (<double *> params)[ind_reverseRate_CKp_bind]
    cdef double forwardRate_CKpPP1_reac = (<double *> params)[ind_forwardRate_CKpPP1_reac]
    cdef double reverseRate_CKpPP1_reac = (<double *> params)[ind_reverseRate_CKpPP1_reac]
    cdef double forwardRate_CKpCamCa4PP1_bind = (<double *> params)[ind_forwardRate_CKpCamCa4PP1_bind]
    cdef double reverseRate_CKpCamCa4PP1_bind = (<double *> params)[ind_reverseRate_CKpCamCa4PP1_bind]
    cdef double forwardRate_CKpCamCa4PP1_reac = (<double *> params)[ind_forwardRate_CKpCamCa4PP1_reac]
    cdef double reverseRate_CKpCamCa4PP1_reac = (<double *> params)[ind_reverseRate_CKpCamCa4PP1_reac]
    cdef double forwardRate_Dgl_CKCamCa4_bind = (<double *> params)[ind_forwardRate_Dgl_CKCamCa4_bind]
    cdef double reverseRate_Dgl_CKCamCa4_bind = (<double *> params)[ind_reverseRate_Dgl_CKCamCa4_bind]
    cdef double forwardRate_Dgl_CKCam_diss = (<double *> params)[ind_forwardRate_Dgl_CKCam_diss]
    cdef double reverseRate_Dgl_CKCam_diss = (<double *> params)[ind_reverseRate_Dgl_CKCam_diss]
    cdef double forwardRate_Dgl_CKpCamCa4_bind = (<double *> params)[ind_forwardRate_Dgl_CKpCamCa4_bind]
    cdef double reverseRate_Dgl_CKpCamCa4_bind = (<double *> params)[ind_reverseRate_Dgl_CKpCamCa4_bind]
    cdef double forwardRate_Dgl_CKpCamCa4_diss = (<double *> params)[ind_forwardRate_Dgl_CKpCamCa4_diss]
    cdef double reverseRate_Dgl_CKpCamCa4_diss = (<double *> params)[ind_reverseRate_Dgl_CKpCamCa4_diss]
    cdef double forwardRate_CaDgl_CKpCamCa4_bind = (<double *> params)[ind_forwardRate_CaDgl_CKpCamCa4_bind]
    cdef double reverseRate_CaDgl_CKpCamCa4_bind = (<double *> params)[ind_reverseRate_CaDgl_CKpCamCa4_bind]
    cdef double forwardRate_CaDgl_CKpCamCa4_diss = (<double *> params)[ind_forwardRate_CaDgl_CKpCamCa4_diss]
    cdef double reverseRate_CaDgl_CKpCamCa4_diss = (<double *> params)[ind_reverseRate_CaDgl_CKpCamCa4_diss]
    cdef double forwardRate_Dgl_CKp_bind = (<double *> params)[ind_forwardRate_Dgl_CKp_bind]
    cdef double reverseRate_Dgl_CKp_bind = (<double *> params)[ind_reverseRate_Dgl_CKp_bind]
    cdef double forwardRate_Dgl_CKp_diss = (<double *> params)[ind_forwardRate_Dgl_CKp_diss]
    cdef double reverseRate_Dgl_CKp_diss = (<double *> params)[ind_reverseRate_Dgl_CKp_diss]
    cdef double forwardRate_pDgl_dephos = (<double *> params)[ind_forwardRate_pDgl_dephos]
    cdef double reverseRate_pDgl_dephos = (<double *> params)[ind_reverseRate_pDgl_dephos]
    cdef double forwardRate_Ca_plus_pDlg__CapDgl = (<double *> params)[ind_forwardRate_Ca_plus_pDlg__CapDgl]
    cdef double reverseRate_Ca_plus_pDlg__CapDgl = (<double *> params)[ind_reverseRate_Ca_plus_pDlg__CapDgl]
    cdef double forwardRate_Dag_plus_CapDlg__DagCapDgl = (<double *> params)[ind_forwardRate_Dag_plus_CapDlg__DagCapDgl]
    cdef double reverseRate_Dag_plus_CapDlg__DagCapDgl = (<double *> params)[ind_reverseRate_Dag_plus_CapDlg__DagCapDgl]
    cdef double forwardRate_DagCapDgl__CapDgl_plus_2ag = (<double *> params)[ind_forwardRate_DagCapDgl__CapDgl_plus_2ag]
    cdef double reverseRate_DagCapDgl__CapDgl_plus_2ag = (<double *> params)[ind_reverseRate_DagCapDgl__CapDgl_plus_2ag]
    cdef double forwardRate_AKAR3_plus_PKAc_bind = (<double *> params)[ind_forwardRate_AKAR3_plus_PKAc_bind]
    cdef double reverseRate_AKAR3_plus_PKAc_bind = (<double *> params)[ind_reverseRate_AKAR3_plus_PKAc_bind]
    cdef double forwardRate_PKAcAKAR3_phos = (<double *> params)[ind_forwardRate_PKAcAKAR3_phos]
    cdef double reverseRate_PKAcAKAR3_phos = (<double *> params)[ind_reverseRate_PKAcAKAR3_phos]
    cdef double forwardRate_pAKAR3_plus_PP1_bind = (<double *> params)[ind_forwardRate_pAKAR3_plus_PP1_bind]
    cdef double reverseRate_pAKAR3_plus_PP1_bind = (<double *> params)[ind_reverseRate_pAKAR3_plus_PP1_bind]
    cdef double forwardRate_PP1pAKAR3_dephos = (<double *> params)[ind_forwardRate_PP1pAKAR3_dephos]
    cdef double reverseRate_PP1pAKAR3_dephos = (<double *> params)[ind_reverseRate_PP1pAKAR3_dephos]
    cdef double forwardRate_GsaGTP_AC1_reac = (<double *> params)[ind_forwardRate_GsaGTP_AC1_reac]
    cdef double reverseRate_GsaGTP_AC1_reac = (<double *> params)[ind_reverseRate_GsaGTP_AC1_reac]
    cdef double forwardRate_AC1GsaGTP_CamCa4_Reac = (<double *> params)[ind_forwardRate_AC1GsaGTP_CamCa4_Reac]
    cdef double reverseRate_AC1GsaGTP_CamCa4_Reac = (<double *> params)[ind_reverseRate_AC1GsaGTP_CamCa4_Reac]
    cdef double forwardRate_AC1GsaGTPCamCa4_ATP_Reac = (<double *> params)[ind_forwardRate_AC1GsaGTPCamCa4_ATP_Reac]
    cdef double reverseRate_AC1GsaGTPCamCa4_ATP_Reac = (<double *> params)[ind_reverseRate_AC1GsaGTPCamCa4_ATP_Reac]
    cdef double forwardRate_AC1GsaGTPCamCa4ATP_diss = (<double *> params)[ind_forwardRate_AC1GsaGTPCamCa4ATP_diss]
    cdef double reverseRate_AC1GsaGTPCamCa4ATP_diss = (<double *> params)[ind_reverseRate_AC1GsaGTPCamCa4ATP_diss]
    cdef double forwardRate_AC1CamCa4_Reac = (<double *> params)[ind_forwardRate_AC1CamCa4_Reac]
    cdef double reverseRate_AC1CamCa4_Reac = (<double *> params)[ind_reverseRate_AC1CamCa4_Reac]
    cdef double forwardRate_AC1CamCa4ATP_bind = (<double *> params)[ind_forwardRate_AC1CamCa4ATP_bind]
    cdef double reverseRate_AC1CamCa4ATP_bind = (<double *> params)[ind_reverseRate_AC1CamCa4ATP_bind]
    cdef double forwardRate_AC1CamCa4ATP_disso = (<double *> params)[ind_forwardRate_AC1CamCa4ATP_disso]
    cdef double reverseRate_AC1CamCa4ATP_disso = (<double *> params)[ind_reverseRate_AC1CamCa4ATP_disso]



    # ---- Stim Ilya model ---- #
    cdef double xst_LTPwin =(<double *> params)[ind_xst_LTPwin]
    cdef double xstep_LTPwin =(<double *> params)[ind_xstep_LTPwin]
    cdef double xst_caL13 =(<double *> params)[ind_xst_caL13]
    cdef double xstep_caL13 =(<double *> params)[ind_xstep_caL13]
    cdef double xst_NMDA =(<double *> params)[ind_xst_NMDA]
    cdef double xstep_NMDA =(<double *> params)[ind_xstep_NMDA]
    cdef double nb_Glu_stim =(<double *> params)[ind_nb_Glu_stim]
    cdef double nb_Iact_stim =(<double *> params)[ind_nb_Iact_stim]
    cdef double n_x_LTPwin =(<double *> params)[ind_n_x_LTPwin]
    cdef double n_x_caL13 =(<double *> params)[ind_n_x_caL13]
    cdef double n_x_NMDA =(<double *> params)[ind_n_x_NMDA]

    cdef int int_n_x_LTPwin = int(n_x_LTPwin)
    cdef int int_n_x_caL13 = int(n_x_caL13)
    cdef int int_n_x_NMDA = int(n_x_NMDA)
    cdef int int_nb_Glu_stim = int(nb_Glu_stim)
    cdef int int_nb_Iact_stim = int(nb_Iact_stim)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # ----------- INPUT ----------- #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # -------------- Glutamate ----------------- #
    cdef double Glu_stim = 0.
    cdef int ii
    cdef int ind_param_Glumax, ind_param_tauglu, ind_param_t_start_glu

    cdef double Glumax_temp, tauglu_temp, t_start_glu_temp

    if pre_on:
        for ii from 0 <= ii < int_nb_Glu_stim:
            ind_param_Glumax = int(ii+1 +ind_n_x_NMDA +int_n_x_LTPwin +3*int_n_x_caL13 +int_n_x_NMDA)
            ind_param_tauglu = int(ind_param_Glumax +int_nb_Glu_stim)
            ind_param_t_start_glu = int(ind_param_tauglu +int_nb_Glu_stim)

            Glumax_temp = (<double *> params)[ind_param_Glumax]
            tauglu_temp = (<double *> params)[ind_param_tauglu]
            t_start_glu_temp = (<double *> params)[ind_param_t_start_glu]

            if t -t_start_glu_temp >0.:
                if t  <= t_start_glu_temp +tauglu_temp *30.:
                    Glu_stim += Glumax_temp *exp(-(t -t_start_glu_temp)/tauglu_temp)

    # -------------- I action post ----------------- #
    cdef double Iact_stim = 0.
    cdef double AP_term_temp = 0.
    cdef int jj
    cdef int ind_param_DPmax, ind_param_APmax, ind_param_tausbAP, ind_param_t_start_Iact

    cdef double DPmax_temp, APmax_temp, tausbAP_temp, t_start_Iact_temp

    if post_on:

        for jj from 0 <= jj < int_nb_Iact_stim:
            ind_param_DPmax = int(jj+1 +ind_n_x_NMDA +int_n_x_LTPwin +3*int_n_x_caL13 +int_n_x_NMDA +3*int_nb_Glu_stim)
            ind_param_APmax = int(ind_param_DPmax +int_nb_Iact_stim)
            ind_param_tausbAP = int(ind_param_APmax +int_nb_Iact_stim)
            ind_param_t_start_Iact = int(ind_param_tausbAP +int_nb_Iact_stim)

            DPmax_temp = (<double *> params)[ind_param_DPmax]
            APmax_temp = (<double *> params)[ind_param_APmax]
            tausbAP_temp = (<double *> params)[ind_param_tausbAP]
            t_start_Iact_temp = (<double *> params)[ind_param_t_start_Iact]

            AP_term_temp = 0.

            if (t -t_start_Iact_temp) > 0.:
                if t <= t_start_Iact_temp +APdur:
                    if t -t_start_Iact_temp -tsdt > 0.:
                        AP_term_temp = -APmax_temp *exp(-(t -t_start_Iact_temp -tsdt)/tausbAP_temp)

                    Iact_stim += -DPmax_temp +AP_term_temp

    cdef double Glu = BaseLevel
    cdef double I_action = 0.

    # ---------------- DA ----------------- #
    cdef double DA_input = input_ampl_DA*(1.0 +copysign(1.0, t%interv_stim_ms_DA -t_stim_on_DA))/2. *(1.0 +copysign(1.0, t_stim_off_DA -t%interv_stim_ms_DA))/2.

    # ---------------- Ca ----------------- #
    cdef double Ca_input = input_ampl_Ca*(1.0 +copysign(1.0, t%interv_stim_ms_Ca -t_stim_on_Ca))/2. *(1.0 +copysign(1.0, t_stim_off_Ca -t%interv_stim_ms_Ca))/2.

    # ------------ Bounds control --------------- # -> ON DIRAIT QUE CA MARCHE ??? #
    if y[Ca_cyt] <0:
        y[Ca_cyt] = 0
        printf('caa')

    # ---------- STIM --------------- #
    if pre_on:
        Glu += Glu_stim

    if post_on:
        I_action += Iact_stim


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # ----------- AUXILIARY FUNCTIONS ----------- #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    cdef double ctrl1 = 0.
    cdef double ctrl2 = 0.

    if on_ECb:
        if ECb_CB1R_on:
            ctrl1 = kCB1R *y[o_CB1R] +gamma1DA *y[Da] # No unit
            ctrl2 = kCB1R *y[o_CB1R] +gamma2DA *y[Da] # No unit

        else:
            ctrl1 = alphatwoAGCB1*y[two_ag] +betaAEACB1*alphaAEACB1*y[AEA] +gamma1DA *y[Da] # No unit
            ctrl2 = alphatwoAGCB1*y[two_ag] +betaAEACB1*alphaAEACB1*y[AEA] +gamma2DA *y[Da] # No unit

    # ---------------- LTPwin_tab ------------------------------- #
    cdef gsl_vector * LTPwin_tab = gsl_vector_alloc(int(n_x_LTPwin))
    cdef int i
    cdef int ind_param_LTPwin
    cdef double x0, LTDw, w, ka, LTDwin, omega

    # --------------- ECb smooth ----------------------------------- #
    if on_ECb_smooth:
        x0 = 0.5*(LTDstart +LTDstop)
        LTDw = x0 -LTDstart
        w = kw_ECb_smooth*LTDw
        ka = K_ECb_smooth + kadd_ECb_smooth *(1 -fabs(ctrl1 -x0)/LTDw)**kn_ECb_smooth *rect_func(ctrl1, x0-LTDw, x0+LTDw)
        LTDwin = (hill_func(ctrl1-x0, ka, n_ECb_smooth)-1)*rect_func(ctrl1,x0-w,x0+w)

        # ---- Set LTPwin_tab ---- #
        for i from 0 <= i < int(n_x_LTPwin):
            ind_param_LTPwin = int(i+1 +ind_n_x_NMDA)
            gsl_vector_set(LTPwin_tab, i, (<double *> params)[ind_param_LTPwin]) # Fill the vector

        LTPwin = lin_interpTable_brds(LTPwin_tab, int(n_x_LTPwin), (ctrl1 -xst_LTPwin)/xstep_LTPwin)
        omega = 1.0 + LTDMax*LTDwin +LTPMax*LTPwin

    else:
        LTDMax_temp = 0.
        LTPMax_temp = 0.

        if ctrl1 -LTDstart > 0.:
            if ctrl1 <= LTDstop:
                LTDMax_temp = LTDMax

        if ctrl1 >= LTPstart:
            LTPMax_temp = LTPMax

        omega = 1 -LTDMax_temp +LTPMax_temp

    # ------------------ caL13 ----------------------------------- #
    cdef double I_caL13 = 0.
    cdef double x, eco, eci, ghk
    cdef gsl_vector * minf_caL13_tab = gsl_vector_alloc(int(n_x_caL13))
    cdef gsl_vector * hinf_caL13_tab = gsl_vector_alloc(int(n_x_caL13))
    cdef gsl_vector * mtau_caL13_tab = gsl_vector_alloc(int(n_x_caL13))
    cdef int ind_param_minf, ind_param_hinf, ind_param_mtau

    if on_caL13: # VSCC current Ca_v1.3

        x = zS*F*y[V]*1e-3/RT # 1e-3 No unit
        eco = Ca_out*efun(x) # microM*1e-3
        eci = y[Ca_cyt]*efun(-x) # microM*1e-3
        ghk = zS*F*(eci -eco) # kCoul.microM*1e-3.mol-1
        I_caL13 = ghk *pcaLbar *y[m_caL13] *y[m_caL13] *y[h_caL13]

        # ---- Set minf_caL13_tab, hinf_caL13_tab and mtau_caL13_tab ---- #
        for i from 0 <= i < int(n_x_caL13):
            ind_param_minf = int(i+1 +ind_n_x_NMDA +int_n_x_LTPwin)
            ind_param_hinf = int(ind_param_minf +int_n_x_caL13)
            ind_param_mtau = int(ind_param_hinf +int_n_x_caL13)

            #            printf('ind_param_minf %d\n', ind_param_minf)
            #            printf('ind_param_hinf %d\n', ind_param_hinf)
            #            printf('ind_param_mtau %d\n', ind_param_mtau)

            #            printf('ind_param_minf %d\n', ind_param_mtau)
            #            printf('minf%f\n', (<double *> params)[ind_param_mtau])

            gsl_vector_set(minf_caL13_tab, i, (<double *> params)[ind_param_minf]) # Fill the vector
            gsl_vector_set(hinf_caL13_tab, i, (<double *> params)[ind_param_hinf]) # Fill the vector
            gsl_vector_set(mtau_caL13_tab, i, (<double *> params)[ind_param_mtau]) # Fill the vector

            #        for i from 0 <= i < int(n_x_caL13):

            #            printf('minf%f\n',gsl_vector_get(minf_caL13_tab,i))
            #            printf('hinf%f\n',gsl_vector_get(hinf_caL13_tab,i))
            #            printf('mtau%f\n\n',gsl_vector_get(mtau_caL13_tab,i))


    cdef double taufpre = P1_ECb / ( P2_ECb**P3_ECb + ctrl2**P3_ECb ) +P4_ECb

    # --------- CB1R ----------- #
    cdef double c_CB1R = 1 -y[o_CB1R] -y[d_CB1R]
    # ----------------------

    # CaM and CaMKII plasticity
    cdef double occ_r_CaMKII = y[B1] + y[B2] + y[B3] + y[B4] + y[B5] + y[B6] + y[B7] + y[B8] + y[B9] + y[B10] + y[B11] + y[B12] + y[B13]

    # B0 is what's left from total #
    cdef double B0 = 2 *CaMKT -occ_r_CaMKII

    # Kinetic equations #
    cdef double phos_sum = y[B1] + 2*(y[B2] + y[B3] + y[B4]) + 3*(y[B5] + y[B6] + y[B7] + y[B8]) + 4*(y[B9] + y[B10] + y[B11]) + 5*y[B12] + 6*y[B13]

    cdef double k10 = k12*y[PP1]/(KM + phos_sum)

    # Calmodulin #
    #cdef double CaM =  CaMT/(1 + Ka4/y[Ca_cyt] + Ka3*Ka4/(y[Ca_cyt]**2) + Ka2*Ka3*Ka4/(y[Ca_cyt]**3) +Ka1*Ka2*Ka3*Ka4/(y[Ca_cyt]**4))

    cdef double gamma = y[CamCa4]/(K5+y[CamCa4])
    cdef double vPKA = kpka0I1 + kpkaI1/(1 + (KdpkaI1/y[CamCa4])**npkaI1)
    cdef double vCaN = kcan0I1 + kcanI1/(1 + (KdcanI1/y[CamCa4])**ncanI1)


    cdef double gamma2 = gamma*gamma
    cdef double k6gamma2 = k6*gamma2
    cdef double k7gamma = k7*gamma

    # ------------- I AMPA ------------------- #
    cdef double I_AMPA = 0.
    if on_AMPA:
        I_AMPA = gAMPA*y[o_AMPA]*y[V]

    cdef double c_AMPA = 1 -y[o_AMPA] -y[d_AMPA]

    # ------------- NMDA ------------------- #
    cdef double G_NMDA = 0
    cdef double I_NMDA = 0
    cdef gsl_vector * B_NMDA_tab = gsl_vector_alloc(int(n_x_NMDA))
    cdef int ind_param_B

    if on_NMDA:

        for i from 0 <= i < int(n_x_NMDA):
            ind_param_B = int(i+1 +ind_n_x_NMDA +int_n_x_LTPwin +3*int_n_x_caL13)

            gsl_vector_set(B_NMDA_tab, i, (<double *> params)[ind_param_B]) # Fill the vector

    G_NMDA = y[o_NMDA] *lin_interpTable_brds(B_NMDA_tab, int(n_x_NMDA), (y[V] -xst_NMDA)/xstep_NMDA) # No unit
    I_NMDA = gNMDA *y[V] *G_NMDA

    # ------------- TRPV1 ------------------- #
    cdef double Q
    cdef double G_TRPV1 = 0
    cdef double I_TRPV1 = 0

    if on_TRPV1:

        Q=y[AEA]/KD_TRPV1 # No unit
        x=z_TRPV1*F*y[V]/RT

        if x <= 85.0:
            J = J0_TRPV1*exp(x)
            G_TRPV1 = 1/(1+(1+J +K_TRPV1 +Q +J*K_TRPV1+J*Q +K_TRPV1*Q+J*K_TRPV1*Q)/(L_TRPV1*(1+J*D_TRPV1 +K_TRPV1*C_TRPV1+Q*P_TRPV1+J*K_TRPV1*C_TRPV1*D_TRPV1 +J*Q*D_TRPV1*P_TRPV1+K_TRPV1*Q*C_TRPV1*P_TRPV1 +J*K_TRPV1*Q*D_TRPV1*C_TRPV1*P_TRPV1)))

        else:
            G_TRPV1 = 1/(1+(1+K_TRPV1+Q+K_TRPV1*Q)/(L_TRPV1*(D_TRPV1+K_TRPV1*C_TRPV1*D_TRPV1+Q*D_TRPV1*P_TRPV1+K_TRPV1*Q*D_TRPV1*C_TRPV1*P_TRPV1)))

        I_TRPV1 = gTRPV1 *y[V] *G_TRPV1 # pA

    # subcellular calcium, IP3, DAG and 2-AG

    cdef double minf = y[IP3]/(y[IP3] +d1_CICR)
    cdef double ninf = y[Ca_cyt]/(y[Ca_cyt] +d5_CICR)
    cdef double J_IP3R = rc_CICR * (minf*ninf*y[h_CICR])**3 *(y[Ca_ER] -y[Ca_cyt])

    cdef double J_serca = ver_CICR * hill_func(y[Ca_cyt],ker_CICR,2)
    cdef double J_leak = rl_CICR * (y[Ca_ER] -y[Ca_cyt])
    cdef double J_CaER = J_IP3R -J_serca +J_leak # Ca from ER

    cdef double J_Ca_ch = -VDCC *I_caL13 -NMDA*I_NMDA -TRPV1*I_TRPV1 # Concentration.time-1

    cdef double J_tot = J_CaER + J_Ca_ch # Concentration.time-1

    # IP3, DAG, ECb
    cdef double Hill1=y[Ca_cyt]/(y[Ca_cyt] +kpi_IP3)
    cdef double vglu = vbeta_IP3 * Glu / (Glu+(kr_IP3 * (1+kp_IP3/kr_IP3*Hill1) ))
    cdef double vplcg = vdelta_IP3/(1 +y[IP3]/kappad_IP3)*hill_func(y[Ca_cyt],kdelta_IP3,2)
    cdef double vip3prod = vglu + vplcg

    cdef double v3k = v3k_IP3*phos_sum * hill_func(y[IP3],k3_IP3,n3_IP3)


    # Membrane potential
    cdef double Ileak = gL * (y[V] - EL)
    cdef double Itotal = -Ileak -I_caL13 -I_TRPV1 -I_AMPA -I_NMDA -I_action


    # CaMKII sub-units concentration #
    #    cdef double CK_sub_unit = conc_max_sub_unit -phos_sum
    #    cdef double CKp_sub_unit = phos_sum
    #    cdef double CK_CamCa4_sub_unit = CK *y[CamCa4]/K5
    #    cdef double CKp_CamCa4_sub_unit = CKp *y[CamCa4]/K9

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # ----------- ODE SOLVING ----------- #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # subcellular calcium, IP3, DAG and 2-AG
    if on_caL13:
        f[h_caL13] = (lin_interpTable_brds(hinf_caL13_tab, int(n_x_caL13), (y[V] -xst_caL13)/xstep_caL13) - y[h_caL13]) / (htau_caL13/hqfact_caL13) # time-1
        f[m_caL13] = (lin_interpTable_brds(minf_caL13_tab, int(n_x_caL13), (y[V] -xst_caL13)/xstep_caL13) -y[m_caL13]) / (lin_interpTable_brds(mtau_caL13_tab, int(n_x_caL13), (y[V] -xst_caL13)/xstep_caL13)/qfact_caL13) # time-1
    else:
        f[h_caL13] = 0.
        f[m_caL13] = 0.

    # ------------- I AMPA ------------------- #
    if on_AMPA:

        f[o_AMPA] = Alpha_AMPA*Glu*c_AMPA -(Beta_AMPA +Gamma_AMPA)*y[o_AMPA]
        f[d_AMPA] = -Epsilon_AMPA*y[d_AMPA] +Gamma_AMPA*y[o_AMPA]
    else:
        f[o_AMPA] = 0.
        f[d_AMPA] = 0.

    f[o_NMDA] = Alpha_NMDA*Glu*(1 -y[o_NMDA]) -Beta_NMDA*y[o_NMDA]

    f[h_CICR] = (a2_CICR *d2_CICR *(y[IP3] +d1_CICR)/(y[IP3] +d3_CICR))*(1 -y[h_CICR]) - a2_CICR *y[Ca_cyt]*y[h_CICR]
    f[Ca_cyt] = (J_tot -(y[Ca_cyt] -Cab)/tauCab
                -forwardRate_CamC_bind*y[Cam]*y[Ca_cyt]**2 +reverseRate_CamC_bind*y[CamCa2C] -forwardRate_CamCa2C_bind*y[CamCa2C]*y[Ca_cyt]**2 +reverseRate_CamCa2C_bind*y[CamCa4] -forwardRate_CamN_bind*y[Cam]*y[Ca_cyt]**2 +reverseRate_CamN_bind*y[CamCa2N] -forwardRate_CamCa2N_bind*y[CamCa2N]*y[Ca_cyt]**2 +reverseRate_CamCa2N_bind*y[CamCa4] -forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac*y[PP2BCam]*y[Ca_cyt]**2 +reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac*y[PP2BCamCa2C] -forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac*y[PP2BCam]*y[Ca_cyt]**2 +reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac*y[PP2BCamCa2N] -forwardRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac*y[PP2BCamCa2C]*y[Ca_cyt]**2 +reverseRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac*y[PP2BCamCa4] -forwardRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac*y[PP2BCamCa2N]*y[Ca_cyt]**2 +reverseRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac*y[PP2BCamCa4]
                -forwardRate_Ca_plus_Dlg__CaDgl*y[Ca_cyt]*y[Dgl] +reverseRate_Ca_plus_Dlg__CaDgl*y[CaDgl] -forwardRate_Ca_plus_pDlg__CapDgl*y[Ca_cyt]*y[pDgl] +reverseRate_Ca_plus_pDlg__CapDgl*y[CapDgl]
                )/tauCa(y[Ca_cyt], BT_CaBuff, KdB_CaBuff)

    f[Ca_ER] = -J_CaER*rhoER_CICR/tauCa(y[Ca_ER], BT_CaBuff, KdB_CaBuff)

    # IP3, DAG, ECb
    f[IP3] = vip3prod -v3k -r5p_IP3 *y[IP3]
    f[DAG] = 0.#vip3prod -rDGL *y[DAGLP]*y[DAG]/(y[DAG]+KDGL) -kDAGK*y[DAG]

    f[DAGLP] = 0.#rK_DAGLP*(alpha_DAGLP_Ca_cyt *y[Ca_cyt])**nK_DAGLP*(1 -y[DAGLP]) -rP_DAGLP*y[DAGLP]

    if on_ECb:

        f[twoAG] = 0.#rDGL * y[DAGLP] * y[DAG] / (y[DAG] + KDGL) - kMAGL * y[twoAG]
        f[AEA] = vATAEA * y[Ca_cyt] - vFAAH * y[AEA] / (KFAAH + y[AEA])
        f[fpre] = (omega-y[fpre]) /taufpre
    else:

        f[twoAG] = 0.
        f[AEA] = 0.
        f[fpre] = 0.


    # dPP1/dt and dI1P/dt #
    f[I1P] = 0.#-k11*y[I1P]*y[PP1] + km11*(PP10 - y[PP1]) + vPKA*I10 - vCaN*y[I1P]
    f[PP1] = -forwardRate_PDE10_d_phosp1*y[pPDE10]*y[PP1] +reverseRate_PDE10_d_phosp1*y[pPDE10PP1] +forwardRate_PDE10_d_phosp2*y[pPDE10PP1] -forwardRate_D32p34_plus_PP1__D32p34pp1_reac*y[D32p34]*y[PP1] +reverseRate_D32p34_plus_PP1__D32p34pp1_reac*y[D32p34PP1] +forwardRate_D32p34PP1PP2BCamCa4__PP1_plus_PP2BCamCa4_plus_D32_reac*y[D32p34PP1PP2BCamCa4] +forwardRate_D32p34PP1PP2ABPR72__PP1_plus_PP2ABPR72_plus_D32_reac*y[D32p34PP1PP2ABPR72] +forwardRate_D32p34PP1PP2AB56d__PP1_plus_PP2AB56d_plus_D32_reac*y[D32p34PP1PP2AB56d]

    # membrane potential
    f[V] = Itotal/Cm

    if on_CB1R:
        f[o_CB1R] = Alpha_CB1R *(y[twoAG] +alphaAEACB1*y[AEA]) *c_CB1R - (Beta_CB1R +Gamma_CB1R)*y[o_CB1R]
        f[d_CB1R] = -Epsilon_CB1R*y[d_CB1R] + Gamma_CB1R*y[o_CB1R]
    else:
        f[o_CB1R] = 0.
        f[d_CB1R] = 0.

    # dBi/dt #
    f[B1] = 6*k6gamma2*B0 - 4*k6gamma2*y[B1] - k7gamma*y[B1] - k10*y[B1] + 2*k10*(y[B2] + y[B3] + y[B4])
    f[B2] = k7gamma*y[B1] + k6gamma2*y[B1] - 3*k6gamma2*y[B2] - k7gamma*y[B2] - 2*k10*y[B2] + k10*(2*y[B5] + y[B6] + y[B7])
    f[B3] = 2*k6gamma2*y[B1] - 2*k7gamma*y[B3] - 2*k6gamma2*y[B3] - 2*k10*y[B3] + k10*(y[B5] + y[B6] + y[B7] + 3*y[B8])
    f[B4] = k6gamma2*y[B1] - 2*k7gamma*y[B4] - 2*k6gamma2*y[B4] - 2*k10*y[B4] + k10*(y[B6] + y[B7])
    f[B5] = k7gamma*y[B2] + k7gamma*y[B3] + k6gamma2*y[B2] - k7gamma*y[B5] - 2*k6gamma2*y[B5] - 3*k10*y[B5] + k10*(2*y[B9] + y[B10])
    f[B6] = k6gamma2*y[B2] + k6gamma2*y[B3]  + 2*k7gamma*y[B4] - k6gamma2*y[B6] - 2*k7gamma*y[B6] - 3*k10*y[B6] + k10*(y[B9] + y[B10] + 2*y[B11])
    f[B7] = k6gamma2*y[B2] + k7gamma*y[B3] + 2*k6gamma2*y[B4] - k6gamma2*y[B7] - 2*k7gamma*y[B7] - 3*k10*y[B7] + k10*(y[B9] + y[B10] + 2*y[B11])
    f[B8] = k6gamma2*y[B3] - 3*k7gamma*y[B8] - 3*k10*y[B8] + k10*y[B10]
    f[B9] = k7gamma*y[B5] + k6gamma2*y[B5] + k7gamma*y[B6] + k7gamma*y[B7] - k6gamma2*y[B9] - k7gamma*y[B9] - 4*k10*y[B9] + 2*k10*y[B12]
    f[B10] = k6gamma2*y[B5] + k6gamma2*y[B6] + k7gamma*y[B7] + 3*k7gamma*y[B8] - 2*k7gamma*y[B10] - 4*k10*y[B10] + 2*k10*y[B12]
    f[B11] = k7gamma*y[B6] +  k6gamma2*y[B7] - 2*k7gamma*y[B11] - 4*k10*y[B11] + k10*y[B12]
    f[B12] = k7gamma*y[B9] + k6gamma2*y[B9] + 2*k7gamma*y[B10] + 2*k7gamma*y[B11] - k7gamma*y[B12] - 5*k10*y[B12] + 6*k10*y[B13]
    f[B13] = k7gamma*y[B12] - 6*k10*y[B13]

    # ---- Blackwell variables ---- #

    f[Ca] = 0.
    f[pmca] = 0.
    f[pmcaCa] = 0.
    f[CaOut] = 0.
    f[ncx] = 0.
    f[ncxCa] = 0.
    f[Leak] = 0.
    f[CaOutLeak] = 0.
    f[Calbin] = 0.
    f[CalbinC] = 0.
    f[CamCa2N] = forwardRate_CamN_bind*y[Cam]*y[Ca_cyt]**2 -reverseRate_CamN_bind*y[CamCa2N] -forwardRate_CamCa2N_bind*y[CamCa2N]*y[Ca_cyt]**2 +reverseRate_CamCa2N_bind*y[CamCa4]

    f[DaDbuf] = -forwardRate_DaDbuf__Da_plus_Dbuf*y[DaDbuf]
    f[Dbuf] = forwardRate_DaDbuf__Da_plus_Dbuf*y[DaDbuf] -forwardRate_DaOut_plus_Dbuf__NULL*y[DaOut]*y[Dbuf]
    f[DaOut] = forwardRate_Da__DaOut*y[Da] -reverseRate_Da__DaOut*y[DaOut] -forwardRate_DaOut_plus_Dbuf__NULL*y[DaOut]*y[Dbuf]
    f[Da] = forwardRate_DaDbuf__Da_plus_Dbuf*y[DaDbuf] -forwardRate_Da__DaOut*y[Da] +reverseRate_Da__DaOut*y[DaOut] -forwardRate_Da_plus_D1R__DaD1R*y[Da]*y[D1R] +reverseRate_Da_plus_D1R__DaD1R*y[DaD1R] -forwardRate_GsD1R_plus_Da__DaD1RGs*y[GsD1R]*y[Da] +reverseRate_GsD1R_plus_Da__DaD1RGs*y[DaD1RGs]
    f[DaD1R] = forwardRate_Da_plus_D1R__DaD1R*y[Da]*y[D1R] -reverseRate_Da_plus_D1R__DaD1R*y[DaD1R] -forwardRate_DaD1R_plus_Gs__DaD1RGs*y[DaD1R]*y[Gsabg] +reverseRate_DaD1R_plus_Gs__DaD1RGs*y[DaD1RGs] +forwardRate_DaD1RGs_DaD1R_plus_GsaGTP_plus_Gbg_reac*y[DaD1RGs]
    f[Gsabg] = -forwardRate_DaD1R_plus_Gs__DaD1RGs*y[DaD1R]*y[Gsabg] +reverseRate_DaD1R_plus_Gs__DaD1RGs*y[DaD1RGs] -forwardRate_Gsabg_plus_D1R__GsD1R*y[Gsabg]*y[D1R] +reverseRate_Gsabg_plus_D1R__GsD1R*y[GsD1R] +forwardRate_GsaGDP_plus_Gbg__Gs_reac*y[GsaGDP]*y[Gbg]
    f[DaD1RGs] = forwardRate_DaD1R_plus_Gs__DaD1RGs*y[DaD1R]*y[Gsabg] -reverseRate_DaD1R_plus_Gs__DaD1RGs*y[DaD1RGs] +forwardRate_GsD1R_plus_Da__DaD1RGs*y[GsD1R]*y[Da] -reverseRate_GsD1R_plus_Da__DaD1RGs*y[DaD1RGs] -forwardRate_DaD1RGs_DaD1R_plus_GsaGTP_plus_Gbg_reac*y[DaD1RGs] -forwardRate_DaD1RGs_phosp1*y[PKAc]*y[DaD1RGs] +reverseRate_DaD1RGs_phosp1*y[PKAcDaD1RGs] +forwardRate_dephosD1R*y[pDaD1RGs]
    f[D1R] = -forwardRate_Da_plus_D1R__DaD1R*y[Da]*y[D1R] +reverseRate_Da_plus_D1R__DaD1R*y[DaD1R] -forwardRate_Gsabg_plus_D1R__GsD1R*y[Gsabg]*y[D1R] +reverseRate_Gsabg_plus_D1R__GsD1R*y[GsD1R]
    f[GsD1R] = forwardRate_Gsabg_plus_D1R__GsD1R*y[Gsabg]*y[D1R] -reverseRate_Gsabg_plus_D1R__GsD1R*y[GsD1R] -forwardRate_GsD1R_plus_Da__DaD1RGs*y[GsD1R]*y[Da] +reverseRate_GsD1R_plus_Da__DaD1RGs*y[DaD1RGs]

    f[GsaGTP] = forwardRate_DaD1RGs_DaD1R_plus_GsaGTP_plus_Gbg_reac*y[DaD1RGs] -forwardRate_GsaGTP__GsaGDP_reac*y[GsaGTP] -forwardRate_GsaGTP_plus_AC5__AC5Gsa_reac*y[GsaGTP]*y[AC5] +reverseRate_GsaGTP_plus_AC5__AC5Gsa_reac*y[AC5Gsa] -forwardRate_GsaGTP_AC1_reac*y[GsaGTP]*y[AC1] +reverseRate_GsaGTP_AC1_reac*y[AC1Gsa]

    f[PKAc] = -forwardRate_DaD1RGs_phosp1*y[PKAc]*y[DaD1RGs] +reverseRate_DaD1RGs_phosp1*y[PKAcDaD1RGs] +forwardRate_DaD1RGs_phosp2*y[PKAcDaD1RGs] -forwardRate_PDE10_phosp1*y[PKAc]*y[PDE10] +reverseRate_PDE10_phosp1*y[PKAcPDE10] +forwardRate_PDE10_phosp2*y[PKAcPDE10] -forwardRate_PDE10cAMP_phosp1*y[PKAc]*y[PDE10cAMP] +reverseRate_PDE10cAMP_phosp1*y[PKAcPDE10cAMP] +forwardRate_PDE10cAMP_phosp2*y[PKAcPDE10cAMP] +forwardRate_PKAcAMP4_diss*y[PKAcAMP4] -reverseRate_PKAcAMP4_diss*y[PKAr]*y[PKAc]**2 -forwardRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac*y[PP2AB56d]*y[PKAc] +reverseRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac*y[PKAcPP2AB56d] +forwardRate_PKAcPP2AB56d__PKAc_plus_pPP2A_reac*y[PKAcPP2AB56d] -forwardRate_D32_plus_PKAc__D32PKAc_reac*y[D32]*y[PKAc] +reverseRate_D32_plus_PKAc__D32PKAc_reac*y[D32PKAc] +forwardRate_D32PKAc__PKAc_plus_D32p34_reac*y[D32PKAc] -forwardRate_D32p75_plus_PKAc__D32p75PKAc_reac*y[D32p75]*y[PKAc] +reverseRate_D32p75_plus_PKAc__D32p75PKAc_reac*y[D32p75PKAc]
    f[PKAcDaD1RGs] = forwardRate_DaD1RGs_phosp1*y[PKAc]*y[DaD1RGs] -reverseRate_DaD1RGs_phosp1*y[PKAcDaD1RGs] -forwardRate_DaD1RGs_phosp2*y[PKAcDaD1RGs]
    f[pDaD1RGs] = forwardRate_DaD1RGs_phosp2*y[PKAcDaD1RGs] -forwardRate_dephosD1R*y[pDaD1RGs]



    f[AC5] = -forwardRate_GsaGTP_plus_AC5__AC5Gsa_reac*y[GsaGTP]*y[AC5] +reverseRate_GsaGTP_plus_AC5__AC5Gsa_reac*y[AC5Gsa] -forwardRate_GiaGTP_AC5_reac*y[GiaGTP]*y[AC5] +reverseRate_GiaGTP_AC5_reac*y[AC5Gia]
    f[AC5Gsa] = forwardRate_GsaGTP_plus_AC5__AC5Gsa_reac*y[GsaGTP]*y[AC5] -reverseRate_GsaGTP_plus_AC5__AC5Gsa_reac*y[AC5Gsa] -forwardRate_AC5Gsa_plus_ATP__AC5GsaATP_reac*y[AC5Gsa]*y[ATP] +reverseRate_AC5Gsa_plus_ATP__AC5GsaATP_reac*y[AC5GsaATP] +forwardRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac*y[AC5GsaATP] -reverseRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac*y[AC5Gsa]*y[cAMP] -forwardRate_GiaGTP_AC5GsaGTP_reac*y[GiaGTP]*y[AC5Gsa] +reverseRate_GiaGTP_AC5GsaGTP_reac*y[AC5GsaGia]
    f[ATP] = -forwardRate_AC5Gsa_plus_ATP__AC5GsaATP_reac*y[AC5Gsa]*y[ATP] +reverseRate_AC5Gsa_plus_ATP__AC5GsaATP_reac*y[AC5GsaATP] -forwardRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP*y[AC5GsaGia]*y[ATP] +reverseRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP*y[AC5GsaGiaATP] +forwardRate_AMP__ATP*y[AMP] -forwardRate_AC1GsaGTPCamCa4_ATP_Reac*y[AC1GsaCamCa4]*y[ATP] +reverseRate_AC1GsaGTPCamCa4_ATP_Reac*y[AC1GsaCamCa4ATP] -forwardRate_AC1CamCa4ATP_bind*y[AC1CamCa4]*y[ATP] +reverseRate_AC1CamCa4ATP_bind*y[AC1CamCa4ATP]
    f[AC5GsaATP] = forwardRate_AC5Gsa_plus_ATP__AC5GsaATP_reac*y[AC5Gsa]*y[ATP] -reverseRate_AC5Gsa_plus_ATP__AC5GsaATP_reac*y[AC5GsaATP] -forwardRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac*y[AC5GsaATP] +reverseRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac*y[AC5Gsa]*y[cAMP]

    f[cAMP] = forwardRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac*y[AC5GsaATP] -reverseRate_AC5GsaATP__AC5Gsa_plus_cAMP_reac*y[AC5Gsa]*y[cAMP] +forwardRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP*y[AC5GsaGiaATP] -reverseRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP*y[AC5GsaGia]*y[cAMP] +forwardRate_AC5GsaGiaATP__AC5Gia_plus_Gs_plus_cAMP*y[AC5GsaGiaATP] -forwardRate_PDE10_plus_cAMP__PDE10cAMP*y[PDE10]*y[cAMP] +reverseRate_PDE10_plus_cAMP__PDE10cAMP*y[PDE10cAMP] -forwardRate_pPDE10_plus_cAMP__pPDE10cAMP*y[pPDE10]*y[cAMP] +reverseRate_pPDE10_plus_cAMP__pPDE10cAMP*y[pPDE10cAMP] -forwardRate_PKA_bind*y[PKA]*y[cAMP]**2 +reverseRate_PKA_bind*y[PKAcAMP2] -forwardRate_PKAcAMP2_bind*y[PKAcAMP2]*y[cAMP]**2 +reverseRate_PKAcAMP2_bind*y[PKAcAMP4]

    f[AC5Gia] = forwardRate_GiaGTP_AC5_reac*y[GiaGTP]*y[AC5] -reverseRate_GiaGTP_AC5_reac*y[AC5Gia] +forwardRate_AC5GsaGiaATP__AC5Gia_plus_Gs_plus_cAMP*y[AC5GsaGiaATP]
    f[AC5GsaGia] = forwardRate_GiaGTP_AC5GsaGTP_reac*y[GiaGTP]*y[AC5Gsa] -reverseRate_GiaGTP_AC5GsaGTP_reac*y[AC5GsaGia] -forwardRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP*y[AC5GsaGia]*y[ATP] +reverseRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP*y[AC5GsaGiaATP] +forwardRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP*y[AC5GsaGiaATP] -reverseRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP*y[AC5GsaGia]*y[cAMP]
    f[AC5GsaGiaATP] = forwardRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP*y[AC5GsaGia]*y[ATP] -reverseRate_AC5GsaGia_plus_ATP__AC5GsaGiaATP*y[AC5GsaGiaATP] -forwardRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP*y[AC5GsaGiaATP] +reverseRate_AC5GsaGiaATP__AC5GsaGia_plus_cAMP*y[AC5GsaGia]*y[cAMP] -forwardRate_AC5GsaGiaATP__AC5Gia_plus_Gs_plus_cAMP*y[AC5GsaGiaATP]

    f[PDE1] = 0.
    f[PDE1CamCa4] = 0.
    f[PDE1CamCa4cAMP] = 0.
    f[AMP] = forwardRate_PDE10cAMP__PDE10_plus_AMP*y[PDE10cAMP] +forwardRate_pPDE10cAMP__pPDE10_plus_AMP*y[pPDE10cAMP] -forwardRate_AMP__ATP*y[AMP]


    f[PDE10] = -forwardRate_PDE10_plus_cAMP__PDE10cAMP*y[PDE10]*y[cAMP] +reverseRate_PDE10_plus_cAMP__PDE10cAMP*y[PDE10cAMP] +forwardRate_PDE10cAMP__PDE10_plus_AMP*y[PDE10cAMP] -forwardRate_PDE10_phosp1*y[PKAc]*y[PDE10] +reverseRate_PDE10_phosp1*y[PKAcPDE10] +forwardRate_PDE10_d_phosp2*y[pPDE10PP1]
    f[PDE10cAMP] = forwardRate_PDE10_plus_cAMP__PDE10cAMP*y[PDE10]*y[cAMP] -reverseRate_PDE10_plus_cAMP__PDE10cAMP*y[PDE10cAMP] -forwardRate_PDE10cAMP__PDE10_plus_AMP*y[PDE10cAMP] -forwardRate_PDE10cAMP_phosp1*y[PKAc]*y[PDE10cAMP] +reverseRate_PDE10cAMP_phosp1*y[PKAcPDE10cAMP]
    f[pPDE10] = -forwardRate_pPDE10_plus_cAMP__pPDE10cAMP*y[pPDE10]*y[cAMP] +reverseRate_pPDE10_plus_cAMP__pPDE10cAMP*y[pPDE10cAMP] +forwardRate_pPDE10cAMP__pPDE10_plus_AMP*y[pPDE10cAMP] +forwardRate_PDE10_phosp2*y[PKAcPDE10] -forwardRate_PDE10_d_phosp1*y[pPDE10]*y[PP1] +reverseRate_PDE10_d_phosp1*y[pPDE10PP1]
    f[pPDE10cAMP] = forwardRate_pPDE10_plus_cAMP__pPDE10cAMP*y[pPDE10]*y[cAMP] -reverseRate_pPDE10_plus_cAMP__pPDE10cAMP*y[pPDE10cAMP] -forwardRate_pPDE10cAMP__pPDE10_plus_AMP*y[pPDE10cAMP] +forwardRate_PDE10cAMP_phosp2*y[PKAcPDE10cAMP]
    f[PKAcPDE10] = forwardRate_PDE10_phosp1*y[PKAc]*y[PDE10] -reverseRate_PDE10_phosp1*y[PKAcPDE10] -forwardRate_PDE10_phosp2*y[PKAcPDE10]
    f[PKAcPDE10cAMP] = forwardRate_PDE10cAMP_phosp1*y[PKAc]*y[PDE10cAMP] -reverseRate_PDE10cAMP_phosp1*y[PKAcPDE10cAMP] -forwardRate_PDE10cAMP_phosp2*y[PKAcPDE10cAMP]
    f[PP1_Blackwell] = 0.
    f[pPDE10PP1] = forwardRate_PDE10_d_phosp1*y[pPDE10]*y[PP1] -reverseRate_PDE10_d_phosp1*y[pPDE10PP1] -forwardRate_PDE10_d_phosp2*y[pPDE10PP1]

    f[PDE2] = 0.
    f[PDE2cAMP] = 0.
    f[PDE2cAMP2] = 0.
    f[PDE4] = 0.
    f[PDE4cAMP] = 0.
    f[pPDE4] = 0.
    f[pPDE4cAMP] = 0.
    f[PKAcPDE4] = 0.
    f[PKAcPDE4cAMP] = 0.
    f[pPDE4PP1] = 0.

    f[Ip3] = 0.
    f[PlcCaGqaPip2] = 0.
    f[Dgl] = -forwardRate_Ca_plus_Dlg__CaDgl*y[Ca_cyt]*y[Dgl] +reverseRate_Ca_plus_Dlg__CaDgl*y[CaDgl] +forwardRate_pDgl_dephos*y[pDgl]
    f[CaDgl] = 0.
    f[DagCaDgl] = forwardRate_Dag_plus_CaDlg__DagCaDgl*y[Dag]*y[CaDgl] -reverseRate_Dag_plus_CaDlg__DagCaDgl*y[DagCaDgl] -forwardRate_DagCaDgl__CaDgl_plus_2ag*y[DagCaDgl]
    f[two_ag] = forwardRate_DagCaDgl__CaDgl_plus_2ag*y[DagCaDgl] -forwardRate_2ag__2agDegrad*y[two_ag] +forwardRate_DagCapDgl__CapDgl_plus_2ag*y[DagCapDgl]
    f[PIkinase] = 0.
    f[Ip3degPIk] = 0.
    f[two_agDegrad] = forwardRate_2ag__2agDegrad*y[two_ag]
    f[Ip3degrad] = 0.
    f[DagK] = 0.
    f[DagKdag] = 0.
    f[PA] = 0.
    f[Pkc] = 0.
    f[PkcCa] = 0.
    f[PkcCaDag] = 0.

    f[PKA] = -forwardRate_PKA_bind*y[PKA]*y[cAMP]**2 +reverseRate_PKA_bind*y[PKAcAMP2]
    f[PKAcAMP2] = forwardRate_PKA_bind*y[PKA]*y[cAMP]**2 -reverseRate_PKA_bind*y[PKAcAMP2] -forwardRate_PKAcAMP2_bind*y[PKAcAMP2]*y[cAMP]**2 +reverseRate_PKAcAMP2_bind*y[PKAcAMP4]
    f[PKAcAMP4] = forwardRate_PKAcAMP2_bind*y[PKAcAMP2]*y[cAMP]**2 -reverseRate_PKAcAMP2_bind*y[PKAcAMP4] -forwardRate_PKAcAMP4_diss*y[PKAcAMP4] +reverseRate_PKAcAMP4_diss*y[PKAr]*y[PKAc]**2
    f[PKAr] = forwardRate_PKAcAMP4_diss*y[PKAcAMP4] -reverseRate_PKAcAMP4_diss*y[PKAr]*y[PKAc]**2

    f[Cam] = -forwardRate_CamC_bind*y[Cam]*y[Ca_cyt]**2 +reverseRate_CamC_bind*y[CamCa2C] -forwardRate_CamN_bind*y[Cam]*y[Ca_cyt]**2 +reverseRate_CamN_bind*y[CamCa2N] -forwardRate_Cam_plus_PP2B__PP2BCam_reac*y[Cam]*y[PP2B] +reverseRate_Cam_plus_PP2B__PP2BCam_reac*y[PP2BCam]
    f[PP2B] = -forwardRate_CamCa4_plus_PP2B__PP2BCamCa4_reac*y[CamCa4]*y[PP2B] +reverseRate_CamCa4_plus_PP2B__PP2BCamCa4_reac*y[PP2BCamCa4]
    f[PP2BCam] = forwardRate_Cam_plus_PP2B__PP2BCam_reac*y[Cam]*y[PP2B] -reverseRate_Cam_plus_PP2B__PP2BCam_reac*y[PP2BCam] -forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac*y[PP2BCam]*y[Ca_cyt]**2 +reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac*y[PP2BCamCa2C] -forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac*y[PP2BCam]*y[Ca_cyt]**2 +reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac*y[PP2BCamCa2N]
    f[CamCa2C] = forwardRate_CamC_bind*y[Cam]*y[Ca_cyt]**2 -reverseRate_CamC_bind*y[CamCa2C] -forwardRate_CamCa2C_bind*y[CamCa2C]*y[Ca_cyt]**2 +reverseRate_CamCa2C_bind*y[CamCa4] -forwardRate_CamCa2C_plus_PP2B__PP2BCamCa2C_reac*y[CamCa2C]*y[PP2B] +reverseRate_CamCa2C_plus_PP2B__PP2BCamCa2C_reac*y[PP2BCamCa2C]

    f[PP2BCamCa2C] = forwardRate_CamCa2C_plus_PP2B__PP2BCamCa2C_reac*y[CamCa2C]*y[PP2B] -reverseRate_CamCa2C_plus_PP2B__PP2BCamCa2C_reac*y[PP2BCamCa2C] +forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac*y[PP2BCam]*y[Ca_cyt]**2 -reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2C_reac*y[PP2BCamCa2C] -forwardRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac*y[PP2BCamCa2C]*y[Ca_cyt]**2 +reverseRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac*y[PP2BCamCa4]
    f[PP2BCamCa2N] = forwardRate_CamCa2N_plus_PP2B__PP2BCamCa2N_reac*y[CamCa2N]*y[PP2B] -reverseRate_CamCa2N_plus_PP2B__PP2BCamCa2N_reac*y[PP2BCamCa2N] +forwardRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac*y[PP2BCam]*y[Ca_cyt]**2 -reverseRate_PP2BCam_plus_Ca2__PP2BCamCa2N_reac*y[PP2BCamCa2N] -forwardRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac*y[PP2BCamCa2N]*y[Ca_cyt]**2 +reverseRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac*y[PP2BCamCa4]
    f[PP2BCamCa4] = forwardRate_CamCa4_plus_PP2B__PP2BCamCa4_reac*y[CamCa4]*y[PP2B] -reverseRate_CamCa4_plus_PP2B__PP2BCamCa4_reac*y[PP2BCamCa4] +forwardRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac*y[PP2BCamCa2C]*y[Ca_cyt]**2 -reverseRate_PP2BCamCa2C_plus_Ca2__PP2BCamCa4_reac*y[PP2BCamCa4] +forwardRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac*y[PP2BCamCa2N]*y[Ca_cyt]**2 -reverseRate_PP2BCamCa2N_plus_Ca2__PP2BCamCa4_reac*y[PP2BCamCa4] -forwardRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac*y[D32p34]*y[PP2BCamCa4] +reverseRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac*y[D32p34PP2BCamCa4] +forwardRate_D32p34PP2BCamCa4__PP2BCamCa4_plus_D32_reac*y[D32p34PP2BCamCa4] -forwardRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac*y[D32p34PP1]*y[PP2BCamCa4] +reverseRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac*y[D32p34PP1PP2BCamCa4] +forwardRate_D32p34PP1PP2BCamCa4__PP1_plus_PP2BCamCa4_plus_D32_reac*y[D32p34PP1PP2BCamCa4]

    f[PKAcPP2AB56d] = forwardRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac*y[PP2AB56d]*y[PKAc] -reverseRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac*y[PKAcPP2AB56d] -forwardRate_PKAcPP2AB56d__PKAc_plus_pPP2A_reac*y[PKAcPP2AB56d]
    f[pPP2A] = forwardRate_PKAcPP2AB56d__PKAc_plus_pPP2A_reac*y[PKAcPP2AB56d] -forwardRate_pPP2A__PP2A_reac*y[pPP2A] -forwardRate_D32p75_plus_pPP2A__D32p75pPP2A_reac*y[D32p75]*y[pPP2A] +reverseRate_D32p75_plus_pPP2A__D32p75pPP2A_reac*y[D32p75pPP2A] +forwardRate_D32p75pPP2A__D32_plus_pPP2A_reac*y[D32p75pPP2A]

    f[D32] = -forwardRate_D32_plus_PKAc__D32PKAc_reac*y[D32]*y[PKAc] +reverseRate_D32_plus_PKAc__D32PKAc_reac*y[D32PKAc] +forwardRate_D32p34PP2BCamCa4__PP2BCamCa4_plus_D32_reac*y[D32p34PP2BCamCa4] +forwardRate_D32p34PP1PP2BCamCa4__PP1_plus_PP2BCamCa4_plus_D32_reac*y[D32p34PP1PP2BCamCa4] +forwardRate_D32p34PP2ABPR72__PP2ABPR72_plus_D32_reac*y[D32p34PP2ABPR72] +forwardRate_D32p34PP2AB56d__PP2AB56d_plus_D32_reac*y[D32p34PP2AB56d] +forwardRate_D32p34PP1PP2ABPR72__PP1_plus_PP2ABPR72_plus_D32_reac*y[D32p34PP1PP2ABPR72] +forwardRate_D32p34PP1PP2AB56d__PP1_plus_PP2AB56d_plus_D32_reac*y[D32p34PP1PP2AB56d] +forwardRate_D32p75pPP2A__D32_plus_pPP2A_reac*y[D32p75pPP2A] +forwardRate_D32p75PP2ABPR72__D32_plus_PP2ABPR72_reac*y[D32p75PP2ABPR72] +forwardRate_D32p75PP2AB56d__D32_plus_PP2AB56d_reac*y[D32p75PP2AB56d] +forwardRate_D32p75PP2Acal_D32_plus_PP2Acal_reac*y[D32p75PP2Acal]
    f[D32PKAc] = forwardRate_D32_plus_PKAc__D32PKAc_reac*y[D32]*y[PKAc] -reverseRate_D32_plus_PKAc__D32PKAc_reac*y[D32PKAc] -forwardRate_D32PKAc__PKAc_plus_D32p34_reac*y[D32PKAc]
    f[D32p34] = forwardRate_D32PKAc__PKAc_plus_D32p34_reac*y[D32PKAc] -forwardRate_D32p34_plus_PP1__D32p34pp1_reac*y[D32p34]*y[PP1] +reverseRate_D32p34_plus_PP1__D32p34pp1_reac*y[D32p34PP1] -forwardRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac*y[D32p34]*y[PP2BCamCa4] +reverseRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac*y[D32p34PP2BCamCa4] -forwardRate_D32p34_plus_PP2ABPR72__D32p34PP2ABPR72_reac*y[D32p34]*y[PP2ABPR72] +reverseRate_D32p34_plus_PP2ABPR72__D32p34PP2ABPR72_reac*y[D32p34PP2ABPR72] -forwardRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac*y[D32p34]*y[PP2AB56d] +reverseRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac*y[D32p34PP2AB56d]
    f[D32p34PP1] = forwardRate_D32p34_plus_PP1__D32p34pp1_reac*y[D32p34]*y[PP1] -reverseRate_D32p34_plus_PP1__D32p34pp1_reac*y[D32p34PP1] -forwardRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac*y[D32p34PP1]*y[PP2BCamCa4] +reverseRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac*y[D32p34PP1PP2BCamCa4] -forwardRate_D32p34pp1_plus_PP2ABPR72__D32p34PP1PP2ABPR72_reac*y[D32p34PP1]*y[PP2ABPR72] +reverseRate_D32p34pp1_plus_PP2ABPR72__D32p34PP1PP2ABPR72_reac*y[D32p34PP1PP2ABPR72] -forwardRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac*y[D32p34PP1]*y[PP2AB56d] +reverseRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac*y[D32p34PP1PP2AB56d]
    f[D32p34PP2BCamCa4] = forwardRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac*y[D32p34]*y[PP2BCamCa4] -reverseRate_D32p34_plus_PP2BCamCa4__D32p34PP2BCamCa4_reac*y[D32p34PP2BCamCa4]
    f[D32p34PP1PP2BCamCa4] = forwardRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac*y[D32p34PP1]*y[PP2BCamCa4] -reverseRate_D32p34PP1_plus_PP2BCamCa4__D32p34PP1PP2BCamCa4_reac*y[D32p34PP1PP2BCamCa4] -forwardRate_D32p34PP1PP2BCamCa4__PP1_plus_PP2BCamCa4_plus_D32_reac*y[D32p34PP1PP2BCamCa4]
    f[D32p34PP2ABPR72] = forwardRate_D32p34_plus_PP2ABPR72__D32p34PP2ABPR72_reac*y[D32p34]*y[PP2ABPR72] -reverseRate_D32p34_plus_PP2ABPR72__D32p34PP2ABPR72_reac*y[D32p34PP2ABPR72] -forwardRate_D32p34PP2ABPR72__PP2ABPR72_plus_D32_reac*y[D32p34PP2ABPR72]
    f[D32p34PP2AB56d] = forwardRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac*y[D32p34]*y[PP2AB56d] -reverseRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac*y[D32p34PP2AB56d] -forwardRate_D32p34PP2AB56d__PP2AB56d_plus_D32_reac*y[D32p34PP2AB56d]
    f[D32p34PP1PP2ABPR72] = forwardRate_D32p34pp1_plus_PP2ABPR72__D32p34PP1PP2ABPR72_reac*y[D32p34PP1]*y[PP2ABPR72] -reverseRate_D32p34pp1_plus_PP2ABPR72__D32p34PP1PP2ABPR72_reac*y[D32p34PP1PP2ABPR72] -forwardRate_D32p34PP1PP2ABPR72__PP1_plus_PP2ABPR72_plus_D32_reac*y[D32p34PP1PP2ABPR72]
    f[D32p34PP1PP2AB56d] = forwardRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac*y[D32p34PP1]*y[PP2AB56d] -reverseRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac*y[D32p34PP1PP2AB56d] -forwardRate_D32p34PP1PP2AB56d__PP1_plus_PP2AB56d_plus_D32_reac*y[D32p34PP1PP2AB56d]

    f[PP2AB56d] = -forwardRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac*y[PP2AB56d]*y[PKAc] +reverseRate_PKAc_plus_PP2A__PKAcPP2AB56d_reac*y[PKAcPP2AB56d] +forwardRate_pPP2A__PP2A_reac*y[pPP2A] -forwardRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac*y[D32p34]*y[PP2AB56d] +reverseRate_D32p34_plus_PP2AB56d__D32p34PP2AB56d_reac*y[D32p34PP2AB56d] +forwardRate_D32p34PP2AB56d__PP2AB56d_plus_D32_reac*y[D32p34PP2AB56d] -forwardRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac*y[D32p34PP1]*y[PP2AB56d] +reverseRate_D32p34pp1_plus_PP2AB56d__D32p34PP1PP2AB56d_reac*y[D32p34PP1PP2AB56d] +forwardRate_D32p34PP1PP2AB56d__PP1_plus_PP2AB56d_plus_D32_reac*y[D32p34PP1PP2AB56d] -forwardRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac*y[D32p75]*y[PP2AB56d] +reverseRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac*y[D32p75PP2AB56d] +forwardRate_D32p75PP2AB56d__D32_plus_PP2AB56d_reac*y[D32p75PP2AB56d]
    f[Cdk5] = 0.
    f[Cdk5D32] = 0.

    f[D32p75] = -forwardRate_D32p75_plus_PKAc__D32p75PKAc_reac*y[D32p75]*y[PKAc] +reverseRate_D32p75_plus_PKAc__D32p75PKAc_reac*y[D32p75PKAc] -forwardRate_D32p75_plus_pPP2A__D32p75pPP2A_reac*y[D32p75]*y[pPP2A] +reverseRate_D32p75_plus_pPP2A__D32p75pPP2A_reac*y[D32p75pPP2A] -forwardRate_D32p75_plus_PP2ABPR72__D32p75PP2ABPR72_reac*y[D32p75]*y[PP2ABPR72] +reverseRate_D32p75_plus_PP2ABPR72__D32p75PP2ABPR72_reac*y[D32p75PP2ABPR72] -forwardRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac*y[D32p75]*y[PP2AB56d] +reverseRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac*y[D32p75PP2AB56d] -forwardRate_D32p75_plus_PP2Acal__D32p75PP2Acal_reac*y[D32p75]*y[PP2Acal] +reverseRate_D32p75_plus_PP2Acal__D32p75PP2Acal_reac*y[D32p75PP2Acal]
    f[D32p75PKAc] = forwardRate_D32p75_plus_PKAc__D32p75PKAc_reac*y[D32p75]*y[PKAc] -reverseRate_D32p75_plus_PKAc__D32p75PKAc_reac*y[D32p75PKAc]
    f[D32p75pPP2A] = forwardRate_D32p75_plus_pPP2A__D32p75pPP2A_reac*y[D32p75]*y[pPP2A] -reverseRate_D32p75_plus_pPP2A__D32p75pPP2A_reac*y[D32p75pPP2A] -forwardRate_D32p75pPP2A__D32_plus_pPP2A_reac*y[D32p75pPP2A]
    f[D32p75PP2ABPR72] = forwardRate_D32p75_plus_PP2ABPR72__D32p75PP2ABPR72_reac*y[D32p75]*y[PP2ABPR72] -reverseRate_D32p75_plus_PP2ABPR72__D32p75PP2ABPR72_reac*y[D32p75PP2ABPR72] -forwardRate_D32p75PP2ABPR72__D32_plus_PP2ABPR72_reac*y[D32p75PP2ABPR72]
    f[D32p75PP2AB56d] = forwardRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac*y[D32p75]*y[PP2AB56d] -reverseRate_D32p75_plus_PP2AB56d__D32p75PP2AB56d_reac*y[D32p75PP2AB56d] -forwardRate_D32p75PP2AB56d__D32_plus_PP2AB56d_reac*y[D32p75PP2AB56d]
    f[D32p75PP2Acal] = forwardRate_D32p75_plus_PP2Acal__D32p75PP2Acal_reac*y[D32p75]*y[PP2Acal] -reverseRate_D32p75_plus_PP2Acal__D32p75PP2Acal_reac*y[D32p75PP2Acal] -forwardRate_D32p75PP2Acal_D32_plus_PP2Acal_reac*y[D32p75PP2Acal]

    f[CK] = 0.
    f[CKCamCa4] = 0.
    f[CKpCamCa4] = 0.
    f[CKp] = 0.
    f[CKpPP1] = 0.
    f[CKpCamCa4PP1] = 0.
    f[Dgl_CKCamCa4] = 0.
    f[pDgl] = -forwardRate_pDgl_dephos*y[pDgl] -forwardRate_Ca_plus_pDlg__CapDgl*y[Ca_cyt]*y[pDgl] +reverseRate_Ca_plus_pDlg__CapDgl*y[CapDgl]
    f[Dgl_CKpCamCa4] = 0.
    f[CaDgl_CKpCamCa4] = 0.
    f[CapDgl] = forwardRate_Ca_plus_pDlg__CapDgl*y[Ca_cyt]*y[pDgl] -reverseRate_Ca_plus_pDlg__CapDgl*y[CapDgl] -forwardRate_Dag_plus_CapDlg__DagCapDgl*y[Dag]*y[CapDgl] +reverseRate_Dag_plus_CapDlg__DagCapDgl*y[DagCapDgl] +forwardRate_DagCapDgl__CapDgl_plus_2ag*y[DagCapDgl]
    f[Dgl_CKp] = 0.
    f[DagCapDgl] = forwardRate_Dag_plus_CapDlg__DagCapDgl*y[Dag]*y[CapDgl] -reverseRate_Dag_plus_CapDlg__DagCapDgl*y[DagCapDgl] -forwardRate_DagCapDgl__CapDgl_plus_2ag*y[DagCapDgl]
    f[AKAR3] = 0.
    f[PKAcAKAR3] = 0.
    f[pAKAR3] = 0.
    f[PP1pAKAR3] = 0.

    f[AC1] = -forwardRate_GsaGTP_AC1_reac*y[GsaGTP]*y[AC1] +reverseRate_GsaGTP_AC1_reac*y[AC1Gsa] -forwardRate_AC1CamCa4_Reac*y[AC1]*y[CamCa4] +reverseRate_AC1CamCa4_Reac*y[AC1CamCa4]
    f[AC1Gsa] = forwardRate_GsaGTP_AC1_reac*y[GsaGTP]*y[AC1] -reverseRate_GsaGTP_AC1_reac*y[AC1Gsa] -forwardRate_AC1GsaGTP_CamCa4_Reac*y[AC1Gsa]*y[CamCa4] +reverseRate_AC1GsaGTP_CamCa4_Reac*y[AC1GsaCamCa4]
    f[AC1GsaCamCa4] = forwardRate_AC1GsaGTP_CamCa4_Reac*y[AC1Gsa]*y[CamCa4] -reverseRate_AC1GsaGTP_CamCa4_Reac*y[AC1GsaCamCa4] -forwardRate_AC1GsaGTPCamCa4_ATP_Reac*y[AC1GsaCamCa4]*y[ATP] +reverseRate_AC1GsaGTPCamCa4_ATP_Reac*y[AC1GsaCamCa4ATP] +forwardRate_AC1GsaGTPCamCa4ATP_diss*y[AC1GsaCamCa4ATP]
    f[AC1GsaCamCa4ATP] = forwardRate_AC1GsaGTPCamCa4_ATP_Reac*y[AC1GsaCamCa4]*y[ATP] -reverseRate_AC1GsaGTPCamCa4_ATP_Reac*y[AC1GsaCamCa4ATP] -forwardRate_AC1GsaGTPCamCa4ATP_diss*y[AC1GsaCamCa4ATP]
    f[AC1CamCa4] = forwardRate_AC1CamCa4_Reac*y[AC1]*y[CamCa4] -reverseRate_AC1CamCa4_Reac*y[AC1CamCa4] -forwardRate_AC1CamCa4ATP_bind*y[AC1CamCa4]*y[ATP] +reverseRate_AC1CamCa4ATP_bind*y[AC1CamCa4ATP] +forwardRate_AC1CamCa4ATP_disso*y[AC1CamCa4ATP]
    f[AC1CamCa4ATP] = forwardRate_AC1CamCa4ATP_bind*y[AC1CamCa4]*y[ATP] -reverseRate_AC1CamCa4ATP_bind*y[AC1CamCa4ATP] -forwardRate_AC1CamCa4ATP_disso*y[AC1CamCa4ATP]


    gsl_vector_free(LTPwin_tab)
    gsl_vector_free(minf_caL13_tab)
    gsl_vector_free(hinf_caL13_tab)
    gsl_vector_free(mtau_caL13_tab)
    gsl_vector_free(B_NMDA_tab)

    return GSL_SUCCESS

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ----------- MAIN ----------- #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

def main(init_array, params_array, nb_eq, t_start, t_end, time_step, h_step = 1e-6, rel_err = 1e-7, abs_err = 1e-7): # Function which can be call in python

# --------------------------------------------------------------------------  #
# --------------------------------- INITIALISATION ---------------------------------  #
# --------------------------------------------------------------------------  #

    cdef int len_init = int(len(init_array)) # Length of the list of init

    # allocate number * sizeof(double) bytes of memory
    cdef double *init_list = <double *>malloc((len_init) * sizeof(double))
    if not init_list: # If allocation doesn't work
        raise MemoryError()

    cdef int l
    for l from 0 <= l < len_init: # Fill the rest of the array of initialisation
        init_list[l] = init_array[l]

    # Initialisation of time computed
    cdef double t = t_start

    cdef int status # Status of the solving
    cdef double ti # Time to compute


# --------------------------------------------------------------------------  #
# --------------------------------- PARAMS ---------------------------------  #
# --------------------------------------------------------------------------  #

    cdef int nb_ode = int(nb_eq) # Number of equation

# ------------------------ EXTRA PARAMS ----------------------------------  #
    cdef int len_params = int(len(params_array)) # Length of the list of params

    # allocate number * sizeof(double) bytes of memory
    cdef double *extra_params = <double *>malloc((len_params) * sizeof(double))
    if not extra_params: # If allocation doesn't work
        raise MemoryError()

    cdef int j
    for j from 0 <= j < len_params: # Fill the rest of the array of parameters
        extra_params[j] = params_array[j]

# --------------------------------------------------------------------------  #

    cdef int nb_time_step # nb_time_step
    nb_time_step = int((t_end -t_start) / time_step)
    #printf("%d \n", nb_time_step)


    # Initialisation of an empty np.ndarray #
    #    cdef np.ndarray[np.float64_t, ndim = 2] res = np.zeros((t_end+1, nb_ode+1), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] res = np.zeros((nb_time_step+1, nb_ode+1), dtype=np.float64)


# --------------------------------------------------------------------------  #
# --------------------------------- ODE ---------------------------------  #
# --------------------------------------------------------------------------  #

    cdef Py_ssize_t i
    cdef Py_ssize_t m_it

    cdef gsl_odeiv2_system sys
    sys.function = func
    sys.jacobian = NULL
    sys.dimension = nb_ode
    sys.params = <void*> extra_params

    cdef gsl_odeiv2_driver * d
    d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_rk8pd,
        h_step, abs_err, rel_err)

    for i from 0 <= i <= nb_time_step:
        ti = t_start + i * time_step # Time to compute

        status = gsl_odeiv2_driver_apply (d, &t, ti, init_list)

        if (status != GSL_SUCCESS):
            print("error, return value=%d\n" % status)
            break



# --------------------- Fill the array of result ---------------------------  #
        res[i,0] = t

        for m_it from 0 <= m_it < len_init: # Fill the rest of the array of parameters
            res[i,m_it+1] = init_list[m_it]


    free(init_list)
    free(extra_params)
    gsl_odeiv2_driver_free(d) # Release the system

    return res

