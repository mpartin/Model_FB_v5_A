
import Model_FB_cython as MSc
import numpy as np
import pandas as pd
import time
from copy import deepcopy
start_time = time.time()

import collections as coll

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ----------- FUNCTIONS ----------- #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

def heaviside_func(x):
    if (x == 0):
        heaviside = 0.5
    elif (x < 0):
        heaviside = 0
    else:
        heaviside = 1.0
    return heaviside

def lin_interpTable_brds(Table, li, i_point):
    ip=int(np.floor(i_point) -1)

    if ip < 0:
        res = Table[0]
    elif ip < li-1:
        res = Table[ip] +(Table[ip+1] -Table[ip])*(i_point-1 -ip)
    else:
        res = Table[li-1]
    return res

def efun(x):
    if (np.abs(x)<1e-4):
        efun = 1-x/2
    elif (x<=85):
        efun = x/(np.exp(x)-1)
    else:
        efun = 0
    return efun


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ----------- PARAMETERS ----------- #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

t_start = 0.0
t_end = 50000.
time_step = 1.

init_dict_ord_UP = coll.OrderedDict([
                        ('h_caL13', 0.991474296812), # No unit
                        ('m_caL13', 0.00398239753677), # No unit
                        ('o_AMPA', 0.0), # No unit | prob
                        ('d_AMPA', 0.0), # No unit | prob
                        ('o_NMDA', 0.0), # No unit | prob
                        ('h_CICR', 0.806764947937), # No unit | prob
                        ('Ca_cyt', 121.328558067), # nM
                        ('Ca_ER', 11674.4333459), # nM
                        ('IP3', 32.0601588447), # nM
                        ('DAG', 5.82936737534), # nM
                        ('DAGLP', 4.19724618359e-07), # No unit
                        ('twoAG', 0.00326167154646), # nM
                        ('AEA', 6.10345406741), # nM
                        ('fpre', 1.), # No unit
                        ('I1P', 42.3801795171), # nM
                        ('PP1', 0.939404212327), # nM
                        ('V', -69.9990159858), # mV
                        ('o_CB1R', 1.41213798961e-06), # No unit | prob
                        ('d_CB1R', 0.0123020404004), # No unit | prob
                        ('B1', 116.895993047), # nM
                        ('B2', 352.196488998), # nM
                        ('B3', 113.729213221), # nM
                        ('B4', 44.8638945164), # nM
                        ('B5', 1003.31918585), # nM
                        ('B6', 372.102986328), # nM
                        ('B7', 381.523343988), # nM
                        ('B8', 57.654316838), # nM
                        ('B9', 3059.26332527), # nM
                        ('B10', 1433.60954626), # nM
                        ('B11', 749.219542768), # nM
                        ('B12', 11318.0618456), # nM
                        ('B13', 14135.0094869) # nM
                        ])



init_dict_ord_DOWN = coll.OrderedDict([
                        ('h_caL13', 0.991474296812), # No unit
                        ('m_caL13', 0.00398239753677), # No unit
                        ('o_AMPA', 0.0), # No unit | prob
                        ('d_AMPA', 0.0), # No unit | prob
                        ('o_NMDA', 0.0), # No unit | prob
                        ('h_CICR', 0.824666350479), # No unit | prob
                        ('Ca_cyt', 121.328558067), # nM
                        ('Ca_ER', 3685.55545234), # nM
                        ('IP3', 57.2919012476), # nM
                        ('DAG', 5.73491809464), # nM
                        ('DAGLP', 4.19724618358e-07), # No unit
                        ('twoAG', 0.00320883499792), # nM
                        ('AEA', 6.10345406741), # nM
                        ('fpre', 1.), # No unit
                        ('I1P', 42.3801795171), # nM
                        ('PP1', 0.939404212327), # nM
                        ('V', -69.9990159858), # mV
                        ('o_CB1R', 1.4121086643e-06), # No unit | prob
                        ('d_CB1R', 0.0123017849288), # No unit | prob
                        ('B1', 233.172566426), # nM
                        ('B2', 3.43009952762), # nM
                        ('B3', 0.288928438476), # nM
                        ('B4', 0.137575875966), # nM
                        ('B5', 0.0363703104237), # nM
                        ('B6', 0.00412797307999), # nM
                        ('B7', 0.00425044688892), # nM
                        ('B8', 0.000125159430492), # nM
                        ('B9', 0.000327011527855), # nM
                        ('B10', 4.54920293517e-05), # nM
                        ('B11', 3.07865465156e-05), # nM
                        ('B12', 2.79756013723e-06), # nM
                        ('B13', 1.32247853165e-08) # nM
                        ])

init_dict_ord_ILYA = coll.OrderedDict([
                        ('h_caL13', 0.991821379067138), # No unit
                        ('m_caL13', 0.0036977671871038817), # No unit
                        ('o_AMPA', 0.0), # No unit | prob
                        ('d_AMPA', 0.0), # No unit | prob
                        ('o_NMDA', 0.0), # No unit | prob
                        ('h_CICR', 0.8246676668946951), # No unit | prob
                        ('Ca_cyt', 0.12132718966407073*1e3), # nM
                        ('Ca_ER', 63.348087686853646*1e3), # nM
                        ('IP3', 0.05729140044675357*1e3), # nM
                        ('DAG', 0.005734867663641929*1e3), # nM
                        ('DAGLP', 4.1969621599776083e-07), # No unit
                        ('twoAG', 3.2085896623941232e-06*1e3), # nM
                        ('AEA', 0.006103384809978344*1e3), # nM
                        ('fpre', 1.0), # No unit
                        ('I1P', 0.042380592866431144*1e3), # nM
                        ('PP1', 0.000939395093112328*1e3), # nM
                        ('V', -69.99901620452822), # mV
                        ('o_CB1R', 3.4373437854140236e-07), # No unit | prob
                        ('d_CB1R', 0.002994487796947427), # No unit | prob
                        ('B1', 0.23316029213700182*1e3), # nM
                        ('B2', 0.0034298074889746086*1e3), # nM
                        ('B3', 0.00028889779878196254*1e3), # nM
                        ('B4', 0.00013756133483052541*1e3), # nM
                        ('B5', 3.6365976788029681e-05*1e3), # nM
                        ('B6', 4.1274017451676494e-06*1e3), # nM
                        ('B7', 4.2498580055485264e-06*1e3), # nM
                        ('B8', 1.2513942987290664e-07*1e3), # nM
                        ('B9', 3.2696082960591099e-07*1e3), # nM
                        ('B10', 4.5484170099234244e-08*1e3), # nM
                        ('B11', 3.078127923587743e-08*1e3), # nM
                        ('B12', 2.7970211543431621e-09*1e3), # nM
                        ('B13', 1.3221817318283754e-11*1e3) # nM
                        ])


# ------ Blackwell ----------------- #

init_dict_ord_nM = coll.OrderedDict([
                                    ('Ca', 81),
                                    ('CaOut', 1997000),
                                    ('CaOutLeak', 0),
                                    #-- total calbindin = 160 uM -->
                                    ('Calbin', 159980),
                                    ('CalbinC', 20),

                                    #-- membrane molecules -->
                                    ('Leak', 0),
                                    ('pmca', 0),
                                    ('pmcaCa', 0),
                                    ('ncx', 0),
                                    ('ncxCa', 0),

                                    #-- cytosolic molecules -->
                                    #-- total calmodulin = 9000 nM -->
                                    ('Cam', 3491),
                                    ('CamCa2C', 185),
                                    ('CamCa2N', 35),
                                    ('CamCa4', 4.),

                                    #-- Dopamine, ACh, Glu and their buffers: cytosolic molecules -->
                                    ('Da', 10),
                                    ('Dbuf', 0),
                                    ('DaDbuf', 0),
                                    ('DaOut', 200000),
                                    ('ACh', 95),
                                    ('Glu', 10),
                                    ('GluOut', 200000),
                                    ('Gbuf', 0),
                                    ('GluGbuf', 0),

                                    #-- D1 associated membrane molecules -->
                                    ('D1R', 0),
                                    ('DaD1R', 0),
                                    ('Gsabg', 0),
                                    ('DaD1RGs', 0),
                                    ('GsD1R', 0),
                                    ('GsaGTP', 0),
                                    ('GsaGDP', 0),
                                    ('Gbg', 0),

                                    #-- mGluR & Gq associated membrane molecules -->
                                    ('MgluR', 0),
                                    ('GluMgluR', 0),
                                    ('GluMgluRdesens', 0),

                                    ('m1R', 0),
                                    ('AChm1R', 0),
                                    ('m1RGq', 0),
                                    ('AChm1RGq', 0),

                                    ('Gqabg', 0),
                                    ('GluMgluRGq', 0),
                                    ('GqaGTP', 0),
                                    ('GqaGDP', 0),

                                    #--cyclases -->
                                    ('AC5', 0),
                                    ('AC5Gsa', 0),
                                    ('AC5GsaATP', 0),
                                    ('AC5Gia', 0),
                                    ('AC5GsaGia', 0),
                                    ('AC5GsaGiaATP', 0),
                                    ('AC5Ca', 0),
                                    ('AC5GsaCa', 0),
                                    ('AC5GsaCaATP', 0),
                                    ('PDE4', 275),
                                    ('PDE4cAMP', 0),
                                    ('PKAcPDE4', 0),
                                    ('PKAcPDE4cAMP', 0),
                                    ('pPDE4', 5),
                                    ('pPDE4PP1', 0),
                                    ('pPDE4cAMP', 0),

                                    ('PDE2', 560),
                                    ('PDE2cAMP', 0),
                                    ('PDE2cAMP2', 0),

                                    #-- cytosolic molecules -->
                                    ('ATP', 1998900),
                                    ('cAMP', 50),
                                    ('AMP', 0),

                                    ('PDE1', 1700),
                                    ('PDE1CamCa4', 300),
                                    ('PDE1CamCa4cAMP', 0),

                                    #-- membrane molecules -->

                                    ('PlcCa', 0),
                                    ('PlcCaPip2', 0),
                                    ('PlcGqa', 0),
                                    ('PlcCaGqa', 0),
                                    ('PlcCaGqaPip2', 0),

                                    ('Dag', 0),
                                    ('DagKdag', 0),
                                    ('PA', 0),

                                    ('DagCaDgl', 0),
                                    ('CaDgl', 0),
                                    ('Dgl', 0),

                                    #-- The next four are cytosolic and should be initialized non-zero -->
                                    ('Ip3', 0),
                                    ('Ip3degrad', 0),

                                    ('two_agDegrad', 0),
                                    ('two_ag', 0),

                                    #-- where did PKC quantity of 15 uM (from BoHung Paper) come from ???-->
                                    ('Pkc', 9950),
                                    ('PkcCa', 50),
                                    ('Pkct', 0),

                                    #-- cytosolic molecules -->
                                    #-- total PP2B = 5000 nM, PP2ABPR72=1400, PP2AB56d=1400-->
                                    ('PP2B', 15),
                                    ('PP2BCam', 3000),
                                    ('PP2BCamCa2C', 1710),
                                    ('PP2BCamCa2N', 260),
                                    ('PP2BCamCa4', 15),
                                    ('PP1_Blackwell', 3400),
                                    ('PP2AB56d', 1100),
                                    ('PKAcPP2AB56d', 0),
                                    ('pPP2A', 20),
                                    ('PP2ABPR72', 1100),
                                    ('PP2Acal', 20),
                                    ('CK', 11930),
                                    ('CKCamCa4', 70),
                                    ('CKpCamCa4', 0),
                                    ('CKp', 0),
                                    ('Complex', 0),
                                    ('pComplex', 0),
                                    ('CKpPP1', 0),
                                    ('CKpCamCa4PP1', 0),

                                    #-- total DARPP32 = 50 uM, 12 uM D32p75 and 1% D32p32 at basal -->
                                    ('D32', 37060),
                                    ('D32PKAc', 0),
                                    ('D32p34', 0),
                                    ('D32p34PP1', 580),
                                    ('D32p34PP2BCamCa4', 0),
                                    ('D32p34PP1PP2BCamCa4', 0),
                                    ('D32p34PP2ABPR72', 0),
                                    ('D32p34PP2AB56d', 0),
                                    ('D32p34PP1PP2ABPR72', 10),
                                    ('D32p34PP1PP2AB56d', 10),
                                    ('Cdk5', 370),
                                    ('Cdk5D32', 1215),
                                    ('D32p75', 10650),
                                    ('D32p75PKAc', 0),
                                    ('D32p75PP2ABPR72', 260),
                                    ('D32p75PP2AB56d', 260),
                                    ('D32p75PP2Acal', 10),
                                    ('D32p75pPP2A', 10),

                                    ('AKAR3', 0),
                                    ('pAKAR3', 0)
                                    ])

sub_memb_vol_micrometer = 0.15 # micro-meter | Blackwell2013

init_dict_ord_PicoSD = coll.OrderedDict([
                                ('Leak', 900/sub_memb_vol_micrometer),
                                ('CaOutLeak', 0/sub_memb_vol_micrometer),
                                ('pmca', 71/sub_memb_vol_micrometer),
                                ('pmcaCa', 29/sub_memb_vol_micrometer),
                                ('ncx', 2840/sub_memb_vol_micrometer),
                                ('ncxCa', 160/sub_memb_vol_micrometer),
                                ('D1R', 0/sub_memb_vol_micrometer),
                                ('DaD1R', 0/sub_memb_vol_micrometer),
                                ('Gsabg', 1745/sub_memb_vol_micrometer),
                                ('DaD1RGs', 3/sub_memb_vol_micrometer),
                                ('GsD1R', 197/sub_memb_vol_micrometer),
                                ('pDaD1RGs', 0/sub_memb_vol_micrometer),
                                ('PKAcDaD1RGs', 0/sub_memb_vol_micrometer),
                                ('GsaGDP', 0/sub_memb_vol_micrometer),
                                ('GsaGTP', 0/sub_memb_vol_micrometer),
                                ('Gbg', 382/sub_memb_vol_micrometer),
                                ('m4R', 5/sub_memb_vol_micrometer),
                                ('AChm4R', 0/sub_memb_vol_micrometer),
                                ('Giabg', 1478/sub_memb_vol_micrometer),
                                ('AChm4RGi', 23/sub_memb_vol_micrometer),
                                ('Gim4R', 172/sub_memb_vol_micrometer),
                                ('GiaGDP', 0/sub_memb_vol_micrometer),
                                ('GiaGTP', 22/sub_memb_vol_micrometer),
                                ('AC5', 190/sub_memb_vol_micrometer),
                                ('AC5Gsa', 0/sub_memb_vol_micrometer),
                                ('AC5GsaATP', 5/sub_memb_vol_micrometer),
                                ('AC5Ca', 0/sub_memb_vol_micrometer),
                                ('AC5CaGsa', 0/sub_memb_vol_micrometer),
                                ('AC5CaGsaATP', 0/sub_memb_vol_micrometer),
                                ('AC5Gia', 255/sub_memb_vol_micrometer),
                                ('AC5GsaGia', 0/sub_memb_vol_micrometer),
                                ('AC5GsaGiaATP', 50/sub_memb_vol_micrometer),
                                ('AC1', 50/sub_memb_vol_micrometer),
                                ('AC1Gsa', 0/sub_memb_vol_micrometer),
                                ('AC1GsaCamCa4', 0/sub_memb_vol_micrometer),
                                ('AC1GsaCamCa4ATP', 0/sub_memb_vol_micrometer),
                                ('AC1CamCa4', 0/sub_memb_vol_micrometer),
                                ('AC1CamCa4ATP', 0/sub_memb_vol_micrometer),
                                ('PDE10', 245/sub_memb_vol_micrometer),
                                ('PDE10cAMP', 35/sub_memb_vol_micrometer),
                                ('PKAcPDE10', 0/sub_memb_vol_micrometer),
                                ('PKAcPDE10cAMP', 0/sub_memb_vol_micrometer),
                                ('pPDE10', 0/sub_memb_vol_micrometer),
                                ('pPDE10PP1', 0/sub_memb_vol_micrometer),
                                ('pPDE10cAMP', 0/sub_memb_vol_micrometer),
                                ('m1R', 50/sub_memb_vol_micrometer),
                                ('m1RGq', 50/sub_memb_vol_micrometer),
                                ('Gqabg', 950/sub_memb_vol_micrometer),
                                ('Plc', 177/sub_memb_vol_micrometer),
                                ('PlcCa', 8/sub_memb_vol_micrometer),
                                ('Pip2', 9565/sub_memb_vol_micrometer),
                                ('PlcCaPip2', 15/sub_memb_vol_micrometer),
                                ('Dag', 40/sub_memb_vol_micrometer),
                                ('DagK', 140/sub_memb_vol_micrometer),
                                ('Dgl', 308/sub_memb_vol_micrometer),
                                ('CaDgl', 112/sub_memb_vol_micrometer),
                                ('PIkinase', 280/sub_memb_vol_micrometer),
                                ('Ip3degPIk', 0/sub_memb_vol_micrometer),
                                ('Ip3', 20/sub_memb_vol_micrometer),
                                ('Ip3degrad', 0/sub_memb_vol_micrometer),
                                ('PKA', 812/sub_memb_vol_micrometer),
                                ('PKAcAMP2', 168/sub_memb_vol_micrometer),
                                ('PKAcAMP4', 20/sub_memb_vol_micrometer),
                                ('PKAc', 0/sub_memb_vol_micrometer),
                                ('PKAr', 0/sub_memb_vol_micrometer)
                                ])

init_dict_ord_Rxn_missing_var = coll.OrderedDict([
                                        ('PKAcAKAR3', 0.),
                                        ('CapDgl', 0.),
                                        ('Pkct', 0.),
                                        ('Dgl_CKp', 0.),
                                        ('pComplex', 0.),
                                        ('CaDgl_CKpCamCa4', 0.),
                                        ('AC5CaGsa', 0.),
                                        ('Dgl_CKpCamCa4', 0.),
                                        ('CaDgl_CKCamCa4', 0.),
                                        ('Complex', 0.),
                                        ('AC5CaGsaATP', 0.),
                                        ('pDgl', 0.),
                                        ('PP1pAKAR3', 0.),
                                        ('Dgl_CKCamCa4', 0.),
                                        ('MgluRGq', 0.),
                                        ('DagCapDgl', 0.),
                                        ('PkcCaDag', 0.)
                                        ])


init_dict_ord = init_dict_ord_nM
init_dict_ord.update(init_dict_ord_PicoSD) # Replace value of the "soma" compartment by value of the sub-membrane
init_dict_ord.update(init_dict_ord_Rxn_missing_var)

# ------ ODE in order ------- #
list_ODE = ['Ca', 'pmca', 'pmcaCa', 'CaOut', 'ncx', 'ncxCa', 'Leak', 'CaOutLeak', 'Calbin', 'CalbinC', 'CamCa4', 'CamCa2N', 'DaDbuf', 'Dbuf', 'DaOut', 'Da', 'DaD1R', 'Gsabg', 'DaD1RGs', 'D1R', 'GsD1R', 'GsaGTP', 'Gbg', 'PKAc', 'PKAcDaD1RGs', 'pDaD1RGs', 'GluGbuf', 'Gbuf', 'Glu', 'GluOut', 'GluMgluR', 'GluMgluRdesens', 'MgluR', 'Gqabg', 'MgluRGq', 'GluMgluRGq', 'GqaGTP', 'ACh', 'm1R', 'AChm1R', 'AChm1RGq', 'm1RGq', 'm4R', 'AChm4R', 'Giabg', 'AChm4RGi', 'Gim4R', 'GsaGDP', 'GqaGDP', 'GiaGTP', 'GiaGDP', 'AC5', 'AC5Gsa', 'ATP', 'AC5GsaATP', 'cAMP', 'AC5Gia', 'AC5GsaGia', 'AC5GsaGiaATP', 'PDE1', 'PDE1CamCa4', 'PDE1CamCa4cAMP', 'AMP', 'PDE10', 'PDE10cAMP', 'pPDE10', 'pPDE10cAMP', 'PKAcPDE10', 'PKAcPDE10cAMP', 'PP1_Blackwell', 'pPDE10PP1', 'PDE2', 'PDE2cAMP', 'PDE2cAMP2', 'PDE4', 'PDE4cAMP', 'pPDE4', 'pPDE4cAMP', 'PKAcPDE4', 'PKAcPDE4cAMP', 'pPDE4PP1', 'Plc', 'PlcCa', 'PlcCaGqa', 'PlcGqa', 'Pip2', 'PlcCaPip2', 'Ip3', 'Dag', 'PlcCaGqaPip2', 'Dgl', 'CaDgl', 'DagCaDgl', 'two_ag', 'Ip3degrad', 'PIkinase', 'Ip3degPIk', 'two_agDegrad', 'DagK', 'DagKdag', 'PA', 'Pkc', 'PkcCa', 'PkcCaDag', 'PKA', 'PKAcAMP2', 'PKAcAMP4', 'PKAr', 'Cam', 'PP2B', 'PP2BCam', 'CamCa2C', 'PP2BCamCa2C', 'PP2BCamCa2N', 'PP2BCamCa4', 'PKAcPP2AB56d', 'pPP2A', 'PP2ABPR72', 'PP2Acal', 'D32', 'D32PKAc', 'D32p34', 'D32p34PP1', 'D32p34PP2BCamCa4', 'D32p34PP1PP2BCamCa4', 'D32p34PP2ABPR72', 'D32p34PP2AB56d', 'D32p34PP1PP2ABPR72', 'D32p34PP1PP2AB56d', 'PP2AB56d', 'Cdk5', 'Cdk5D32', 'D32p75', 'D32p75PKAc', 'D32p75pPP2A', 'D32p75PP2ABPR72', 'D32p75PP2AB56d', 'D32p75PP2Acal', 'CK', 'CKCamCa4', 'CKpCamCa4', 'CKp', 'CKpPP1', 'CKpCamCa4PP1', 'Dgl_CKCamCa4', 'pDgl', 'Dgl_CKpCamCa4', 'CaDgl_CKpCamCa4', 'CapDgl', 'Dgl_CKp', 'DagCapDgl', 'AKAR3', 'PKAcAKAR3', 'pAKAR3', 'PP1pAKAR3', 'AC1', 'AC1Gsa', 'AC1GsaCamCa4', 'AC1GsaCamCa4ATP', 'AC1CamCa4', 'AC1CamCa4ATP']
init_dict_ord = coll.OrderedDict([(k,init_dict_ord[k]) for k in list_ODE if k in list_ODE])

init_dict_Blackwell = init_dict_ord

init_dict = init_dict_ord_ILYA
init_dict.update(init_dict_Blackwell)

init_array = init_dict.values()
init_keys = init_dict.keys()


# ------------------------------------------------------ #

params_dict_ord_caL13 = coll.OrderedDict([
    ("on_caL13", 1.), # No unit
    ("pcaLbar", 1.02e-09), # microL.ms-1
    ("mslope_caL13", -6.7), # mV
    ("hshift_caL13", 0.0), # not used
    ("vm_caL13", -8.124), # mV
    ("mshift_caL13", 0.0), # not used
    ("hslope_caL13", 11.9), # mV
    ("kpr_caL13", 31.4), # mV
    ("c_caL13", 0.0398), # mV-1.ms-1
    ("k_caL13", 9.005), # mV
    ("hvhalf_caL13", -13.4), # mV
    ("mvhalf_caL13", -33), # mV
    ("cpr_caL13", 0.9900), # ms-1
    ("htau_caL13", 44.3), # ms
    ("hqfact_caL13", 3.0), # No unit
    ("qfact_caL13", 3.0) # No unit
	])
params_dict_ord_TRPV1 = coll.OrderedDict([
    ("on_TRPV1", 1.), # No unit
    ("gTRPV1", 0.0003), # nS
    ("p_ca_TRPV1", 2.23204558561e-52), # microL.ms-1
    ("C_TRPV1", 23367.0), # No unit
    ("D_TRPV1", 1100.0), # No unit
    ("DH_TRPV1", 205000.0), # kJ.mol-1
    ("KD_TRPV1", 500.), # nM
    ("J0_TRPV1", 0.0169), # No unit
    ("L_TRPV1", 0.00042), # No unit
    ("P_TRPV1", 750.0), # No unit
    ("z_TRPV1", 0.6), # No unit
    ("DS_TRPV1", 615.0), # J.mol-1.K-1
    ("K_TRPV1", 0.00182634305618) # No unit
	])
params_dict_ord_I_to_Ca_flux = coll.OrderedDict([
    ("NMDA", 70.0), # microM.pCoulomb-1 -> microM.pA-1.s-1 -> nM.pA-1.ms-1
    ("VDCC", 84.0), # microM.pCoulomb-1 -> microM.pA-1.s-1 -> nM.pA-1.ms-1
    ("TRPV1", 310.) # microM.pCoulomb-1 -> microM.pA-1.s-1 -> nM.pA-1.ms-1
	])
params_dict_ord_CaBuff = coll.OrderedDict([
    ("BT_CaBuff", 4500.), # nM
    ("Cab", 100.), # nM
    ("tauCab", 7.), # ms
    ("KdB_CaBuff", 500.) # nM
	])
params_dict_ord_CICR = coll.OrderedDict([
    ("a2_CICR", 0.5e-6), # nM-1.ms-1
    ("ver_CICR", 8.0), # nM.ms-1
    ("d3_CICR", 943.4), # nM
    ("rc_CICR", 4.0e-3), # ms-1
    ("rl_CICR", 1.0e-4), # ms-1
    ("rhoER_CICR", 0.3), # No unit
    ("d5_CICR", 120.), # nM
    ("d2_CICR", 3049.), # nM
    ("ker_CICR", 50.), # nM
    ("d1_CICR", 130.) # nM
	])
params_dict_ord_IP3 = coll.OrderedDict([
    ("kappad_IP3", 1500.), # nM
    ("kdelta_IP3", 100.), # nM
    ("r5p_IP3", 2.0e-4), # ms-1
    ("v3k_IP3", 0.001e-3), # ms-1 | To obtain CaMKIIact in no unit : CaMKIIact = phos_sum (Concentration)/[Concentration]
    ("kd_IP3", 1500.), # nM
    ("vdelta_IP3", 0.02), # nM.ms-1
    ("k3_IP3", 1000.), # nM
    ("kr_IP3", 1300.), # nM
    ("kp_IP3", 10000.), # nM
    ("vbeta_IP3", 0.8), # nM.ms-1
    ("kpi_IP3", 600.), # nM
    ("n3_IP3", 1.0), # No unit
	])
params_dict_ord_DGLandDAG = coll.OrderedDict([
    ("KDGL", 30000.), # nM
    ("kMAGL", 0.5e-3), # ms-1
    ("rDGL", 20000.0), # nM.ms-1
    ("kDAGK", 2.0e-3) # ms-1
	])
params_dict_ord_KandP_on_DAGLP = coll.OrderedDict([
    ("nK_DAGLP", 6.), # No unit
    ("rP_DAGLP", 0.380), # ms-1
    ("rK_DAGLP", 50.0e-3) # ms-1
	])
params_dict_ord_ECb = coll.OrderedDict([
    ("on_ECb", 1.), # No unit
    ("ECb_CB1R_on", 1.), # No unit
    ("kCB1R", 3000.0), # No unit
    ("alphaAEACB1", 0.1), # No unit
    ("P1_ECb", 1e-06), # ms
    ("P2_ECb", 1e-05), # No unit
    ("P3_ECb", 7.), # No unit
    ("P4_ECb", 2.0e3), # ms
    ("vATAEA", 2.0e-4), # ms-1
    ("LTDstart", 0.027), # No unit
    ("LTDstop", 0.047), # No unit
    ("LTDMax", 0.65), # No unit
    ("LTPstart", 0.086), # No unit
    ("LTPMax", 13.5425), # No unit
    ("KFAAH", 1000.), # nM
    ("vFAAH", 4.0), # nM.ms-1
	])
params_dict_ord_ECb_smooth = coll.OrderedDict([
    ("on_ECb_smooth", 0.), # No unit
    ("K_ECb_smooth", 0.0007), # No unit
    ("n_ECb_smooth", 2.), # No unit
    ("kw_ECb_smooth", 10.), # No unit
    ("tau_ECb_smooth", 0.0001), # No unit
    ("kadd_ECb_smooth", 0.), # No unit
    ("kn_ECb_smooth", 1.) # No unit
	])
params_dict_ord_post_CaMKII_plast = coll.OrderedDict([
    ("kpkaI1", 4.67e-3), # ms-1
    ("PP10", 200.), # nM
    ("KM", 400.), # nM
    ("K5", 100.), # nM
    ("k12", 6.0), # ms-1
    ("k11", 500.0e-6), # nM-1.ms-1
    ("kcan0I1", 0.05e-3), # ms-1
    ("km11", 1.0e-4), # ms-1
    ("CaMKT", 16600.), # nM
    ("k7", 6.0e-3), # ms-1
    ("k6", 6.0e-3), # ms-1
    ("ncanI1", 3.0), # No unit
    ("I10", 1000.), # nM
    ("CaMT", 70.52), # nM
    ("KdpkaI1", 159.), # nM
    ("kcanI1", 20.5e-3), # ms-1
    ("KdcanI1", 53.), # nM
    ("kpka0I1", 0.0025e-3), # ms-1
    ("Ka3", 320.), # nM
    ("Ka2", 25.), # nM
    ("Ka1", 100.), # nM
    ("Ka4", 400.), # nM
    ("npkaI1", 3.0) # No unit
	])

params_dict_ord_post_CaMKII_bis = coll.OrderedDict([
    ('conc_max_sub_unit', 200000.), # nM
    ('K9', 1e-1) # nM
    ])
params_dict_ord_DA = coll.OrderedDict([
    ("gamma1DA", 0.7e-3), # nM-1
    ("gamma2DA", 0.07e-3), # nM-1
    ("DA", 10.) # nM
	])
params_dict_ord_AMPA = coll.OrderedDict([
    ("on_AMPA", 1.), # No unit
    ("gAMPA", 5.1), # nS
    ("Epsilon_AMPA", 0.0), # ms-1
    ("Beta_AMPA", 190.0e-3), # ms-1
    ("Alpha_AMPA", 1.02e-6), # nM-1.ms-1
    ("Gamma_AMPA", 0.0) # ms-1
	])
params_dict_ord_NMDA = coll.OrderedDict([
    ("on_NMDA", 1.), # No unit
    ("gNMDA", 1.53), # nS
    ("p_ca_NMDA", 2.08324254657e-49), # microL.ms-1
    ("Mg_NMDA", 1.0), # mM because it's only use with the constant 3.57, which is in mM
    ("Alpha_NMDA", 0.072e-6), # nM-1.ms-1
    ("Beta_NMDA", 0.1) # ms-1
	])
params_dict_ord_mem = coll.OrderedDict([
    ("EL", -70.0), # mV
    ("gL", 10.0), # nS
    ("Cm", 0.1e3) # nF -> nA *s *V-1 -> nA *1e3ms *V-1
	])
params_dict_ord_CB1R = coll.OrderedDict([
    ("on_CB1R", 1.), # No unit
    ("Alpha_CB1R", 0.240194904182e-6), # nM-1.ms-1
    ("Beta_CB1R", 11.0718971839e-3), # ms-1
    ("Gamma_CB1R", 416.378884767e-3), # ms-1
    ("Epsilon_CB1R", 0.0477956844649e-3) # ms-1
	])
params_dict_ord_common = coll.OrderedDict([
    ("R", 8.3144621), # J.mol-1.K-1
    ("RT", 2553.78703401), # J.mol-1
    ("T", 307.15), # K
    ("F", 96.5), # kCoulomb.mol-1
    ("zS", 2.0), # No unit
    ("Ca_out", 5000.0e3) # nM
	])
params_dict_ord_integration = coll.OrderedDict([
    ("t_step", time_step), # ms
    ("t_start", t_start), # ms
    ("t_end", t_end), # ms
    ("ATOL", 1e-07), # No unit
    ("RTOL", 1e-07), # No unit
    ("MXSTEP", 1000.), # No unit
    ("HMAX", 50.0e3) # ms
	])
params_dict_ord_action = coll.OrderedDict([
    ("APdur", 30.), # ms
    ("action_as_VDCC", False), # No unit
    ("DPmax", 495.0), # pA
    ("APmax", 7020.0), # pA
    ("tausbAP", 1.) # ms
	])
params_dict_ord_Glu_release = coll.OrderedDict([
    ("steadyrise_on", 0.), # No unit
    ("Glumax", 2000.0e3), # nM
    ("BaseLevel", 0.0), # nM
    ("tauGlu", 5.) # ms
	])
params_dict_ord_stimulation = coll.OrderedDict([
    ("regular_on", 0.), # No unit
    ("tsdt", 15.), # ms
    ("tpost", 500.), # ms
    ("post_on", 0.), # No unit
    ("Delta_t_STDP", 20.), # ms
    ("num_stim", 20), # No unit | Integer
    ("pre_on", 1.), # No unit
    ("Freq", 1.0), # Hz
    ("tables_step", 5e-02) # ms
])

params_dict_ord_trad_unit = coll.OrderedDict([
    ("betaAEACB1", 1e3), # ms
    ("alphatwoAGCB1", 1e-3), # nM-1
    ("alpha_DAGLP_Ca_cyt", 1e-3) # nM-1
    ])

# --------------------- Inputs ampl --------------------------------- #

params_dict_ord_inputs_DA = coll.OrderedDict([
#                    ('input_ampl_DA', 100.),
                    ('input_ampl_DA', 0.), # nM
                    ('t_stim_on_DA', 0.), # ms
                    ('t_stim_off_DA', 0.3), # ms | Time of stim if stim is continuous, time of each peak of stim if it's non-continuous
                    ('interv_stim_ms_DA', 2.) # ms
                    ])

params_dict_ord_inputs_Ca = coll.OrderedDict([
#                    ('input_ampl_Ca', 100.),
                    ('input_ampl_Ca', 0.), # nM
                    ('t_stim_on_Ca', 0.), # ms
                    ('t_stim_off_Ca', 0.3), # ms | Time of stim if stim is continuous, time of each peak of stim if it's non-continuous
                    ('interv_stim_ms_Ca', 2.) # ms
                    ])

# ---- Blackwell params ---- #
params_dict_reactionRate = coll.OrderedDict([
            ('forwardRate_Ca_pump1a', 5e-05),
            ('reverseRate_Ca_pump1a', 0.009),
            ('forwardRate_Ca_pump1b', 0.003),
            ('reverseRate_Ca_pump1b', 0.0),
            ('forwardRate_Ca_pump2a', 1.1e-05),
            ('reverseRate_Ca_pump2a', 0.0112),
            ('forwardRate_Ca_pump2b', 0.0056),
            ('reverseRate_Ca_pump2b', 0.0),
            ('forwardRate_Ca_leak', 1e-08),
            ('reverseRate_Ca_leak', 0.0011),
            ('forwardRate_Ca_leak_1', 0.0011),
            ('reverseRate_Ca_leak_1', 0.0),
            ('forwardRate_Ca_buffer', 2.8e-05),
            ('reverseRate_Ca_buffer', 19.6),
            ('forwardRate_CamC_bind', 6e-06),
            ('reverseRate_CamC_bind', 0.0091),
            ('forwardRate_CamCa2C_bind', 0.0001),
            ('reverseRate_CamCa2C_bind', 1.0),
            ('forwardRate_CamN_bind', 0.0001),
            ('reverseRate_CamN_bind', 1.0),
            ('forwardRate_CamCa2N_bind', 6e-06),
            ('reverseRate_CamCa2N_bind', 0.0091),
            ('forwardRate_DaDbuf--Da+Dbuf', 0.01),
            ('reverseRate_DaDbuf--Da+Dbuf', 0.0),
            ('forwardRate_Da--DaOut', 0.001),
            ('reverseRate_Da--DaOut', 5.5e-08),
            ('forwardRate_DaOut+Dbuf--NULL', 1e-06),
            ('reverseRate_DaOut+Dbuf--NULL', 0.0),
            ('forwardRate_Da+D1R--DaD1R', 1.66667e-06),
            ('reverseRate_Da+D1R--DaD1R', 0.005),
            ('forwardRate_DaD1R+Gs--DaD1RGs', 3e-05),
            ('reverseRate_DaD1R+Gs--DaD1RGs', 0.0004),
            ('forwardRate_Gsabg+D1R--GsD1R', 1e-05),
            ('reverseRate_Gsabg+D1R--GsD1R', 0.0004),
            ('forwardRate_GsD1R+Da--DaD1RGs', 5e-06),
            ('reverseRate_GsD1R+Da--DaD1RGs', 0.005),
            ('forwardRate_DaD1RGs-DaD1R+GsaGTP+Gbg_reac', 0.00025),
            ('reverseRate_DaD1RGs-DaD1R+GsaGTP+Gbg_reac', 0.0),
            ('forwardRate_DaD1RGs_phosp1', 8e-07),
            ('reverseRate_DaD1RGs_phosp1', 8e-05),
            ('forwardRate_DaD1RGs_phosp2', 2e-05),
            ('reverseRate_DaD1RGs_phosp2', 0.0),
            ('forwardRate_dephosD1R', 2e-06),
            ('reverseRate_dephosD1R', 0.0),
            ('forwardRate_Glubuf--Glu+Gbuf', 0.01),
            ('reverseRate_Glubuf--Glu+Gbuf', 0.0),
            ('forwardRate_Glu--GluOut', 0.0006),
            ('reverseRate_Glu--GluOut', 3e-08),
            ('forwardRate_GluOut-buf_reac', 1e-06),
            ('reverseRate_GluOut-buf_reac', 0.0),
            ('forwardRate_Glu+MgluR--GluMgluRdesens', 6.25e-05),
            ('reverseRate_Glu+MgluR--GluMgluRdesens', 1e-06),
            ('forwardRate_Glu+MgluR--GluMgluR', 2e-06),
            ('reverseRate_Glu+MgluR--GluMgluR', 0.001),
            ('forwardRate_Gqabg+GluMgluR--GluMgluRGq', 1.5e-06),
            ('reverseRate_Gqabg+GluMgluR--GluMgluRGq', 0.00068),
            ('forwardRate_Gqabg+MgluR-_-MgluRGq', 7.5e-07),
            ('reverseRate_Gqabg+MgluR-_-MgluRGq', 0.0068),
            ('forwardRate_Glu+MgluRGq-_-GluMgluRGq', 1e-06),
            ('reverseRate_Glu+MgluRGq-_-GluMgluRGq', 0.001),
            ('forwardRate_GluMgluRGq--GqaGTP+GluMgluRGbg', 0.0005),
            ('reverseRate_GluMgluRGq--GqaGTP+GluMgluRGbg', 0.0),
            ('forwardRate_ACh+m1R--AChm1R', 9.5e-08),
            ('reverseRate_ACh+m1R--AChm1R', 0.0025),
            ('forwardRate_Gqabg+AChm1R--AChm1RGq', 2.4e-05),
            ('reverseRate_Gqabg+AChm1R--AChm1RGq', 0.00042),
            ('forwardRate_Gqabg+m1R-_-m1RGq', 5.76e-07),
            ('reverseRate_Gqabg+m1R-_-m1RGq', 0.00042),
            ('forwardRate_ACh+m1RGq-_-AChm1RGq', 3.96e-06),
            ('reverseRate_ACh+m1RGq-_-AChm1RGq', 0.0025),
            ('forwardRate_AChm1RGq--GqaGTP+AChm1RGbg', 0.0005),
            ('reverseRate_AChm1RGq--GqaGTP+AChm1RGbg', 0.0),
            ('forwardRate_ACh--null', 0.006),
            ('forwardRate_ACh+m4R--AChm4R', 3.38e-07),
            ('reverseRate_ACh+m4R--AChm4R', 0.0025),
            ('forwardRate_Giabg+AChm4R--AChm4RGi', 2.4e-05),
            ('reverseRate_Giabg+AChm4R--AChm4RGi', 0.00042),
            ('forwardRate_Giabg+m4R--Gim4R', 1.78e-06),
            ('reverseRate_Giabg+m4R--Gim4R', 0.00042),
            ('forwardRate_Gim4R+ACh--AChm4RGi', 4.56e-06),
            ('reverseRate_Gim4R+ACh--AChm4RGi', 0.0025),
            ('forwardRate_AChm4RGi--GiaGTP+AChm4R+Gbg', 0.001),
            ('reverseRate_AChm4RGi--GiaGTP+AChm4R+Gbg', 0.0),
            ('forwardRate_GsaGTP--GsaGDP_reac', 0.0005),
            ('reverseRate_GsaGTP--GsaGDP_reac', 0.0),
            ('forwardRate_GsaGDP+Gbg--Gs_reac', 0.002),
            ('reverseRate_GsaGDP+Gbg--Gs_reac', 0.0),
            ('forwardRate_GqaGTP--GqaGDP', 0.001),
            ('reverseRate_GqaGTP--GqaGDP', 0.0),
            ('forwardRate_GqaGDP+Gbg--Gqabg', 0.002),
            ('reverseRate_GqaGDP+Gbg--Gqabg', 0.0),
            ('forwardRate_GiaGTP--GiaGDP', 0.001),
            ('reverseRate_GiaGTP--GiaGDP', 0.0),
            ('forwardRate_GiaGDP--Giabg', 0.002),
            ('reverseRate_GiaGDP--Giabg', 0.0),
            ('forwardRate_GsaGTP+AC5--AC5Gsa_reac', 3.7e-07),
            ('reverseRate_GsaGTP+AC5--AC5Gsa_reac', 0.0001),
            ('forwardRate_AC5Gsa+ATP--AC5GsaATP_reac', 1.276e-07),
            ('reverseRate_AC5Gsa+ATP--AC5GsaATP_reac', 0.0002612),
            ('forwardRate_AC5GsaATP--AC5Gsa+cAMP_reac', 0.02846),
            ('reverseRate_AC5GsaATP--AC5Gsa+cAMP_reac', 2.592e-07),
            ('forwardRate_GiaGTP_AC5_reac', 1.25e-05),
            ('reverseRate_GiaGTP_AC5_reac', 0.002),
            ('forwardRate_GiaGTP_AC5GsaGTP_reac', 1.25e-05),
            ('reverseRate_GiaGTP_AC5GsaGTP_reac', 0.002),
            ('forwardRate_AC5GsaGia+ATP--AC5GsaGiaATP', 1.276e-08),
            ('reverseRate_AC5GsaGia+ATP--AC5GsaGiaATP', 2.612e-05),
            ('forwardRate_AC5GsaGiaATP--AC5GsaGia+cAMP', 0.0002746),
            ('reverseRate_AC5GsaGiaATP--AC5GsaGia+cAMP', 2.592e-07),
            ('forwardRate_AC5GsaGiaATP--AC5Gia+Gs+cAMP', 1e-05),
            ('reverseRate_AC5GsaGiaATP--AC5Gia+Gs+cAMP', 0.0),
            ('forwardRate_PDE1+CamCa4--PDE1CamCa4', 0.0001),
            ('reverseRate_PDE1+CamCa4--PDE1CamCa4', 0.001),
            ('forwardRate_PDE1CamCa4+cAMP--PDE1CamCa4cAMP', 4.6e-06),
            ('reverseRate_PDE1CamCa4+cAMP--PDE1CamCa4cAMP', 0.044),
            ('forwardRate_PDE1CamCa4cAMP--PDE1CamCa4+AMP', 0.011),
            ('reverseRate_PDE1CamCa4cAMP--PDE1CamCa4+AMP', 0.0),
            ('forwardRate_PDE10+cAMP--PDE10cAMP', 8.4e-05),
            ('reverseRate_PDE10+cAMP--PDE10cAMP', 0.0168),
            ('forwardRate_PDE10cAMP--PDE10+AMP', 0.0042),
            ('reverseRate_PDE10cAMP--PDE10+AMP', 0.0),
            ('forwardRate_pPDE10+cAMP--pPDE10cAMP', 0.000336),
            ('reverseRate_pPDE10+cAMP--pPDE10cAMP', 0.0672),
            ('forwardRate_pPDE10cAMP--pPDE10+AMP', 0.0168),
            ('reverseRate_pPDE10cAMP--pPDE10+AMP', 0.0),
            ('forwardRate_AMP--ATP', 0.001),
            ('reverseRate_AMP--ATP', 0.0),
            ('forwardRate_PDE10_phosp1', 2.5e-07),
            ('reverseRate_PDE10_phosp1', 2.4e-05),
            ('forwardRate_PDE10_phosp2', 6e-06),
            ('reverseRate_PDE10_phosp2', 0.0),
            ('forwardRate_PDE10cAMP_phosp1', 2.5e-07),
            ('reverseRate_PDE10cAMP_phosp1', 2.4e-05),
            ('forwardRate_PDE10cAMP_phosp2', 6e-06),
            ('reverseRate_PDE10cAMP_phosp2', 0.0),
            ('forwardRate_PDE10_d_phosp1', 8e-09),
            ('reverseRate_PDE10_d_phosp1', 3.2e-06),
            ('forwardRate_PDE10_d_phosp2', 8e-07),
            ('reverseRate_PDE10_d_phosp2', 0.0),
            ('forwardRate_PDE2+cAMP--PDE2cAMP', 2e-08),
            ('reverseRate_PDE2+cAMP--PDE2cAMP', 0.0005),
            ('forwardRate_PDE2cAMP+cAMP--PDE2cAMP2', 5.9e-06),
            ('reverseRate_PDE2cAMP+cAMP--PDE2cAMP2', 0.0005),
            ('forwardRate_PDE2cAMP2--PDE2cAMP+AMP', 0.0054),
            ('reverseRate_PDE2cAMP2--PDE2cAMP+AMP', 0.0),
            ('forwardRate_PDE4_enzyme1', 2.16e-05),
            ('reverseRate_PDE4_enzyme1', 0.06895),
            ('forwardRate_PDE4_actvcomp', 0.017233),
            ('reverseRate_PDE4_actvcomp', 0.0),
            ('forwardRate_pPDE4_enzyme1', 8.64e-05),
            ('reverseRate_pPDE4_enzyme1', 0.2758),
            ('forwardRate_pPDE4_actvcomp', 0.068932),
            ('reverseRate_pPDE4_actvcomp', 0.0),
            ('forwardRate_PDE4_phosp1', 2.5e-07),
            ('reverseRate_PDE4_phosp1', 8e-05),
            ('forwardRate_PDE4_phosp2', 2e-05),
            ('reverseRate_PDE4_phosp2', 0.0),
            ('forwardRate_PDE4cAMP_phosp1', 2.5e-07),
            ('reverseRate_PDE4cAMP_phosp1', 8e-05),
            ('forwardRate_PDE4cAMP_phosp2', 2e-05),
            ('reverseRate_PDE4cAMP_phosp2', 0.0),
            ('forwardRate_PDE4_d_phosp1', 8e-09),
            ('reverseRate_PDE4_d_phosp1', 3.2e-06),
            ('forwardRate_PDE4_d_phosp2', 8e-07),
            ('reverseRate_PDE4_d_phosp2', 0.0),
            ('forwardRate_Plc+Ca--PlcCa', 5e-07),
            ('reverseRate_Plc+Ca--PlcCa', 0.001),
            ('forwardRate_PlcCa+GqaGTP--PlcCaGqa', 3.5e-06),
            ('reverseRate_PlcCa+GqaGTP--PlcCaGqa', 0.0007),
            ('forwardRate_Plc+GqaGTP--PlcGqa', 7e-07),
            ('reverseRate_Plc+GqaGTP--PlcGqa', 0.0007),
            ('forwardRate_PlcGaGTP+Ca--PlcCaGqa', 2.5e-06),
            ('reverseRate_PlcGaGTP+Ca--PlcCaGqa', 0.001),
            ('forwardRate_PlcCa+Pip2--PlcCaPip2', 5e-08),
            ('reverseRate_PlcCa+Pip2--PlcCaPip2', 0.0018),
            ('forwardRate_PlcCaPip2--PlcCa+Dag+Ip3', 0.00045),
            ('reverseRate_PlcCaPip2--PlcCa+Dag+Ip3', 0.0),
            ('forwardRate_PlcCaGqa+Pip2--PlcCaGqaPip2', 1.665e-06),
            ('reverseRate_PlcCaGqa+Pip2--PlcCaGqaPip2', 0.06),
            ('forwardRate_PlcCaGqaPip2--PlcCaGqaDag+Ip3', 0.03),
            ('reverseRate_PlcCaGqaPip2--PlcCaGqaDag+Ip3', 0.0),
            ('forwardRate_PlcGqa--Plc+GqaGDP', 0.012),
            ('reverseRate_PlcGqa--Plc+GqaGDP', 0.0),
            ('forwardRate_PlcCaGqa--PlcCa+GaGDP', 0.012),
            ('reverseRate_PlcCaGqa--PlcCa+GaGDP', 0.0),
            ('forwardRate_Ca+Dlg--CaDgl', 0.000125),
            ('reverseRate_Ca+Dlg--CaDgl', 0.05),
            ('forwardRate_Dag+CaDlg--DagCaDgl', 5e-07),
            ('reverseRate_Dag+CaDlg--DagCaDgl', 0.001),
            ('forwardRate_DagCaDgl--CaDgl+2ag', 0.00025),
            ('reverseRate_DagCaDgl--CaDgl+2ag', 0.0),
            ('forwardRate_Ip3--Ip3degrad', 0.0001),
            ('reverseRate_Ip3--Ip3degrad', 0.0),
            ('forwardRate_Ip3degrad+PIkinase--Ip3degPIk', 5e-07),
            ('reverseRate_Ip3degrad+PIkinase--Ip3degPIk', 0.0004),
            ('forwardRate_Ip3degPIk--PIP2_+_PIkinase', 0.0001),
            ('reverseRate_Ip3degPIk--PIP2_+_PIkinase', 0.0),
            ('forwardRate_2ag--2agDegrad', 0.00025),
            ('reverseRate_2ag--2agDegrad', 0.0),
            ('forwardRate_DagKdag1', 7e-08),
            ('reverseRate_DagKdag1', 0.0008),
            ('forwardRate_DagKdag1_1', 0.0002),
            ('reverseRate_DagKdag1_1', 0.0),
            ('forwardRate_Pkc+Ca--PkcCa', 2e-05),
            ('reverseRate_Pkc+Ca--PkcCa', 0.05),
            ('forwardRate_PkcCa+Dag--PkcCaDag', 1.5e-08),
            ('reverseRate_PkcCa+Dag--PkcCaDag', 0.00015),
            ('forwardRate_PKA_bind', 2.61e-07),
            ('reverseRate_PKA_bind', 6e-05),
            ('forwardRate_PKAcAMP2_bind', 3.5e-07),
            ('reverseRate_PKAcAMP2_bind', 0.0006),
            ('forwardRate_PKAcAMP4_diss', 0.00024),
            ('reverseRate_PKAcAMP4_diss', 2.55e-05),
            ('forwardRate_Cam+PP2B--PP2BCam_reac', 4.6e-06),
            ('reverseRate_Cam+PP2B--PP2BCam_reac', 1.2e-06),
            ('forwardRate_CamCa2C+PP2B--PP2BCamCa2C_reac', 4.6e-05),
            ('reverseRate_CamCa2C+PP2B--PP2BCamCa2C_reac', 1.2e-06),
            ('forwardRate_CamCa2N+PP2B--PP2BCamCa2N_reac', 4.6e-05),
            ('reverseRate_CamCa2N+PP2B--PP2BCamCa2N_reac', 1.2e-06),
            ('forwardRate_PP2BCam+Ca2--PP2BCamCa2C_reac', 6e-06),
            ('reverseRate_PP2BCam+Ca2--PP2BCamCa2C_reac', 0.00091),
            ('forwardRate_PP2BCam+Ca2--PP2BCamCa2N_reac', 0.0001),
            ('reverseRate_PP2BCam+Ca2--PP2BCamCa2N_reac', 0.1),
            ('forwardRate_CamCa4+PP2B--PP2BCamCa4_reac', 4.6e-05),
            ('reverseRate_CamCa4+PP2B--PP2BCamCa4_reac', 1.2e-06),
            ('forwardRate_PP2BCamCa2C+Ca2--PP2BCamCa4_reac', 0.0001),
            ('reverseRate_PP2BCamCa2C+Ca2--PP2BCamCa4_reac', 1.0),
            ('forwardRate_PP2BCamCa2N+Ca2--PP2BCamCa4_reac', 6e-06),
            ('reverseRate_PP2BCamCa2N+Ca2--PP2BCamCa4_reac', 0.0091),
            ('forwardRate_PKAc+PP2A--PKAcPP2AB56d_reac', 2.5e-06),
            ('reverseRate_PKAc+PP2A--PKAcPP2AB56d_reac', 0.00037),
            ('forwardRate_PKAcPP2AB56d--PKAc+pPP2A_reac', 6e-05),
            ('reverseRate_PKAcPP2AB56d--PKAc+pPP2A_reac', 0.0),
            ('forwardRate_pPP2A--PP2A_reac', 2e-05),
            ('reverseRate_pPP2A--PP2A_reac', 0.0),
            ('forwardRate_PP2ABPR72+Ca--PP2Acal_reac', 2e-08),
            ('reverseRate_PP2ABPR72+Ca--PP2Acal_reac', 0.0001),
            ('forwardRate_D32+PKAc--D32PKAc_reac', 2.7e-06),
            ('reverseRate_D32+PKAc--D32PKAc_reac', 0.008),
            ('forwardRate_D32PKAc--PKAc+D32p34_reac', 0.002),
            ('reverseRate_D32PKAc--PKAc+D32p34_reac', 0.0),
            ('forwardRate_D32p34+PP1--D32p34pp1_reac', 0.0004),
            ('reverseRate_D32p34+PP1--D32p34pp1_reac', 0.00058),
            ('forwardRate_D32p34+PP2BCamCa4--D32p34PP2BCamCa4_reac', 2.5e-06),
            ('reverseRate_D32p34+PP2BCamCa4--D32p34PP2BCamCa4_reac', 0.004),
            ('forwardRate_D32p34PP2BCamCa4--PP2BCamCa4+D32_reac', 0.001),
            ('reverseRate_D32p34PP2BCamCa4--PP2BCamCa4+D32_reac', 0.0),
            ('forwardRate_D32p34PP1+PP2BCamCa4--D32p34PP1PP2BCamCa4_reac', 2.5e-06),
            ('reverseRate_D32p34PP1+PP2BCamCa4--D32p34PP1PP2BCamCa4_reac', 0.004),
            ('forwardRate_D32p34PP1PP2BCamCa4--PP1+PP2BCamCa4+D32_reac', 0.001),
            ('reverseRate_D32p34PP1PP2BCamCa4--PP1+PP2BCamCa4+D32_reac', 0.0),
            ('forwardRate_D32p34+PP2ABPR72--D32p34PP2ABPR72_reac', 1.52e-06),
            ('reverseRate_D32p34+PP2ABPR72--D32p34PP2ABPR72_reac', 0.056),
            ('forwardRate_D32p34PP2ABPR72--PP2ABPR72+D32_reac', 0.014),
            ('reverseRate_D32p34PP2ABPR72--PP2ABPR72+D32_reac', 0.0),
            ('forwardRate_D32p34+PP2AB56d--D32p34PP2AB56d_reac', 1.52e-06),
            ('reverseRate_D32p34+PP2AB56d--D32p34PP2AB56d_reac', 0.056),
            ('forwardRate_D32p34PP2AB56d--PP2AB56d+D32_reac', 0.014),
            ('reverseRate_D32p34PP2AB56d--PP2AB56d+D32_reac', 0.0),
            ('forwardRate_D32p34pp1+PP2ABPR72--D32p34PP1PP2ABPR72_reac', 1.5e-07),
            ('reverseRate_D32p34pp1+PP2ABPR72--D32p34PP1PP2ABPR72_reac', 0.0061),
            ('forwardRate_D32p34PP1PP2ABPR72--PP1+PP2ABPR72+D32_reac', 0.0014),
            ('reverseRate_D32p34PP1PP2ABPR72--PP1+PP2ABPR72+D32_reac', 0.0),
            ('forwardRate_D32p34pp1+PP2AB56d--D32p34PP1PP2AB56d_reac', 1.5e-07),
            ('reverseRate_D32p34pp1+PP2AB56d--D32p34PP1PP2AB56d_reac', 0.0061),
            ('forwardRate_D32p34PP1PP2AB56d--PP1+PP2AB56d+D32_reac', 0.0014),
            ('reverseRate_D32p34PP1PP2AB56d--PP1+PP2AB56d+D32_reac', 0.0),
            ('forwardRate_Cdk5+D32--Cdk5D32_reac', 4.5e-06),
            ('reverseRate_Cdk5+D32--Cdk5D32_reac', 0.04),
            ('forwardRate_Cdk5D32--D32p75+Cdk5_reac', 0.01),
            ('reverseRate_Cdk5D32--D32p75+Cdk5_reac', 0.0),
            ('forwardRate_D32p75+PKAc--D32p75PKAc_reac', 3.7e-07),
            ('reverseRate_D32p75+PKAc--D32p75PKAc_reac', 0.001),
            ('forwardRate_D32p75+pPP2A--D32p75pPP2A_reac', 1.5e-05),
            ('reverseRate_D32p75+pPP2A--D32p75pPP2A_reac', 0.168),
            ('forwardRate_D32p75pPP2A--D32+pPP2A_reac', 0.042),
            ('reverseRate_D32p75pPP2A--D32+pPP2A_reac', 0.0),
            ('forwardRate_D32p75+PP2ABPR72--D32p75PP2ABPR72_reac', 2.3e-06),
            ('reverseRate_D32p75+PP2ABPR72--D32p75PP2ABPR72_reac', 0.084),
            ('forwardRate_D32p75PP2ABPR72--D32+PP2ABPR72_reac', 0.021),
            ('reverseRate_D32p75PP2ABPR72--D32+PP2ABPR72_reac', 0.0),
            ('forwardRate_D32p75+PP2AB56d--D32p75PP2AB56d_reac', 2.3e-06),
            ('reverseRate_D32p75+PP2AB56d--D32p75PP2AB56d_reac', 0.084),
            ('forwardRate_D32p75PP2AB56d--D32+PP2AB56d_reac', 0.021),
            ('reverseRate_D32p75PP2AB56d--D32+PP2AB56d_reac', 0.0),
            ('forwardRate_D32p75+PP2Acal--D32p75PP2Acal_reac', 1.5e-05),
            ('reverseRate_D32p75+PP2Acal--D32p75PP2Acal_reac', 0.168),
            ('forwardRate_D32p75PP2Acal-D32+PP2Acal_reac', 0.042),
            ('reverseRate_D32p75PP2Acal-D32+PP2Acal_reac', 0.0),
            ('forwardRate_CK_bind', 1e-05),
            ('reverseRate_CK_bind', 0.003),
            ('forwardRate_CKCam_bind', 1.3e-17),
            ('reverseRate_CKCam_bind', 0.0),
            ('forwardRate_CKCam_bind_1', 2e-12),
            ('reverseRate_CKCam_bind_1', 0.0),
            ('forwardRate_CKCam_bind_2', 2.71e-17),
            ('reverseRate_CKCam_bind_2', 0.0),
            ('forwardRate_CKp_auto', 8e-07),
            ('reverseRate_CKp_auto', 1e-05),
            ('forwardRate_CKp_bind', 1e-08),
            ('reverseRate_CKp_bind', 8.5e-05),
            ('forwardRate_CKpPP1_reac', 2.5e-05),
            ('reverseRate_CKpPP1_reac', 0.0),
            ('forwardRate_CKpCamCa4PP1_bind', 0.0),
            ('reverseRate_CKpCamCa4PP1_bind', 0.0),
            ('forwardRate_CKpCamCa4PP1_reac', 0.0),
            ('reverseRate_CKpCamCa4PP1_reac', 0.0),
            ('forwardRate_Dgl_CKCamCa4_bind', 3.6e-07),
            ('reverseRate_Dgl_CKCamCa4_bind', 0.0001),
            ('forwardRate_Dgl_CKCam_diss', 2.5e-05),
            ('reverseRate_Dgl_CKCam_diss', 0.0),
            ('forwardRate_Dgl_CKpCamCa4_bind', 9e-07),
            ('reverseRate_Dgl_CKpCamCa4_bind', 0.00025),
            ('forwardRate_Dgl_CKpCamCa4_diss', 6.25e-05),
            ('reverseRate_Dgl_CKpCamCa4_diss', 0.0),
            ('forwardRate_CaDgl_CKpCamCa4_bind', 9e-07),
            ('reverseRate_CaDgl_CKpCamCa4_bind', 0.00025),
            ('forwardRate_CaDgl_CKpCamCa4_diss', 6.25e-05),
            ('reverseRate_CaDgl_CKpCamCa4_diss', 0.0),
            ('forwardRate_Dgl_CKp_bind', 3.6e-07),
            ('reverseRate_Dgl_CKp_bind', 0.0001),
            ('forwardRate_Dgl_CKp_diss', 2.5e-05),
            ('reverseRate_Dgl_CKp_diss', 0.0),
            ('forwardRate_pDgl_dephos', 0.0001),
            ('reverseRate_pDgl_dephos', 0.0),
            ('forwardRate_Ca+pDlg--CapDgl', 0.000125),
            ('reverseRate_Ca+pDlg--CapDgl', 0.05),
            ('forwardRate_Dag+CapDlg--DagCapDgl', 1e-07),
            ('reverseRate_Dag+CapDlg--DagCapDgl', 0.0002),
            ('forwardRate_DagCapDgl--CapDgl+2ag', 5e-05),
            ('reverseRate_DagCapDgl--CapDgl+2ag', 0.0),
            ('forwardRate_AKAR3+PKAc_bind', 1.25e-05),
            ('reverseRate_AKAR3+PKAc_bind', 0.00025),
            ('forwardRate_PKAcAKAR3_phos', 0.000375),
            ('reverseRate_PKAcAKAR3_phos', 0.0),
            ('forwardRate_pAKAR3+PP1_bind', 2.5e-07),
            ('reverseRate_pAKAR3+PP1_bind', 0.0009),
            ('forwardRate_PP1pAKAR3_dephos', 0.000625),
            ('reverseRate_PP1pAKAR3_dephos', 0.0),
            ('forwardRate_GsaGTP_AC1_reac', 3.85e-05),
            ('reverseRate_GsaGTP_AC1_reac', 0.01),
            ('forwardRate_AC1GsaGTP_CamCa4_Reac', 6e-06),
            ('reverseRate_AC1GsaGTP_CamCa4_Reac', 0.0009),
            ('forwardRate_AC1GsaGTPCamCa4_ATP_Reac', 1e-05),
            ('reverseRate_AC1GsaGTPCamCa4_ATP_Reac', 2.273),
            ('forwardRate_AC1GsaGTPCamCa4ATP_diss', 0.05684),
            ('reverseRate_AC1GsaGTPCamCa4ATP_diss', 0.0),
            ('forwardRate_AC1CamCa4_Reac', 6e-06),
            ('reverseRate_AC1CamCa4_Reac', 0.0009),
            ('forwardRate_AC1CamCa4ATP_bind', 1e-05),
            ('reverseRate_AC1CamCa4ATP_bind', 2.273),
            ('forwardRate_AC1CamCa4ATP_disso', 0.005684),
            ('reverseRate_AC1CamCa4ATP_disso', 0.0)])


# -------------------- Params for arrays ----------------------------- #
# --------------- LTPwin_tab ------------------------- #
n_x_LTPwin = 100
xst_LTPwin = params_dict_ord_ECb['LTPstart'] -6*params_dict_ord_ECb_smooth['tau_ECb_smooth']
xstep_LTPwin = (2*6*params_dict_ord_ECb_smooth['tau_ECb_smooth'])/n_x_LTPwin

LTPwin_tab = np.zeros(n_x_LTPwin)

for i in xrange(0, n_x_LTPwin):

    V = xst_LTPwin +xstep_LTPwin*(i)
    LTPwin_tab[i] = 1/(1+np.exp(-(V -params_dict_ord_ECb['LTPstart'])/params_dict_ord_ECb_smooth['tau_ECb_smooth']))

# --------------- caL13_tab ------------------------- #
xst_caL13 = -100. # mV
xfin_caL13 = 100. # mV
n_x_caL13 = 401

xstep_caL13 = (xfin_caL13 -xst_caL13)/n_x_caL13
mtau_caL13_tab = np.zeros(n_x_caL13)
hinf_caL13_tab = np.zeros(n_x_caL13)
minf_caL13_tab = np.zeros(n_x_caL13)

for i in xrange(0, n_x_caL13):
    V = xst_caL13 +xstep_caL13*(i)
    minf_caL13_tab[i] = 1./( 1 + np.exp((V -params_dict_ord_caL13['mvhalf_caL13'] -params_dict_ord_caL13['mshift_caL13'])/params_dict_ord_caL13['mslope_caL13']) ) # No unit
    hinf_caL13_tab[i] = 1./( 1 + np.exp((V -params_dict_ord_caL13['hvhalf_caL13'] -params_dict_ord_caL13['hshift_caL13'])/params_dict_ord_caL13['hslope_caL13']) ) # No unit

    malpha = params_dict_ord_caL13['c_caL13'] *(V -params_dict_ord_caL13['vm_caL13'])/(np.exp((V -params_dict_ord_caL13['vm_caL13'])/params_dict_ord_caL13['k_caL13']) -1) # time-1
    mbeta = params_dict_ord_caL13['cpr_caL13'] * np.exp(V/params_dict_ord_caL13['kpr_caL13']) # time-1
    mtau_caL13_tab[i] = 1./(malpha + mbeta) # time


# --------------- NMDA_tab ------------------------- #
xst_NMDA = -100. # mV
xfin_NMDA = 100. # mV
n_x_NMDA = 401
xstep_NMDA = (xfin_NMDA -xst_NMDA)/n_x_NMDA
B_NMDA_tab = np.zeros(n_x_NMDA)

for i in xrange(0, n_x_NMDA):
    V = xst_NMDA +xstep_NMDA*(i)
    B_NMDA_tab[i] = 1.0/(1 +params_dict_ord_NMDA['Mg_NMDA']/3.57*np.exp(-0.062*V)) # No unit


# ---------------- STIMULATIONS -------------------------- #
# --------- Glu ---------- #
nb_Glu_stim = params_dict_ord_stimulation['num_stim']
freq_glu = params_dict_ord_stimulation['Freq'] # Hz

start_stim = 0.0 # ms
end_stim = start_stim +(nb_Glu_stim) *1e3/freq_glu # ms

Glumax_tab = np.ones(nb_Glu_stim) *params_dict_ord_Glu_release['Glumax']
tauglu_tab = np.ones(nb_Glu_stim) *params_dict_ord_Glu_release['tauGlu']

t_pre_glu_tab = np.arange(start_stim, end_stim, 1e3/freq_glu)
t_start_glu_tab = t_pre_glu_tab +params_dict_ord_stimulation['tpost'] -params_dict_ord_stimulation['tsdt'] -params_dict_ord_stimulation['Delta_t_STDP']

# --------- Iaction ---------- #
nb_Iact_stim = nb_Glu_stim
freq_Iact = freq_glu # Hz

start_stim = 0.0 # s
end_stim = start_stim +(nb_Iact_stim) *1e3/freq_Iact # ms

DPmax_tab = np.ones(nb_Iact_stim) *params_dict_ord_action['DPmax']
APmax_tab = np.ones(nb_Iact_stim) *params_dict_ord_action['APmax']
tausbAP_tab = np.ones(nb_Iact_stim) *params_dict_ord_action['tausbAP']

t_start_Iact_tab = t_pre_glu_tab +params_dict_ord_stimulation['tpost'] -2.*params_dict_ord_stimulation['tsdt']

# ---------------------- #

# ---------------------- #
params_dict_ord_arrays = coll.OrderedDict([
                    ('xst_LTPwin', xst_LTPwin),
                    ('xstep_LTPwin', xstep_LTPwin),
                    ('xst_caL13', xst_caL13),
                    ('xstep_caL13', xstep_caL13),
                    ('xst_NMDA', xst_NMDA),
                    ('xstep_NMDA', xstep_NMDA),
                    ('nb_Glu_stim', nb_Glu_stim),
                    ('nb_Iact_stim', nb_Iact_stim),
                    ('n_x_LTPwin', n_x_LTPwin),
                    ('n_x_caL13', n_x_caL13),
                    ('n_x_NMDA', n_x_NMDA)
                    ])

params_dict = coll.OrderedDict([])
params_dict.update(params_dict_ord_caL13)
params_dict.update(params_dict_ord_TRPV1)
params_dict.update(params_dict_ord_I_to_Ca_flux)
params_dict.update(params_dict_ord_CaBuff)
params_dict.update(params_dict_ord_CICR)
params_dict.update(params_dict_ord_IP3)
params_dict.update(params_dict_ord_DGLandDAG)
params_dict.update(params_dict_ord_KandP_on_DAGLP)
params_dict.update(params_dict_ord_ECb)
params_dict.update(params_dict_ord_ECb_smooth)
params_dict.update(params_dict_ord_post_CaMKII_plast)
params_dict.update(params_dict_ord_post_CaMKII_bis)
params_dict.update(params_dict_ord_DA)
params_dict.update(params_dict_ord_AMPA)
params_dict.update(params_dict_ord_NMDA)
params_dict.update(params_dict_ord_mem)
params_dict.update(params_dict_ord_CB1R)
params_dict.update(params_dict_ord_common)
params_dict.update(params_dict_ord_integration)
params_dict.update(params_dict_ord_action)
params_dict.update(params_dict_ord_Glu_release)
params_dict.update(params_dict_ord_stimulation)
params_dict.update(params_dict_ord_trad_unit)
params_dict.update(params_dict_ord_inputs_DA)
params_dict.update(params_dict_ord_inputs_Ca)

params_dict.update(params_dict_reactionRate)
#
#
params_dict.update(params_dict_ord_arrays)

# -------- To steady-states ----- #
params_dict_no_stim = deepcopy(params_dict)
params_dict_no_stim['nb_Glu_stim'] = 0
params_dict_no_stim['nb_Iact_stim'] = 0

params_array = params_dict.values() # List of parameters
params_array_no_stim = params_dict_no_stim.values()

## ----------- Concat params array with LTPwin and caL13 tab----- #
params_array = np.concatenate([params_array, LTPwin_tab, minf_caL13_tab, hinf_caL13_tab, mtau_caL13_tab, B_NMDA_tab])
params_array_no_stim = np.concatenate([params_array_no_stim, LTPwin_tab, minf_caL13_tab, hinf_caL13_tab, mtau_caL13_tab, B_NMDA_tab])
#
## ----------- Concat params array with Glu stim tab----- #
params_array = np.concatenate([params_array, Glumax_tab, tauglu_tab, t_start_glu_tab])
params_array_no_stim = np.concatenate([params_array_no_stim, Glumax_tab, tauglu_tab, t_start_glu_tab])

## ----------- Concat params array with Glu stim tab----- #
params_array = np.concatenate([params_array, DPmax_tab, APmax_tab, tausbAP_tab, t_start_Iact_tab])
params_array_no_stim = np.concatenate([params_array_no_stim, DPmax_tab, APmax_tab, tausbAP_tab, t_start_Iact_tab])

nb_eq = len(init_array) # Number of equations

# Not be executed if it's imported #
#if __name__ == '__main__':

    # ----------- Meta - Continue -------------------------- #
#nb_rep = 100
#for k in xrange(nb_rep): # Instead of having a big array which take all memory
#
#    # ----------- Continue -------------------------- #
#    for j in xrange(len(y_serie.columns)):
#        init_dict[y_serie.columns[j]] = y_serie.iloc[-1,j]
#    init_array = init_dict.values()

    #params_dict['I_N_inj'] = 8000.
    #params_dict['G_Kir'] = G_Kir_eq

    #    params_array = params_dict.values()
    #
    #    #    t_end = 10.
    #    #    time_step = 0.001
    #    # ----------------------------------------------- #


# ----- To steady-states ------ #
t_end_to_steady_states = 50000.

res = MSc.main(init_array, params_array_no_stim, nb_eq, t_start, t_end_to_steady_states, time_step, h_step = 1e-8, rel_err = 1e-8, abs_err = 1e-8)
y_serie = pd.DataFrame(res[:,1:], columns = init_keys, index = res[:,0])


# ----------- Continue -------------------------- #
for j in xrange(len(y_serie.columns)):
    init_dict[y_serie.columns[j]] = y_serie.iloc[-1,j]

#init_dict.pop('fpost')
#init_dict.pop('phos_sum')
#init_dict.pop('yCB1R')

init_array = init_dict.values()



res = MSc.main(init_array, params_array, nb_eq, t_start, t_end, time_step, h_step = 1e-8, rel_err = 1e-8, abs_err = 1e-8)
y_serie = pd.DataFrame(res[:,1:], columns = init_keys, index = res[:,0])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ----------- Post-treatment ----------- #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# ------------------- #
Ileak = np.zeros(len(y_serie))
for i in xrange(len(y_serie)):
    Ileak[i] = params_dict['gL'] * (y_serie['V'].values[i] -params_dict['EL'])

# ------------------- #
I_caL13 = np.zeros(len(y_serie))
for i in xrange(len(y_serie)):
    x = params_dict['zS']*params_dict['F']*y_serie['V'].values[i]*1e-3/params_dict['RT'] # 1e-3 No unit
    eco = params_dict['Ca_out']*efun(x) # microM*1e-3
    eci = y_serie['Ca_cyt'].values[i]*efun(-x) # microM*1e-3
    ghk = params_dict['zS']*params_dict['F']*(eci -eco) # kCoul.microM*1e-3.mol-1
    I_caL13[i] = ghk *params_dict['pcaLbar'] *y_serie['m_caL13'].values[i] *y_serie['m_caL13'].values[i] *y_serie['h_caL13'].values[i]


# ------------------- #
I_TRPV1 = np.zeros(len(y_serie))
for i in xrange(len(y_serie)):
    Q = y_serie['AEA'].values[i]/params_dict['KD_TRPV1'] # No unit
    x = params_dict['z_TRPV1']*params_dict['F']*y_serie['V'].values[i]/params_dict['RT']

    if x <= 85.0:
        J = params_dict['J0_TRPV1']*np.exp(x)
        G_TRPV1 = 1/(1+(1+J+params_dict['K_TRPV1']+Q+J*params_dict['K_TRPV1']+J*Q+params_dict['K_TRPV1']*Q+J*params_dict['K_TRPV1']*Q)/(params_dict['L_TRPV1']*(1+J*params_dict['D_TRPV1']+params_dict['K_TRPV1']*params_dict['C_TRPV1']+Q*params_dict['P_TRPV1']+J*params_dict['K_TRPV1']*params_dict['C_TRPV1']*params_dict['D_TRPV1']+J*Q*params_dict['D_TRPV1']*params_dict['P_TRPV1']+params_dict['K_TRPV1']*Q*params_dict['C_TRPV1']*params_dict['P_TRPV1']+J*params_dict['K_TRPV1']*Q*params_dict['D_TRPV1']*params_dict['C_TRPV1']*params_dict['P_TRPV1'])))
    else:
        G_TRPV1 = 1/(1+(1+params_dict['K_TRPV1']+Q+params_dict['K_TRPV1']*Q)/(params_dict['L_TRPV1']*(params_dict['D_TRPV1']+params_dict['K_TRPV1']*params_dict['C_TRPV1']*params_dict['D_TRPV1']+Q*params_dict['D_TRPV1']*params_dict['P_TRPV1']+params_dict['K_TRPV1']*Q*params_dict['D_TRPV1']*params_dict['C_TRPV1']*params_dict['P_TRPV1'])))

    I_TRPV1[i] = params_dict['gTRPV1'] *y_serie['V'].values[i] *G_TRPV1

# ------------------- #
I_AMPA = np.zeros(len(y_serie))
for i in xrange(len(y_serie)):
    I_AMPA[i] = params_dict['gAMPA'] *y_serie['o_AMPA'].values[i]*y_serie['V'].values[i]

# ------------------- #
I_NMDA = np.zeros(len(y_serie))
G_NMDA = np.zeros(len(y_serie))
for i in xrange(len(y_serie)):

    G_NMDA[i] = y_serie['o_NMDA'].values[i] *lin_interpTable_brds(B_NMDA_tab, int(n_x_NMDA), (y_serie['V'].values[i] -xst_NMDA)/xstep_NMDA)
    I_NMDA[i] = params_dict['gNMDA'] *y_serie['V'].values[i] *G_NMDA[i]
# ------------------- #
I_action = np.zeros(len(y_serie))
for i in xrange(len(y_serie)):

    Iact_stim = 0
    for jj in xrange(nb_Iact_stim):
        AP_term_temp = 0.
        if  y_serie.index[i] -t_start_Iact_tab[jj] > 0.:
            if y_serie.index[i] <= t_start_Iact_tab[jj] +params_dict['APdur']:
                if y_serie.index[i] -t_start_Iact_tab[jj] -params_dict['tsdt'] > 0.:
                    AP_term_temp = -APmax_tab[jj] *np.exp(-(y_serie.index[i] -t_start_Iact_tab[jj] -params_dict['tsdt'])/tausbAP_tab[jj])

                Iact_stim += -DPmax_tab[jj] +AP_term_temp

    I_action[i] += Iact_stim
# ------------------- #
Glu_serie = np.zeros(len(y_serie))
for i in xrange(len(y_serie)):
    Glu_stim = 0.
    for jj in xrange(nb_Glu_stim):

        if y_serie.index[i] -t_start_glu_tab[jj] >0.:
            if y_serie.index[i] <= t_start_glu_tab[jj] +tauglu_tab[jj]*30.:

                Glu_stim += Glumax_tab[jj] *np.exp(-(y_serie.index[i] -t_start_glu_tab[jj])/tauglu_tab[jj])
    Glu_serie[i] += Glu_stim
# ------------------- #

#Itotal = -Ileak -I_caL13 -I_TRPV1 -I_AMPA -I_NMDA -I_action
#plt.plot(Itotal)

phos_sum = y_serie['B1'] + 2*(y_serie['B2'] + y_serie['B3'] + y_serie['B4']) + 3*(y_serie['B5'] + y_serie['B6'] + y_serie['B7'] + y_serie['B8']) + 4*(y_serie['B9'] + y_serie['B10'] + y_serie['B11']) + 5*y_serie['B12'] + 6*y_serie['B13']

y_serie['phos_sum'] = phos_sum
y_serie['Glu'] = Glu_serie


#print y_serie

phos_sum_max = 164.6*1e3 # C'est quoi le calcul ???
fpost = 1 +3.5 *phos_sum/phos_sum_max
y_serie['fpost'] = fpost


if params_dict['on_ECb']:
    if params_dict['ECb_CB1R_on']:
        ctrl1 = params_dict['kCB1R'] *y_serie['o_CB1R'] +params_dict['gamma1DA'] *params_dict['DA'] # No unit
        ctrl2 = params_dict['kCB1R'] *y_serie['o_CB1R'] +params_dict['gamma2DA'] *params_dict['DA'] # No unit

    else:
        ctrl1 = params_dict['alphatwoAGCB1'] *y_serie['twoAG'] +params_dict['betaAEACB1']*params_dict['alphaAEACB1']*y_serie['AEA'] +params_dict['gamma1DA'] *params_dict['DA'] # No unit
        ctrl2 = params_dict['alphatwoAGCB1'] *y_serie['twoAG'] +params_dict['betaAEACB1']*params_dict['alphaAEACB1']*y_serie['AEA'] +params_dict['gamma2DA'] *params_dict['DA'] # No unit

y_serie['yCB1R'] = ctrl1

#
#
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
## ----------- PLOT ----------- #
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
#
#
#import pylab as plt
#plt.figure()
#plt.plot(y_serie['V'])

print("--- %s seconds ---" % (time.time() - start_time))