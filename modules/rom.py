import numpy as np
import pandas as pd
import pickle
import json
import glob
import pickle
from collections import defaultdict
from .function import pymatgen_comp
from pymatgen import core as mg


bcc_file,fcc_file,hcp_file,mass_file,ppk_file,vol_file = sorted(glob.glob('dataset/*.pkl'))
with open(bcc_file,'rb') as fid:
    bcc_dict = pickle.load(fid)
with open(fcc_file,'rb') as fid:
    fcc_dict = pickle.load(fid)
with open(hcp_file,'rb') as fid:
    hcp_dict = pickle.load(fid)
with open(ppk_file,'rb') as fid:
    price_per_kg_dict = pickle.load(fid)
with open(mass_file,'rb') as fid:
    mass_dict = pickle.load(fid)
with open(vol_file,'rb') as fid:
    vol_dict = pickle.load(fid)

excelFile = 'dataset/FundemantalDescriptors_PureElements.xlsx'
metaDF = pd.read_excel(excelFile)
meta = metaDF.to_json(orient="split")
metaIndex = json.loads(meta)['columns']
metaParsed = json.loads(meta)['data']
meta_dict=defaultdict(dict)

pt_file= 'dataset/periodic_table.csv'
vec_file = 'dataset/VEC.csv'
pt_df = pd.read_csv(pt_file)
vec_df = pd.read_csv(vec_file,header=None,names = ['Element','VEC'])
grp_dict = {x:y for x,y in zip(pt_df['Symbol'].values,pt_df['Group'].values)}
vec_dict = {x:y for x,y in zip(vec_df['Element'].values,vec_df['VEC'].values)}

def structure_choose(metaIndex_dict,meta_dict,material,n_index):
    metaIndex_update1={}
    for j,k in metaIndex_dict.items():
        comb_final=0
        data={}
        comb=0
        sum_comb=0
        if k>2:
            for o in material['compositionDictionary'].keys():
                if o not in meta_dict:
                    data[j] = None
                    break

                structure = ''
                if n_index != None:
                    s = n_index
                    structure = material['structure'][s]
                else:
                    try:
                        for a in meta_dict[o].keys():
                            int(a)
                            structure = meta_dict[o][a][1]
                    except:
                        structure='BCC'
                    #print('strucutre',material['compositionDictionary'],o,structure)

                try:
                    data[j]=meta_dict[o][structure][k]
                    float(data[j])
                    ##print('BCC')
                except:
                    ##print('No vaule for '+j+' of '+o+' for the phase in records, try other structures')
                    if structure=='BCC':
                        try:
                            data[j]=meta_dict[o]['FCC'][k]
                            float(data[j])
                            ##print('FCC')
                        except:
                            try:
                                data[j]=meta_dict[o]['HCP'][k]
                                float(data[j])
                                ##print('HCP')
                            except:
                                ##print('BREAk')
                                data[j] = None
                                break
    
                    elif structure=='FCC':
                        try:
                            data[j]=meta_dict[o]['HCP'][k]
                            float(data[j])
                        except:
                            try:
                                data[j]=meta_dict[o]['BCC'][k]
                                float(data[j])
                            except:
                                data[j] = None
                                break
                    elif structure=='HCP':
                        try:
                            data[j]=meta_dict[o]['FCC'][k]
                            float(data[j])
                        except:
                            try:
                                data[j]=meta_dict[o]['BCC'][k]
                                float(data[j])
                            except:
                                data[j] = None
                                break
                    elif structure=='Others':
                        try:
                            data[j]=meta_dict[o]['BCC'][k]
                            float(data[j])
                            #print('others','BCC')
                        except:
                            try:
                                data[j]=meta_dict[o]['FCC'][k]
                                float(data[j])
                                #print('others','FCC')
                            except:
                                try:
                                    data[j]=meta_dict[o]['HCP'][k]
                                    float(data[j])
                                    #print('others','HCP')
                                except:
                                    data[j] = None
                                    break
                

                ##print('data',comb,i['material']['compositionDictionary'][o])
                comb=comb+material['compositionDictionary'][o]*data[j]
                
                sum_comb=sum_comb+material['compositionDictionary'][o]
                        
            if data[j] != None:
                metaIndex_update1[j] = round(float(comb/sum_comb),6)
            else:
                metaIndex_update1[j] = None

    ##print(metaIndex_update1)
    return metaIndex_update1

def structure_calculate(metaIndex_dict,meta_dict,material):
    all_structure=['BCC','FCC','HCP']
    result=[]
    n = 0
    n_index=[]
    try:
        for i in range(len(material['structure'])):
            if material['structure'][i] in all_structure:
                n=n+1
                n_index.append(i) 
        #print('n and st',n)
    except:
        pass
    if n>1:
        #print('n_value',material['formula'],material['structure'][n_index[0]])
        for s in n_index:
            singleResult = structure_choose(metaIndex_dict,meta_dict,material,s)
            singleResult['structure'] = material['structure'][s]
            result.append(singleResult)
    elif n==1:
        #print('n_value1',material['formula'],material['structure'][n_index[0]])
        singleResult = structure_choose(metaIndex_dict,meta_dict,material,n_index[0])
        singleResult['structure'] = material['structure'][n_index[0]]
        result.append(singleResult)
    elif n==0:
        #print('n_value0',material['formula'])
        singleResult = structure_choose(metaIndex_dict, meta_dict, material, None)
        singleResult['structure'] = '?'
        result.append(singleResult)
    ##print(metaIndex_update)
    return result


def linear_combination_run(data, metaParsed = metaParsed,meta_dict = meta_dict,metaIndex = metaIndex):
    
    metaIndex_dict={}
    for k in metaParsed:
        meta_dict[k[2]][k[0]]=k
    ##print(metaIndex,meta_dict)
    for j in range(len(metaIndex)):
        metaIndex_dict[metaIndex[j]] = j
    result = structure_calculate(metaIndex_dict,meta_dict,data['material'])
    return result


def calc_entropy_mixing(comp):
  delta = 0
  for v in comp.get_el_amt_dict().values():
    if v>0:
      delta += v*np.log(v)
  return delta

def get_lc_vals(comp, metaParsed =  metaParsed,meta_dict = meta_dict,metaIndex = metaIndex): 
    ks = {}
    temp = {}
    temp['formula'] = comp.formula
    temp['compositionDictionary']= comp.as_dict()
    temp['reducedFormula']=comp.reduced_formula
    temp['structure']=['?']
    ks['material']=temp
    LCR = linear_combination_run(ks,metaParsed,meta_dict,metaIndex)
    return LCR

def calculate_d_param(comp,metaParsed = metaParsed,meta_dict = meta_dict,metaIndex = metaIndex):
  lc_vals_dict = get_lc_vals(comp,metaParsed,meta_dict,metaIndex)[0]
  return 0.3*float(lc_vals_dict['SurfEne'])/float(lc_vals_dict['USFE'])

def calculate_b_g_ratio(comp,metaParsed = metaParsed,meta_dict = meta_dict,metaIndex = metaIndex):
  lc_vals_dict = get_lc_vals(comp,metaParsed,meta_dict,metaIndex)[0]
  return lc_vals_dict['DFTBh']/lc_vals_dict['DFTGh']

def FT_Rice_92(comp,metaParsed = metaParsed,meta_dict = meta_dict,metaIndex = metaIndex):
    lc_vals_dict = get_lc_vals(comp,metaParsed,meta_dict,metaIndex)[0]
    Shear_Modulus,Unstable_Stacking_Fault_Energy,Poisson_Ratio  = lc_vals_dict['DFTGh'],lc_vals_dict['USFE'],lc_vals_dict['DFTpoisson']
    if None in [Unstable_Stacking_Fault_Energy, Shear_Modulus, Poisson_Ratio]:
        return None
    else:
        return np.sqrt(2*Shear_Modulus*Unstable_Stacking_Fault_Energy/(1-Poisson_Ratio))

def get_price(comp, price_per_kg_dict = price_per_kg_dict, mass_dict = mass_dict):
  system = comp.chemical_system
  list_of_elements=system.split('-')
  tot_price = 0
  for el in list_of_elements:
    ppkg = price_per_kg_dict[el]
    m_kg = mass_dict[el]*0.001
    ppm = ppkg*m_kg
    tot_price += comp.get_atomic_fraction(el)*ppm
  return tot_price


def get_rom_density(comp, mass_dict = mass_dict, vol_dict = vol_dict):
  el_dict = mg.Composition(comp).get_el_amt_dict()
  am = 0
  vol = 0
  for el in el_dict:
    am+=el_dict[el]*mass_dict[el]
    vol+=el_dict[el]*vol_dict[el]
  return am/vol

def calc_vec(comp,vec_dict = vec_dict):
  vec = 0
  div = np.sum(list(comp.get_el_amt_dict().values()))
  for el in comp.get_el_amt_dict().keys():
    vec += (comp.get_el_amt_dict()[el]/div)*vec_dict[el]
  return vec/len(list(comp.get_el_amt_dict().keys()))

def calc_melting_t(comp,metaParsed = metaParsed,meta_dict = meta_dict,metaIndex = metaIndex):
  return get_lc_vals(comp,metaParsed,meta_dict,metaIndex)[0]['MeltingT']

def calc_elec_paul(comp,metaParsed = metaParsed,meta_dict = meta_dict,metaIndex = metaIndex):
  return get_lc_vals(comp,metaParsed,meta_dict,metaIndex)[0]['EleNeg_Pauling']

def calc_atom_rad(comp,metaParsed = metaParsed,meta_dict = meta_dict,metaIndex = metaIndex):
  return get_lc_vals(comp,metaParsed,meta_dict,metaIndex)[0]['Radius_vDW']

def calc_mass(comp,metaParsed = metaParsed,meta_dict = meta_dict,metaIndex = metaIndex):
  return get_lc_vals(comp,metaParsed,meta_dict,metaIndex)[0]['Mass']

def calc_bulk(comp,metaParsed = metaParsed,meta_dict = meta_dict,metaIndex = metaIndex):
  return get_lc_vals(comp,metaParsed,meta_dict,metaIndex)[0]['DFTBh']

def calc_specific_dens_dev(comp, mass_dict = mass_dict, vol_dict = vol_dict,bcc_dict = bcc_dict,
fcc_dict = fcc_dict,hcp_dict = hcp_dict):
    sp_density = 1/get_rom_density(comp,mass_dict,vol_dict)
    el_dict = mg.Composition(comp).get_el_amt_dict()
    el_dict_keys = list(mg.Composition(comp).get_el_amt_dict().keys())
    std_sp_density = 0
    for el in el_dict_keys:
        if el in bcc_dict.keys():
            sp_dens = 1/bcc_dict[el]['density']
        elif el in fcc_dict.keys():
            sp_dens = 1/fcc_dict[el]['density']
        elif el in hcp_dict.keys():
            sp_dens = 1/hcp_dict[el]['density']
        std_sp_density += el_dict[el]*np.abs(sp_dens-sp_density)
    return np.sqrt(std_sp_density/len(el_dict_keys))


def calc_weight_dev(comp):
    atomic_wgt = calc_mass(comp)
    el_dict = mg.Composition(comp).get_el_amt_dict()
    el_dict_keys = list(mg.Composition(comp).get_el_amt_dict().keys())
    std_atomic_wgt = 0
    for el in el_dict_keys:
        atom_wgt = mass_dict[el]
        std_atomic_wgt += el_dict[el]*np.abs(atom_wgt-atomic_wgt)
    return np.sqrt(std_atomic_wgt/len(el_dict_keys))

def calc_group(comp,group_dict=grp_dict):
    el_dict_keys = list(mg.Composition(comp).get_el_amt_dict().keys())
    avg_gp = 0
    for el in el_dict_keys:
        avg_gp += group_dict[el]
    return avg_gp/len(el_dict_keys)

def calc_grp_dev(comp,group_dict=grp_dict):
    grp = calc_group(comp,group_dict)
    el_dict = mg.Composition(comp).get_el_amt_dict()
    el_dict_keys = list(mg.Composition(comp).get_el_amt_dict().keys())
    std_grp = 0
    for el in el_dict_keys:
        temp_grp = group_dict[el]
        std_grp += el_dict[el]*np.abs(temp_grp - grp)
    return np.sqrt(std_grp/len(el_dict_keys))

def properties_from_comp(comps):
    prop_arr = np.zeros((len(comps),8)) #hardcoded as of now
    key = ['Atomic wt. dev.','Column dev.','Specific dens. dev.','VEC','Melt. T.','Bulk mod.','Delta S','Density']
    for i,c in enumerate(comps):
        if type(c) != mg.Composition:
            c = mg.Composition(c)
        adaw = calc_weight_dev(c)
        adc = calc_grp_dev(c)
        adsv = calc_specific_dens_dev(c)
        vec = calc_vec(c)
        melt_t = calc_melting_t(c)
        bulk = calc_bulk(c)
        delta_s = calc_entropy_mixing(c)
        density = get_rom_density(c)
        prop_arr[i] = [adaw,adc,adsv,vec,melt_t,bulk,delta_s,density]
    return prop_arr, key  


