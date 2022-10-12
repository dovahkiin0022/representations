import torch
import pickle
import joblib
import numpy as np
from pymatgen import core as mg
import pandas as pd
import os
from .function import pymatgen_comp, check_cuda, data_generator_img, data_generator_vec
from .encoder import Encoder,Identity
from .rom import calc_weight_dev, calc_grp_dev, calc_specific_dens_dev, calc_vec, calc_melting_t, calc_bulk, calc_entropy_mixing, get_rom_density

gfa_dataset_file = 'gfa_dataset.txt'
z_row_column_file = 'Z_row_column.txt'
element_property_file = 'element_property.txt'
common_path = "Files_from_GTDL_paper/{}" 
gfa_dataset = pickle.load(open(common_path.format(gfa_dataset_file), 'rb'))  
RC = pickle.load(open(common_path.format(z_row_column_file), 'rb')) 
new_index=[int(i[4]) for i in RC]#new order 
Z_row_column = pickle.load(open(common_path.format(z_row_column_file), 'rb'))
[property_name_list,property_list,element_name,_]=pickle.load(open(common_path.format(element_property_file), 'rb'))

periodic_table_file = 'dataset/periodic_table.csv'
periodic_df = pd.read_csv(periodic_table_file)
atomic_number_order = periodic_df['Symbol'].values[:103] #only the first 103 elements

alternate_orders_file = 'dataset/alternate_orders.pkl'
with open(alternate_orders_file,'rb') as fid:
    alternate_order_dict = pickle.load(fid)
pettifor_order = alternate_order_dict['pettifor']
modified_pettifor_order = alternate_order_dict['modified_pettifor']
random_order = sorted(atomic_number_order)


def get_PTR_features(comps,cuda=check_cuda()):
  saved_models_path = 'saved_models/best_models'
  filename = '2DEncoder_PTR.pt'
  if os.path.exists(f'{saved_models_path}/{filename}'):
    PTR_encoder =  joblib.load(f'{saved_models_path}/{filename}')
  else:
    print('No file found!')
  #PTR_encoder.mapf = Identity()
  comps = pymatgen_comp(comps)
  comps_dset = data_generator_img(comps)
  test = torch.from_numpy(comps_dset.real_data.astype('float32'))
  if cuda:
    test = test.cuda()
  with torch.no_grad():
    test_encoding = PTR_encoder.hidden_rep(test).to('cpu').detach().numpy()
  return test_encoding


def properties_from_comp(comps):
    prop_arr = np.zeros((len(comps),8)) #hardcoded as of now
    key = ['Atomic wt. dev.','Column dev.','Specific dens. dev.','VEC','Melt. T.','Bulk mod.','Delta S','Density']
    comps = pymatgen_comp(comps)
    for i,c in enumerate(comps):
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

def get_vectorized_featues(comps):
  comps = pymatgen_comp(comps)
  dset = data_generator_vec(comps)
  return dset.real_data, dset.elements

def get_atomic_number_features(comps,el_list = atomic_number_order):
    comps = pymatgen_comp(comps)
    dset = data_generator_vec(comps,el_list)
    return dset.real_data.reshape(-1,1,len(el_list)), dset.elements

def get_pettifor_features(comps,el_list = pettifor_order):
    comps = pymatgen_comp(comps)
    dset = data_generator_vec(comps,el_list)
    return dset.real_data.reshape(-1,1,len(el_list)), dset.elements

def get_modified_pettifor_features(comps,el_list = modified_pettifor_order):
    comps = pymatgen_comp(comps)
    dset = data_generator_vec(comps,el_list)
    return dset.real_data.reshape(-1,1,len(el_list)), dset.elements

def get_random_features(comps, el_list = random_order):
    comps = pymatgen_comp(comps)
    dset = data_generator_vec(comps,el_list)
    return dset.real_data.reshape(-1,1,len(el_list)), dset.elements

def get_random_features_dense(comps, el_list = random_order):
    comps = pymatgen_comp(comps)
    dset = data_generator_vec(comps,el_list)
    return dset.real_data.reshape(-1, len(el_list)), dset.elements

def enc1d_features(comps, name, cuda=check_cuda()):
  types = ['atomic','pettifor','mod_pettifor','random','dense']
  location = 'saved_models/best_models'
  if name not in types:
    print('Invalid format')
  else:
    encoder1D = joblib.load(os.path.join(location,'1DEncoder_{}.pt'.format(name)))
    #encoder1D.mapf = Identity()
  comps = pymatgen_comp(comps)
  if name == 'atomic':
    formatted_comps,_ = get_atomic_number_features(comps)
  elif name == 'pettifor':
    formatted_comps,_ = get_pettifor_features(comps)
  elif name == 'mod_pettifor':
    formatted_comps,_ = get_modified_pettifor_features(comps)
  elif name == 'random':
    formatted_comps,_ = get_random_features(comps)
  elif name == 'dense':
    formatted_comps,_ = get_random_features_dense(comps)
  test = torch.from_numpy(formatted_comps.astype('float32'))
  if cuda:
    test = test.cuda()
  with torch.no_grad():
    test_encoding = encoder1D.hidden_rep(test).to('cpu').detach().numpy()
  return test_encoding


