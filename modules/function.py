import re
import numpy as np
import copy
import os
import torch
from pymatgen import core as mg

def check_cuda():
  if torch.cuda.is_available():
    cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
  else:
    cuda = False
  return cuda

def special_formatting(comp):
  """take pymatgen compositions and does string formatting"""
  comp_d = comp.get_el_amt_dict()
  denom = np.sum(list(comp_d.values()))
  string = ''
  for k in comp_d.keys():
    string += k + '$_{' + '{}'.format(round(comp_d[k]/denom,3)) + '}$'
  return string


def image_gfa(i,property_list,element_name,RC):#PTR psuedoimage using special formula
    #i0='Mo.5Nb.5'
    #i=i0.split(' ')[0]

    X= [[[0.0 for ai in range(18)]for aj in range(9)] for ak in range(1) ]
    gfa=re.findall('\[[a-c]?\]',i)[0]  
    tx1_element=re.findall('[A-Z][a-z]?', i)#[B, Fe, P,No]
    tx2_temp=re.findall('\$_{[0-9.]+}\$', i)#[$_{[50]}$, ] [50 30 20]
    tx2_value=[float(re.findall('[0-9.]+', i_tx2)[0]) for i_tx2 in tx2_temp]
    for j in range(len(tx2_value)):
        index=int(property_list[element_name.index(tx1_element[j])][1])#atomic number
        xi=int(RC[index-1][1])#row num
        xj=int(RC[index-1][2])#col num
        X[0][xi-1][xj-1]=tx2_value[j]/100.0
    
    #properties at the first row, from 5th to 8th column for hardness
    X = np.array(X)
    return X


def PTR(i,property_list,element_name,Z_row_column):#periodical table representation
    #i='4 La$_{66}$Al$_{14}$Cu$_{10}$Ni$_{10}$ [c][15]'
    X= [[[0.0 for ai in range(18)]for aj in range(9)] for ak in range(1) ]
    gfa=re.findall('\[[a-c]?\]',i)[0]
    
    tx1_element=re.findall('[A-Z][a-z]?', i)#[B, Fe, P,No]
    tx2_temp=re.findall('\$_{[0-9.]+}\$', i)#[$_{[50]}$, ] [50 30 20]
    tx2_value=[float(re.findall('[0-9.]+', i_tx2)[0]) for i_tx2 in tx2_temp]
    for j in range(len(tx2_value)):
        index=int(property_list[element_name.index(tx1_element[j])][1])#atomic number Z
        xi=int(Z_row_column[index-1][1])#row num
        xj=int(Z_row_column[index-1][2])#col num
        X[0][xi-1][xj-1]=tx2_value[j]/100.0
    X_BMG=copy.deepcopy(X)
    X_BMG[0][0][8]=1.0 #processing parameter
    
    if gfa=='[c]':
        Y=[0,0]
    if gfa=='[b]': 
        Y=[1,0]
    if gfa=='[a]' :
        Y=[1,1]

    return [X,X_BMG],Y 

class data_generator_gfa(object):
    def __init__(self, comps, gfa_dataset):

        #with open(csv_file, 'r') as fid:
            #l = fid.readlines()
        #data = [x.strip().split(',')[1] for x in l]
        #data.remove('Composition')

        #remove single elements from dataset, want only HEAs. Also keep unqiue compositions
        self.length = len(comps)
        all_imgs = []
        for i in range(len(gfa_dataset)):
          c_img = image_gfa(gfa_dataset[i])
          all_imgs.append(c_img)
        
        self.real_data = np.array(all_imgs).reshape(-1,1,9,18)

    def sample(self, N):
        idx = np.random.choice(np.arange(self.length), N, replace=False)
        data = self.real_data[idx]

        return np.array(data, dtype=np.float32)



