import re
import numpy as np
import copy
import os
import pickle
import joblib
import torch
from pymatgen import core as mg
from .encoder import Encoder,Identity
from sklearn import metrics
#from modules.representation_schemes import get_atomic_number_features, get_pettifor_features, get_modified_pettifor_features, get_random_features, get_random_features_dense, random_order_alpha


gfa_dataset_file = 'gfa_dataset.txt'
z_row_column_file = 'Z_row_column.txt'
element_property_file = 'element_property.txt'
common_path = "Files_from_GTDL_paper/{}" 
gfa_dataset = pickle.load(open(common_path.format(gfa_dataset_file), 'rb'))  
RC = pickle.load(open(common_path.format(z_row_column_file), 'rb')) 
new_index=[int(i[4]) for i in RC]#new order 
Z_row_column = pickle.load(open(common_path.format(z_row_column_file), 'rb'))
[property_name_list,property_list,element_name,_]=pickle.load(open(common_path.format(element_property_file), 'rb'))


def convert_hv_to_gpa(hv_list):
  gpa_list = [x*0.009807 for x in hv_list]
  return np.array(gpa_list)

def check_cuda():
  if torch.cuda.is_available():
    cuda = True
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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


def image(i,property_list = property_list,element_name = element_name,RC = RC, normalized=True):
  #PTR psuedoimage using pymatgen
    #i0='Mo.5Nb.5'
    #i=i0.split(' ')[0]
    X= [[[0.0 for ai in range(18)]for aj in range(9)] for ak in range(1) ]  
    tx1_element=re.findall('[A-Z][a-z]?', i)#[B, Fe, P,No]
    tx2_temp=re.findall('[0-9.]+', i)#[$_{[50]}$, ] [50 30 20]
    tx2_value=[float(re.findall('[0-9.]+', i_tx2)[0]) for i_tx2 in tx2_temp]
    for j in range(len(tx2_value)):
        index=int(property_list[element_name.index(tx1_element[j])][1])#atomic number
        xi=int(RC[index-1][1])#row num
        xj=int(RC[index-1][2])#col num
        X[0][xi-1][xj-1]=tx2_value[j]

    #properties at the first row, from 5th to 8th column for hardness
    X = np.array(X)
    if normalized:
      return X/np.sum(X)
    else:
      return X

def image_modified(i0,property_list = property_list,element_name = element_name,RC = RC):
  #PTR psuedoimage using special formula
    #i0='Mo.5Nb.5'
    i=i0.split(' ')[1]
    X= [[[0.0 for ai in range(18)]for aj in range(9)] for ak in range(1) ]  
    tx1_element=re.findall('[A-Z][a-z]?', i)#[B, Fe, P,No]
    tx2_temp=re.findall('[0-9.]+', i)#[$_{[50]}$, ] [50 30 20]
    tx2_value=[float(re.findall('[0-9.]+', i_tx2)[0]) for i_tx2 in tx2_temp]
    for j in range(len(tx2_value)):
        index=int(property_list[element_name.index(tx1_element[j])][1])#atomic number
        xi=int(RC[index-1][1])#row num
        xj=int(RC[index-1][2])#col num
        X[0][xi-1][xj-1]=tx2_value[j]

    #properties at the first row, from 5th to 8th column for hardness
    X = np.array(X)
    return X/np.sum(X)

def PTR(i,property_list = property_list,element_name = element_name,Z_row_column = Z_row_column):#periodical table representation
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
    process = [0,1]
    
    if gfa=='[c]':
        Y=[0,0]
    if gfa=='[b]': 
        Y=[1,0]
    if gfa=='[a]' :
        Y=[1,1]

    return [X,X_BMG],process,Y 

def PTR_modified(i,property_list = property_list,element_name = element_name,Z_row_column = Z_row_column):#periodical table representation
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
    
    if gfa=='[c]':
        Y=[0]
    if gfa=='[b]': 
        Y=[1]
    if gfa=='[a]' :
        Y=[1]

    return [X],Y 

class data_generator_vec(object):
    def __init__(self, comps, el_list = []):

        #with open(csv_file, 'r') as fid:
            #l = fid.readlines()
        #data = [x.strip().split(',')[1] for x in l]
        #data.remove('Composition')

        #remove single elements from dataset, want only HEAs. Also keep unqiue compositions

        if len(el_list) == 0:
          all_eles = []
          for c in comps:
            all_eles += list(c.get_el_amt_dict().keys())
          eles = np.array(sorted(list(set(all_eles))))
        else:
          eles = np.array(el_list)
          
        self.elements = eles
        self.size = len(eles)
        self.length = len(comps)

        all_vecs = np.zeros([len(comps), len(self.elements)])
        for i, c in enumerate(comps):
            for k, v in c.get_el_amt_dict().items():
                j = np.argwhere(eles == k)
                all_vecs[i, j] = v
        all_vecs = all_vecs / np.sum(all_vecs, axis=1).reshape(-1, 1)
        self.real_data = np.array(all_vecs, dtype=np.float32)

    def sample(self, N):
        idx = np.random.choice(np.arange(self.length), N, replace=False)
        data = self.real_data[idx]

        return np.array(data, dtype=np.float32),idx
    
    def elements(self):
      return eles

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
          c_img = image(gfa_dataset[i])
          all_imgs.append(c_img)
        
        self.real_data = np.array(all_imgs).reshape(-1,1,9,18)

    def sample(self, N):
        idx = np.random.choice(np.arange(self.length), N, replace=False)
        data = self.real_data[idx]

        return np.array(data, dtype=np.float32)

class data_generator_img(object):
    def __init__(self, comps):

        #with open(csv_file, 'r') as fid:
            #l = fid.readlines()
        #data = [x.strip().split(',')[1] for x in l]
        #data.remove('Composition')

        #remove single elements from dataset, want only HEAs. Also keep unqiue compositions

        all_eles = []
        for c in comps:
            all_eles += list(c.get_el_amt_dict().keys())
        eles = np.array(sorted(list(set(all_eles))))

        self.elements = eles
        self.size = len(eles)
        self.length = len(comps)

        sp_comps = [special_formatting(x) for x in comps]

        all_imgs = []
        for i in range(len(sp_comps)):
          c_img = image(sp_comps[i])
          all_imgs.append(c_img)
        
        self.real_data = np.array(all_imgs).reshape(-1,1,9,18)

    def sample(self, N):
        idx = np.random.choice(np.arange(self.length), N, replace=False)
        data = self.real_data[idx]

        return np.array(data, dtype=np.float32).reshape(-1,1,9,18)
    
    def elements(self):
      return eles


def decode_img(image,property_list = property_list,element_name=element_name):
  """from image, get the composition"""
  image = image.reshape(-1,1,9,18)
  row,col = np.nonzero(image)[2:]
  comp_dict = {}
  props = []
  for j in range(len(row)):
    for r in RC:
      if int(r[1]) == row[j]+1 and int(r[2]) == col[j]+1:
        for i in range(len(property_list)):
         if int(property_list[i][1]) == int(r[0]) and image[0][0][row[j]][col[j]] >= 0.0:
            comp_dict[element_name[i]] = image[0][0][row[j]][col[j]]
  return mg.Composition(comp_dict)




def stratify_data(data, min, max, by):
  samples_per_bin, bins, = np.histogram(data, bins=np.arange(min,max,by))
  return np.digitize(data,bins) 


def get_elem_count(comp_list:list):
  elem_dict = {}
  
  for c in comp_list:
    c = mg.Composition(c)
    for elems in c.get_el_amt_dict().keys():
        if elems not in elem_dict.keys():
          elem_dict[elems] = 1
        else:
          elem_dict[elems] += 1
  return elem_dict

def get_number_of_components(comp_list):
  count_list = []
  for c in comp_list:
    if not type(c) == mg.Composition:
      c = mg.Composition(c)
      count_list.append(len(list(c.get_el_amt_dict().keys())))
  return count_list

def get_comp_count_over_bins(vals, nbins=10):
    max_dig = len(str(int(max(abs(x) for x in vals))))
    return max_dig, np.linspace(np.round(vals.min()),np.round(vals.max()), nbins)

def pymatgen_comp(comp_list):
  return [mg.Composition(x) for x in comp_list]


def get_metrics(true_labels, predicted_labels):
    accuracy=np.round(metrics.accuracy_score(true_labels,predicted_labels),4)
    precision=np.round(metrics.precision_score(true_labels, predicted_labels,average='weighted'),4)
    recall= np.round(metrics.recall_score(true_labels,predicted_labels,average='weighted'),4)
    F1=np.round(metrics.f1_score(true_labels,predicted_labels,average='weighted'),4)
                                           
#    print('Accuracy:', accuracy)
#    print('Precision:', precision)
#    print('Recall:', recall)
#    print('F1 Score:', F1)
    return [accuracy,precision,recall,F1]



def alt_read_gfa_dataset(dataset = gfa_dataset,to_discard = ['Rf','Db','Sg','Bh','Hs']):
    str_comp = []
    Y = []
    p = []
    for i in  dataset:
        
        gfa=re.findall('\[[a-c]?\]',i)[0]
        tx1_element=re.findall('[A-Z][a-z]?', i)#[B, Fe, P,No]
        tx2_temp=re.findall('\$_{[0-9.]+}\$', i)#[$_{[50]}$, ] [50 30 20]
        tx2_value=[float(re.findall('[0-9.]+', i_tx2)[0]) for i_tx2 in tx2_temp]
        test = ''.join([x+str(y) for x,y in zip(tx1_element,tx2_value)])
        if gfa=='[c]':
            y= [0,0]
        if gfa=='[b]':
             y = [1,0]
        if gfa == '[a]': 
            y = [1,1]
        if len(set(tx1_element).intersection(set(to_discard))) == 0:
            str_comp.extend([test]*2)
            Y.extend(y)
            p.extend([0,1])
    return pymatgen_comp(str_comp),Y, p

def read_gfa_dataset(dataset = gfa_dataset,to_discard = ['Rf','Db','Sg','Bh','Hs']):
    X = []
    Y = []
    p = []
    for i in  dataset:
      tx1_element=re.findall('[A-Z][a-z]?', i)
      X_temp, p_temp, Y_temp = PTR(i)
      if len(set(tx1_element).intersection(set(to_discard))) == 0:
            X.extend(X_temp)
            Y.extend(Y_temp)
            p.extend(p_temp)
    X = np.array(X).reshape(-1, 1,9, 18).astype('float32') 
    Y = np.array(Y).reshape(-1,1).astype('float32')
    p = np.array(p).reshape(-1,1).astype('float32')
    return X ,Y, p


