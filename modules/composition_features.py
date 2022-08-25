from function import data_generator_img,check_cuda
import torch


def get_PTR_features(comps,pca,trained_enc,property_list,element_name,RC,cuda=check_cuda()):
  comps_dset = data_generator_img(comps,property_list,element_name,RC)
  test = torch.from_numpy(comps_dset.real_data.astype('float32'))
  if cuda:
    test = test.cuda()
  with torch.no_grad():
    test_encoding = trained_enc(test).to('cpu').detach().numpy()
  X = pca.transform(test_encoding)
  return test_encoding