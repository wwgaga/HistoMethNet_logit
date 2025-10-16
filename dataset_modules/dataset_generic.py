import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
import random
from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth, ci_loss_interval

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
		
	else:
		df = pd.concat(splits, ignore_index=True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns=['train', 'val', 'test'])

	df.to_csv(filename, index=False)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False,
		label_col = None,
		patient_voting = 'max',
		training = False
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		self.training = training
		
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		self.patient_cls_ids = [[] for i in range(self.num_classes)]		
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		# store ids corresponding each class at the slide level
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				
				label = np.unique(label)[0]
				
				# stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		# print(patient_labels)
		# # patient_labels = np.array(patient_labels, dtype=object)
		# print(patient_labels)
		# exit()
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids
					}

		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))] 

			for split in range(len(ids)): 
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)
		if split_key == 'train':
			training_flag = True
		else:
			training_flag = False
		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):

		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
			
			else:
				test_split = None
			
			# if val_split is not None and test_split is not None:
			# 	val_data_combined = pd.concat([val_data, test_data]).reset_index(drop=True)
			# 	val_split = Generic_Split(val_data_combined, data_dir=self.data_dir, num_classes=self.num_classes)
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			
			# if val_split is not None and test_split is not None:
			# 	val_data_combined = pd.concat([val_split.slide_data, test_split.slide_data]).reset_index(drop=True)
			# 	val_split = Generic_Split(val_data_combined, data_dir=self.data_dir, num_classes=self.num_classes)
		
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]

		print('hello1 Generic_MIL_Dataset is called')
		exit()
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir
		
		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
				features = torch.load(full_path)
				return features, label
			
			else:
				# return slide_id, label
				return slide_id, label

		else:
			full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, label, coords




class Generic_MIL_Dataset_cell_type(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset_cell_type, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False
		

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def sampling(self, index_i_list, index_j_list, cell_type_percentages_i, cell_type_percentages_j, stem_i, stem_j, immune_i, immune_j):
		"""Sampling methods
		Args:
			index_i (list):
			index_j (list):
			sampled_lp (float)
		Returns:
			expected_lp (float):
			index_i (list):
			index_j (list):
			min_error (float):
			max_error (float):
		"""
		
		# Convert tensors to lists of floats
		cell_type_percentages_i = cell_type_percentages_i.detach().cpu().numpy()
		cell_type_percentages_j = cell_type_percentages_j.detach().cpu().numpy()
		stem_i = stem_i.detach().cpu().numpy()
		stem_j = stem_j.detach().cpu().numpy()
		# immune_i = immune_i.detach().cpu().numpy()
		# immune_j = immune_j.detach().cpu().numpy()
		
		# uniform sampling
		sep_i = np.random.randint(1, len(index_i_list))
		sep_j = np.random.randint(1, len(index_j_list))
		index_i = random.sample(index_i_list, sep_i)	
		index_j = random.sample(index_j_list, sep_j) # V1: len(index_j_list) - sep_j;  V2: sep_j
		
		CI	= 0.005 # confidence_interval = 0.005 means 99% confidential interval"
		# print(cell_type_percentages_i, cell_type_percentages_j)
		# print(len(index_i), len(index_j))
		
		ci_min_cell_type_percentages, ci_max_cell_type_percentages, expected_cell_type_percentages = ci_loss_interval(
			cell_type_percentages_i, cell_type_percentages_j, len(index_i), len(index_j), CI	
		)
		
		ci_min_stem, ci_max_stem, expected_stem = ci_loss_interval(
			stem_i, stem_j, len(index_i), len(index_j), CI	
		)
		ci_min_immune, ci_max_immune, expected_immune = ci_loss_interval(
			immune_i, immune_j, len(index_i), len(index_j), CI	
		)
	
		 
		result_dict = {
			'index_i': index_i,
			'index_j': index_j,
			'expected_stem': expected_stem,
			'ci_min_stem': ci_min_stem,
			'ci_max_stem': ci_max_stem,
			'expected_immune': expected_immune,
			'ci_min_immune': ci_min_immune,
			'ci_max_immune': ci_max_immune,
			'expected_cell_type_percentages': expected_cell_type_percentages,
			'ci_min_cell_type_percentages': ci_min_cell_type_percentages,
			'ci_max_cell_type_percentages': ci_max_cell_type_percentages
		}
		return result_dict

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]
		
		training = (len(self.slide_data['slide_id']) > 150)
		
		stem_like = self.slide_data['stem-like'][idx]
		diff = self.slide_data['diff'][idx]
		immune = self.slide_data['immune'][idx]
		neuron = self.slide_data['neuron'][idx]
		glia = self.slide_data['glia'][idx]

		# Extract cell type percentages
		endothelial = self.slide_data['endothelial'][idx]
		GN = self.slide_data['GN'][idx]
		lymphoid_cells = self.slide_data['lymphoid_cells'][idx]
		MES_ATYP = self.slide_data['MES_ATYP'][idx]
		MES_TYP = self.slide_data['MES_TYP'][idx]
		myeloid_cells = self.slide_data['myeloid_cells'][idx]
		RTK_I = self.slide_data['RTK_I'][idx]
		RTK_II = self.slide_data['RTK_II'][idx]
		
		# stem = torch.tensor([
		# 	stem_like, 1-stem_like
		# ], dtype=torch.float)

		stem = torch.tensor([
			stem_like, diff, immune, neuron, 
			glia
		], dtype=torch.float)

		immune = torch.tensor([
			immune, 1-immune
		], dtype=torch.float)

		cell_type_percentages = torch.tensor([
			endothelial, GN, lymphoid_cells, MES_ATYP, 
			MES_TYP, myeloid_cells, RTK_I, RTK_II
		], dtype=torch.float)
		
		if isinstance(self.data_dir, dict):
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir
		
		if not self.use_h5:	
			full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
			features = torch.load(full_path, weights_only=True)
		else:
			full_path = os.path.join(data_dir, 'h5_files', '{}.h5'.format(slide_id))
			with h5py.File(full_path, 'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]
			features = torch.from_numpy(features)

		# MixBag = random.choice([True, True, True, True, True, False, False])
		
		j = np.random.randint(0, len(self.slide_data['slide_id']))
		slide_id_j, labels_j = (
			self.slide_data['slide_id'][j],
			self.slide_data['label'][j]
		)
		
		MixBag = (labels_j==label)
		
		# MixBag = False
		if MixBag and training:
			# print('mixbag turns on!!')
			# j = np.random.randint(0, len(self.slide_data['slide_id']))
			# slide_id_j, labels_j = (
			# 	self.slide_data['slide_id'][j],
			# 	self.slide_data['label'][j]
			# )
			id_list_i = list(range(features.shape[0]))
			if not self.use_h5:
				full_path_j = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id_j))
				features_j = torch.load(full_path_j, weights_only=True)
			
			id_list_j = list(range(features_j.shape[0]))

			stem_like_j = self.slide_data['stem-like'][j]
			
			diff_j = self.slide_data['diff'][j]
			# immune = self.slide_data['immune'][j]
			immune_j = self.slide_data['immune'][j]

			neuron_j = self.slide_data['neuron'][j]
			glia_j = self.slide_data['glia'][j]


			endothelial_j = self.slide_data['endothelial'][j]
			GN_j = self.slide_data['GN'][j]
			lymphoid_cells_j = self.slide_data['lymphoid_cells'][j]
			MES_ATYP_j = self.slide_data['MES_ATYP'][j]
			MES_TYP_j = self.slide_data['MES_TYP'][j]
			myeloid_cells_j = self.slide_data['myeloid_cells'][j]
			RTK_I_j = self.slide_data['RTK_I'][j]
			RTK_II_j = self.slide_data['RTK_II'][j]
			
			
			stem_j = torch.tensor([
			stem_like_j, diff_j, immune_j, neuron_j, 
			glia_j
			], dtype=torch.float)

			immune_j= torch.tensor([
				immune_j, 1-immune_j
			], dtype=torch.float)

			cell_type_percentages_j = torch.tensor([
				endothelial_j, GN_j, lymphoid_cells_j, MES_ATYP_j, 
				MES_TYP_j, myeloid_cells_j, RTK_I_j, RTK_II_j
			], dtype=torch.float)
			
			
            # expected_lp: mixed_bag's label proportion
            # id_i: index used for creating subbag_i from data_i
            # id_j: index used for creating subbag_j from data_j
            # ci_min: minimam value of confidence interval
            # ci_max: maximam value of confidence interval
			mix_sample_result_dict = self.sampling(id_list_i, id_list_j, cell_type_percentages, cell_type_percentages_j, stem, stem_j, immune, immune_j)

			index_i = mix_sample_result_dict['index_i']
			index_j = mix_sample_result_dict['index_j']
			
			subbag_i = features[index_i]
			subbag_j = features_j[index_j]

			mixed_bag = np.concatenate([subbag_i, subbag_j], axis=0)
			mixed_label = label

			features = mixed_bag
			stem_min = mix_sample_result_dict['ci_min_stem']
			stem_max = mix_sample_result_dict['ci_max_stem']

			immune_min = mix_sample_result_dict['ci_min_immune']
			immune_max = mix_sample_result_dict['ci_max_immune']
			
			cell_type_percentages_min = mix_sample_result_dict['ci_min_cell_type_percentages']
			cell_type_percentages_max = mix_sample_result_dict['ci_max_cell_type_percentages']
			
			features = torch.tensor(mixed_bag)
			
			stem = torch.tensor(mix_sample_result_dict['expected_stem'])
			immune = torch.tensor(mix_sample_result_dict['expected_immune'])
			cell_type_percentages = torch.tensor(mix_sample_result_dict['expected_cell_type_percentages'])	
			
			stem_min = torch.tensor(stem_min)
			stem_max = torch.tensor(stem_max)
			immune_min = torch.tensor(immune_min)
			immune_max = torch.tensor(immune_max)
			cell_type_percentages_min = torch.tensor(cell_type_percentages_min)
			cell_type_percentages_max = torch.tensor(cell_type_percentages_max)
			
		
		if not MixBag or not training:
			# print('NO mixbag !!')
			stem_min, stem_max = (
                torch.full((1, 5), -1).reshape(5).float(),
                torch.full((1, 5), -1).reshape(5).float(),
            )
			immune_min = torch.full((1, 2), -1).reshape(2).float()
			immune_max = torch.full((1, 2), -1).reshape(2).float()
			
			cell_type_percentages_min = torch.full((1, 8), -1).reshape(8).float()
			cell_type_percentages_max = torch.full((1, 8), -1).reshape(8).float()
			
		min_max_dict = {
			'stem_min': stem_min,
			'stem_max': stem_max,
			'immune_min': immune_min,
			'immune_max': immune_max,
			'cell_type_min': cell_type_percentages_min,
			'cell_type_max': cell_type_percentages_max
		}
	
		if not self.use_h5:
			if self.data_dir:
				
				return features, label, stem, immune, cell_type_percentages, min_max_dict
			else:
				return slide_id, label, stem, immune, cell_type_percentages, min_max_dict
		else:
			return features, label, stem, immune, cell_type_percentages, coords, min_max_dict
		
		

# class Generic_Split(Generic_MIL_Dataset): uncomment if you want to use the original dataset
class Generic_Split(Generic_MIL_Dataset_cell_type):
	def __init__(self, slide_data, data_dir=None, num_classes=2):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
		
	def __len__(self):
		return len(self.slide_data)
		


