import pandas as pd
import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, ndcg_score


class Trainer:
	
	def __init__(self, data_path):
		self.data_path = data_path
	
	
	def run(self, preds=[], model='lgb', label_weight=True):
		# return a dict
		res_auc = {}
		res_ndcg = {}
		res_sen = {}
		res_prec = {}
		
		for pred in tqdm.tqdm(preds):
			auc, ndcg, sen, prec = self.train(pred=pred, model=model, label_weight=label_weight)
			res_auc[pred] = auc
			res_ndcg[pred] = ndcg
			res_sen[pred] = sen
			res_prec[pred] = prec
		return res_auc, res_ndcg, res_sen, res_prec
	
	def make_data(self, pred=2003):
		# return train, test
		data = pd.read_csv(self.data_path)
		train_data = pd.DataFrame([])
		
		beg = 1990
		end = pred - 1
		for year in range(beg, end):
			train_data = pd.concat([data[data['fyear']==year], train_data])
		ind = np.arange(train_data.shape[0])
		np.random.shuffle(ind)
		train_data = train_data.iloc[ind, :]
		
		return train_data, data[data['fyear']==pred]
	
	def train(self, pred=2003, model='lgb', label_weight=True):
		train_data, test_data= self.make_data(pred)
		train_labels = train_data.iloc[:, 8]
		train_features = train_data.iloc[:, 9:36]
		test_labels = test_data.iloc[:, 8]
		test_features = test_data.iloc[:, 9:36]
		
		if model == 'lgb':
			import lightgbm as lgb
			param = {'num_leaves': 31, 'objective': 'binary'}
			param['metric'] = ['auc', 'binary_logloss']

			train_data = lgb.Dataset(train_features, label=train_labels)
# 			test_data = lgb.Dataset(test_features,label=test_labels)
			num_round = 50
			self.model = lgb.train(param, train_data, num_round)
		else:
			from sklearn.ensemble import RandomForestClassifier
			if label_weight:
				self.model = RandomForestClassifier(class_weight={0:1, 1:150}, max_depth=5)
			else:
				self.model = RandomForestClassifier(max_depth=5)
			
			self.model.fit(train_features, train_labels)
		
		auc, ndcg, sen, prec = self.eval(test_features, test_labels)
		return auc, ndcg, sen, prec
		
		
	def eval(self, test_features, test_labels):
		ypred = self.model.predict(test_features)
		
		auc = roc_auc_score(test_labels, ypred)
		ypred = (ypred > 0.5).astype(np.int)
		ndcg = ndcg_score(np.array(test_labels).reshape(1, -1), ypred.reshape(1, -1), k=int(0.01 * test_labels.shape[0]))
		tn, fp, fn, tp = confusion_matrix(test_labels, ypred).ravel()
		sen = tp / (tp + fn)
		prec = tp / (tp + fp)
		return auc, ndcg, sen, prec

def exp_val(res_auc, res_ndcg, res_sen, res_prec, start, end):
	auc, ndcg, sen, prec = [], [], [], []
	for year in range(start, end+1):
		auc.append(res_auc.get(year))
		ndcg.append(res_ndcg.get(year))
		sen.append(res_sen.get(year))
		prec.append(res_prec.get(year))
		
	return np.mean(auc), np.mean(ndcg), np.mean(sen), np.mean(prec)