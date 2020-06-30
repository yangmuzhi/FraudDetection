from trainer import Trainer, exp_val
import numpy as np
trainer = Trainer(data_path="data/uscecchini28.csv")

res_auc, res_ndcg, res_sen, res_prec = trainer.run(preds=range(2003, 2015), 
                                                   model='lgb')
exp_val(res_auc, res_ndcg, res_sen, res_prec, start=2003, end=2011)