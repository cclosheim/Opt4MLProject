import numpy as np
import pandas as pd
from Crossvalidation import CV
from Model import GD, GD_fix_gamma,SGD,ADAM,RMSProp
from SaveData import save_experiments

experiment_name='exp_wine_final_with_decrease_SGD_sqrt.json'
np.random.seed(0)
iterations=1200

data_wine = pd.read_csv('data/wine.csv',sep=';')
Xtrain, ytrain = data_wine.drop(columns=['quality']), data_wine['quality']
cv=CV()


gd=GD_fix_gamma(iterations)

sgd_1=SGD(iterations,batch_size=1)
sgd_16=SGD(iterations,batch_size=16)
sgd_32=SGD(iterations,batch_size=32)

rms_1=RMSProp(iterations,batch_size=1)
rms_16=RMSProp(iterations,batch_size=16)
rms_32=RMSProp(iterations,batch_size=32)

adam_1=ADAM(iterations,batch_size=1)
adam_16=ADAM(iterations,batch_size=16)
adam_32=ADAM(iterations,batch_size=32)


models=[gd,sgd_1,sgd_16,sgd_32, rms_1,rms_16,rms_32,adam_1,adam_16,adam_32]
param_grid = {'gamma': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7], 'alpha':[1e1,1,0,1e-1,1e-2]}
mse_train_per_model, mse_test_per_model, mean_runtime_per_model,mses_train_per_model, mses_test_per_model, runtimes_per_model,best_parameters_per_model =cv.cross_validate_multiple_model(models,Xtrain,ytrain,param_grid)


save_experiments(experiment_name, models, mse_train_per_model, mse_test_per_model, mean_runtime_per_model,mses_train_per_model, mses_test_per_model, runtimes_per_model,best_parameters_per_model)
