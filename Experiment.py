import numpy as np
import pandas as pd
from Crossvalidation import CV
from Model import GD, GD_fix_gamma,SGD,ADAM,RMSProp
from SaveData import save_experiments

np.random.seed(0)
iterations=10000

experiment_name='exp_bike_final_GD_smaller_lr.json'
data_bike = pd.read_csv('data/SeoulBikeData.csv')
Xtrain, ytrain = data_bike.drop(columns=['Rented Bike Count','Date','Seasons','Holiday','Functioning Day']), data_bike['Rented Bike Count']
cv=CV()


gd=GD_fix_gamma(iterations)

sgd_32=SGD(iterations,batch_size=32)

rms_32=RMSProp(iterations,batch_size=32)

adam_32=ADAM(iterations,batch_size=32)


models=[gd,sgd_32,rms_32,adam_32]
param_grid = {'gamma': [1e-7,1e-8,1e-9,1e-10,1e-11], 'alpha':[1e1,1,0,1e-1,1e-2]}
mse_train_per_model, mse_test_per_model, mean_runtime_per_model,mses_train_per_model, mses_test_per_model, runtimes_per_model,best_parameters_per_model =cv.cross_validate_multiple_model(models,Xtrain,ytrain,param_grid)

save_experiments(experiment_name, models, mse_train_per_model, mse_test_per_model, mean_runtime_per_model,mses_train_per_model, mses_test_per_model, runtimes_per_model,best_parameters_per_model)
