from sklearn.model_selection import KFold
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product

# Inspired from Code from EML lecture WS 22/23


def eval_model(model, X_train, Y_train, X_test, Y_test, parameters):

    print('start'+model.get_name() +str(time.time()))
    kf = KFold(n_splits=5, random_state=0, shuffle=True)

    best_score = None
    best_params = None

    param_combinations = list(product(*parameters.values()))
    for params in param_combinations:
        model.set_parameter(params)

        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_train_train, y_train_train = X_train.iloc[train_idx], Y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], Y_train.iloc[val_idx]

            X_train_train = X_train_train.reset_index(drop=True)
            y_train_train = y_train_train.reset_index(drop=True)

            model.fit(X_train_train, y_train_train)

            y_pred = model.predict(X_val)
            score = min([0.5*((y_val - ypred) ** 2).sum() / len(y_val) for ypred in y_pred])
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        if best_score is None or avg_score < best_score:
            best_score = avg_score
            best_params = params


    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    model.set_parameter(best_params)
    fit_runtime, train_error = model.fit(X_train, Y_train)

    ypred_test = model.predict(X_test)

    mses_train = train_error
    mses_test = [0.5*((Y_test - ypred) ** 2).sum() / len(Y_test) for ypred in ypred_test]
    print('end' + model.get_name() + str(time.time()))
    return mses_train, mses_test, fit_runtime,best_params

class CV():
    def __init__(self):
        self.folds = 5

        self.loops=1
    def cross_validate(self, model, X, Y):
        """Performs cross validation for a regression model"""

        mse_list_train = []
        mse_list_test = []
        runtime_list=[]

        for split_trn, split_tst in self.kf.split(X):
            X_train, Y_train = X.iloc[split_trn], Y[split_trn]
            X_test, Y_test = X.iloc[split_tst], Y[split_tst]

            fit_runtime, train_error=model.fit(X_train, Y_train)

            ypred_test = model.predict(X_test)

            mses_train=train_error
            mses_test = [((Y_test - ypred) ** 2).sum() / len(Y_test) for ypred in ypred_test]

            mse_list_train.append(mses_train)
            mse_list_test.append(mses_test)
            runtime_list.append(fit_runtime)

        mse_train =[sum(elements) / self.folds for elements in zip(*mse_list_train)]# sum(mse_list_train) / self.folds
        mse_test = [sum(elements) / self.folds for elements in zip(*mse_list_test)]#sum(mse_list_test) / self.folds
        mean_runtime=sum(runtime_list)/ self.folds
        return mse_train, mse_test, mean_runtime


    def cross_validate_multiple_model(self, models, X, Y, parameters):
        """Performs cross validation for a regression model"""

        mses_train_per_model=[[] for _ in models]
        mses_test_per_model =[[] for _ in models]
        runtimes_per_model =[[] for _ in models]
        best_parameters_per_model=[[] for _ in models]
        for i in range(self.loops):
            self.kf = KFold(n_splits=self.folds, random_state=i, shuffle=True)
            for split_trn, split_tst in self.kf.split(X):
                X_train, Y_train = X.iloc[split_trn], Y.iloc[split_trn]
                X_test, Y_test = X.iloc[split_tst], Y.iloc[split_tst]

                parameter=[(m,X_train, Y_train,X_test, Y_test, parameters) for m in models]

                with ThreadPoolExecutor() as executor:
                    ergebnis_liste = executor.map(lambda params: eval_model(*params), parameter)
                    ergebnis_liste=list(ergebnis_liste)
                for i, erg in enumerate(ergebnis_liste):
                    mses_train_per_model[i].append(erg[0])
                    mses_test_per_model[i].append(erg[1])
                    runtimes_per_model[i].append(erg[2])
                    best_parameters_per_model[i].append(erg[3])

        mse_train_per_model=[[sum(elemente) / len(mses_per_model) for elemente in zip(*mses_per_model)] for mses_per_model in mses_train_per_model]
        mse_test_per_model=[[sum(elemente) / len(mses_per_model) for elemente in zip(*mses_per_model)] for mses_per_model in mses_test_per_model]
        mean_runtime_per_model=[np.mean(runtimes) for runtimes in runtimes_per_model]
        return mse_train_per_model, mse_test_per_model, mean_runtime_per_model,mses_train_per_model,mses_test_per_model,runtimes_per_model,best_parameters_per_model



