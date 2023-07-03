from SaveData import get_experiments_data
import matplotlib.pyplot as plt

def plot_mse_graph(mses,modelnames,name, skip_first=0):
    length=max(len(lst) for lst in mses)
    iters = range(length)  # Annahme: Alle Listen haben dieselbe Länge

    fig=plt.figure()
    if skip_first > 0:
        iters=iters[skip_first:]
    for mse, model in zip(mses, modelnames):
        if skip_first > 0:
            mse = mse[skip_first:]
        plt.plot(iters, mse, label=model)
    plt.ylim(120000, 240000)
    #plt.ylim(0, 7)
    #plt.xlim(0, 600)
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    fig.savefig(name, dpi=250, bbox_inches="tight")


data=get_experiments_data('Experiments/exp_bike_final.json')

names=[]
mses_train=[]
mses_test=[]
times=[]
for d in data:
    names.append(d['Model'])
    mses_train.append(d['Train_error'])
    mses_test.append(d['Test_error'])
    times.append(d['All_train_time'])

plot_mse_graph(mses_train,names,'bike_train_32',0)
plot_mse_graph(mses_test,names,'bike_test_32',0)
for n, tr,te,t in zip(names,mses_train,mses_test,times):
    print(n, t)