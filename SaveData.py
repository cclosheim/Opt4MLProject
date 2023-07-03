import json

def save_experiment(filename, model, train_error, test_error, training_time, mses_train, mses_test, runtimes,params):
    data = {
        'Model': model.get_name(),
        'Parameter': model.get_parameter(),
        'parameters': params,
        'Train_error': train_error,
        'All_train_error':mses_train,
        'Test_error': test_error,
        'All_test_error':mses_test,
        'Train_time': training_time,
        'All_train_time':runtimes,
        'x_values':[array.tolist() for array in model.x_final],

    }

    with open(filename, 'a') as file:
        json.dump(data, file)
        file.write('\n')


def save_experiments(filename, model, train_error, test_error, training_time, mses_train, mses_test, runtimes, parameters):
    for m,tr,ts,ti,mtr,mts,mti,param in zip(model,train_error,test_error,training_time, mses_train, mses_test, runtimes,parameters):
        save_experiment(filename,m,tr,ts,ti,mtr,mts,mti,param)


def get_experiments_data(filename):
    json_objects = []

    with open(filename, 'r') as file:
        for line in file:
            try:
                json_object = json.loads(line)
                json_objects.append(json_object)
            except json.JSONDecodeError as e:
                print("Fehler beim Decodieren der JSON-Daten:", str(e))

    return json_objects
