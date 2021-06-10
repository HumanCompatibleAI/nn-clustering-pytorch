import csv
import json
import os
import os.path

from shuffle_and_cluster import shuffle_and_clust

# basically:
# - look at everything in models, get its ncut and rel clust (loop based on
#   training details)
# - find each models' corresponding directory, get pre- and post-pruning
#   accuracy

model_dir = "./models/"
sacred_dir = "./training_runs/"
num_shuffles = 30

results_path = "results.csv"

for dataset in ['cifar10', 'mnist', 'kmnist']:
    model_types = ['mlp', 'cnn'] if dataset != 'cifar10' else ['cnn']
    for model_type in model_types:
        for cg_lambda in [0, 1e-3, 1e-2]:
            for pruning in [False, True]:
                results = []
                for i in range(5):
                    file_name = model_type + "_" + dataset + "_"
                    if cg_lambda == 0:
                        file_name += str(i)
                    elif cg_lambda == 1e-3:
                        file_name += "clust-grad_" + str(i)
                    else:
                        file_name += "clust-grad_" + str(i) + "_lambda=0.01"
                    if not pruning:
                        file_name += "_unpruned"
                    file_name += ".pth"
                    r = shuffle_and_clust.run(
                        config_updates={
                            'weights_path': model_dir + file_name,
                            'num_samples': num_shuffles,
                            'net_type': model_type
                        })
                    clust_dict = r.result
                    for j in range(101, -1, -1):
                        maybe_file = sacred_dir + str(j) + "/" + file_name
                        if os.path.exists(maybe_file):
                            break
                    else:
                        print("oh no, this file never got recorded in sacred\
                        somehow")
                    net_dir = sacred_dir + str(j) + "/"
                    with open(net_dir + "config.json") as f:
                        config = json.load(f)
                    assert config["net_type"] == model_type
                    assert config["dataset"] == dataset
                    if cg_lambda > 0:
                        assert config["cluster_gradient"]
                        assert (config["cluster_gradient_config"]["lambda"] ==
                                cg_lambda)
                    if cg_lambda == 0:
                        assert not config["cluster_gradient"]
                    num_epochs = config["num_epochs"]
                    num_pruning = config["pruning_config"][
                        "num_pruning_epochs"]
                    pruning_start = num_epochs - num_pruning
                    with open(net_dir + "metrics.json") as f:
                        metrics = json.load(f)
                    read_index = (num_epochs -
                                  1 if pruning else pruning_start - 1)
                    test_acc = metrics['test.accuracy']['values'][read_index]
                    test_loss = metrics['test.loss']['values'][read_index]
                    with open(results_path, 'w', newline='') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerow([
                            dataset, model_type, cg_lambda, pruning,
                            clust_dict['true n-cut'], clust_dict['mean'],
                            clust_dict['stdev'], clust_dict['percentile'],
                            clust_dict['z-score'], test_acc, test_loss
                        ])
