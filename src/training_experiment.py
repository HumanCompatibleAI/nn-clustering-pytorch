from train_model import train_exp

num_runs = 5
reg_lambda = 1e-2

for dataset in ['mnist', 'kmnist']:
    for model_type in ['mlp', 'cnn']:
        for clust_grad in [False, True]:
            for i in range(num_runs):
                with_word = " with " if clust_grad else " without "
                my_string = (
                    f"\nTraining {model_type} number {i} on {dataset}" +
                    with_word + "clust grad:\n")
                print(my_string)
                model_name = str(i) + clust_grad * f"_lambda={reg_lambda}"
                train_exp.run(
                    config_updates={
                        'dataset': dataset,
                        'cluster_gradient': clust_grad,
                        'cluster_gradient_config': {
                            'lambda': reg_lambda
                        },
                        'net_type': model_type,
                        'training_run_string': model_name
                    })

for clust_grad in [False, True]:
    with_word = " with " if clust_grad else " without "
    for i in range(num_runs):
        my_string = (f"\nTraining CNN number {i} on CIFAR-10" + with_word +
                     "clust grad:\n")
        print(my_string)
        model_name = str(i) + clust_grad * f"_lambda={reg_lambda}"
        train_exp.run(
            config_updates={
                'dataset': 'cifar10',
                'cluster_gradient': clust_grad,
                'net_type': 'cnn',
                'net_choice': 'cifar10',
                'num_epochs': 50,
                'cluster_gradient_config': {
                    'num_pruning_epochs': 10,
                    'lambda': reg_lambda
                },
                'training_run_string': model_name
            })
