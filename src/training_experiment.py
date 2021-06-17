from train_model import train_exp

num_runs = 5
reg_lambda_range = [1e-3, 1e-2, 1e-1]


def get_with_word(clust_grad):
    return " with " if clust_grad else " without "


for dataset in ['mnist', 'kmnist']:
    for model_type in ['mlp', 'cnn']:
        for clust_grad in [False, True]:
            with_word = get_with_word(clust_grad)
            reg_lambda_range_ = reg_lambda_range if clust_grad else [0]
            for reg_lambda in reg_lambda_range_:
                lambda_text = f" (lambda = {reg_lambda})" if clust_grad else ""
                for i in range(num_runs):
                    my_string = (
                        f"\nTraining {model_type} number {i} on {dataset}" +
                        with_word + "clust grad" + lambda_text + ":\n")
                    print(my_string)
                    model_name = str(i) + clust_grad * f"_lambda_{reg_lambda}"
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
    with_word = get_with_word(clust_grad)
    reg_lambda_range_ = reg_lambda_range if clust_grad else [0]
    for reg_lambda in reg_lambda_range_:
        lambda_text = f" (lambda = {reg_lambda})" if clust_grad else ""
        for i in range(num_runs):
            my_string = (f"\nTraining 6-layer CNN number {i} on CIFAR-10" +
                         with_word + "clust grad" + lambda_text + ":\n")
            print(my_string)
            model_name = str(i) + clust_grad * f"_lambda={reg_lambda}"
            train_exp.run(
                config_updates={
                    'dataset': 'cifar10',
                    'cluster_gradient': clust_grad,
                    'net_type': 'cnn',
                    'net_choice': 'cifar10_6',
                    'num_epochs': 50,
                    'cluster_gradient_config': {
                        'lambda': reg_lambda
                    },
                    'pruning_config': {
                        'num_pruning_epochs': 10
                    },
                    'training_run_string': model_name
                })

for clust_grad in [False, True]:
    with_word = get_with_word(clust_grad)
    reg_lambda_range_ = reg_lambda_range if clust_grad else [0]
    for reg_lambda in reg_lambda_range_:
        lambda_text = f" (lambda = {reg_lambda})" if clust_grad else ""
        my_string = ("\nTraining VGG-11 on CIFAR-10" + with_word +
                     "clust grad" + lambda_text + ":\n")
        print(my_string)
        model_name = "vgg11" + clust_grad * f"_lambda={reg_lambda}"
        train_exp.run(
            config_updates={
                'dataset': 'cifar10',
                'net_type': 'cnn',
                'net_choice': 'cifar10_vgg11',
                'num_epochs': 300,
                'pruning_config': {
                    'num_pruning_epochs': 100
                },
                'optim_func': 'sgd',
                'optim_kwargs': {
                    'lr': 0.5,
                    'momentum': 0.9,
                    'weight_decay': 5e-4
                },
                'decay_lr': True,
                'decay_lr_factor': 0.5,
                'decay_lr_epochs': 30
            })
