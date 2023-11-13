from data_loader import *
from pipeline import *

DATASET_PATH = './dataset'

if __name__ == '__main__':

    # optimizer is either RMSprop, Adagrad, Adam, SGD
    config = {
        'seed': 100,
        'dataset_path': './dataset',
        'device': 'cpu',
        'train_percentage': 0.8,
        'learning_rate': 0.03,
        'optimizer': 'RMSprop',
        'num_epochs': 15,
        'model_norm': 0.9,
        'batch_size': 32,
    }

    wandb.init(
        project="lab5",
        config=config
    )

    data_loaders_builder = CIFAR10DataLoadersBuilder(seed=config['seed'], dataset_path=DATASET_PATH,
                                    train_ratio=config['train_percentage'], device=config['device'],
                                                     batch_size=config['batch_size'])
    trainset, testset = data_loaders_builder.get_data_loaders()

    pipeline = CIFAR10Pipeline(
        data_loader_builder=data_loaders_builder,
        device=config['device'],
        config=config,
        log_file='./cifar10_log.txt'
    )

    pipeline.train()

    pipeline.evaluate()

    wandb.finish()
