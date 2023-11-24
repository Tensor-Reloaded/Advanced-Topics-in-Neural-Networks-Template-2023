from data_loader import CIFAR10DataLoadersBuilder
from pipeline import CIFAR10Pipeline
import wandb
import torch

def grid_search():
    learning_rates = [0.001, 0.01, 0.1]
    optimizers = ['Adam', 'SGD', 'RMSPROP', 'Adagrad']
    batch_sizes = [32, 64, 128]


    for learning_rate in learning_rates:
        for optimizer in optimizers:
            for batch_size in batch_sizes:
                config = {
                    'seed': 100,
                    'dataset_path': './dataset',
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'train_percentage': 0.8,
                    'learning_rate': learning_rate,
                    'optimizer': optimizer,
                    'num_epochs': 50,
                    'model_norm': 0.9,
                    'batch_size': batch_size,
                    'scripted': True
                }

                # Initialize WandB
                wandb.init(project="lab7assignment6", config=config)

                # Build data loaders
                data_loaders_builder = CIFAR10DataLoadersBuilder(
                    seed=config['seed'],
                    dataset_path=config['dataset_path'],
                    train_ratio=config['train_percentage'],
                    device=config['device'],
                    batch_size=config['batch_size']
                )

                pipeline = CIFAR10Pipeline(
                    data_loader_builder=data_loaders_builder,
                    device=config['device'],
                    config=config,
                    log_file='./cifar10_log.txt'
                )

                pipeline.train()
                [test_loss, test_accuracy] = pipeline.evaluate()

                wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy})

                wandb.finish()


def compare_scripted_traced_and_base_model():

    # Create configuration
    base_config = {
        'seed': 100,
        'dataset_path': './dataset',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'train_percentage': 0.8,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'num_epochs': 5,
        'model_norm': 0.9,
        'batch_size': 32,
        'scripted': False
    }

    data_loaders_builder = CIFAR10DataLoadersBuilder(
        seed=base_config['seed'],
        dataset_path=base_config['dataset_path'],
        train_ratio=base_config['train_percentage'],
        device=base_config['device'],
        batch_size=base_config['batch_size']
    )

    scripted_config = base_config.copy()
    scripted_config['scripted'] = True


    # Create pipeline
    pipeline_base = CIFAR10Pipeline(
        data_loader_builder=data_loaders_builder,
        device=base_config['device'],
        config=base_config,
        log_file='./cifar10_log.txt'
    )


    pipeline_scripted = CIFAR10Pipeline(
        data_loader_builder=data_loaders_builder,
        device=base_config['device'],
        config=scripted_config,
        log_file='./cifar10_log.txt'
    )



    wandb.init(project="lab7assignment6_comparison", config=base_config)
    base_elapsed = get_time(pipeline_base.train)

    wandb.finish()

    wandb.init(project="lab7assignment6_comparison", config=scripted_config)
    scripted_elapsed = get_time(pipeline_scripted.train)
    wandb.finish()

    print(f"Base model took {base_elapsed} seconds to train")
    print(f"Scripted model took {scripted_elapsed} seconds to train")



def get_time(func):
    import time
    start = time.time()
    func()
    end = time.time()
    return end-start

if __name__ == '__main__':
    # grid_search()
    compare_scripted_traced_and_base_model()