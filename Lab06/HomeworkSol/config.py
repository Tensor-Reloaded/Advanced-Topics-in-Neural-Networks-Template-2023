# Define all configurations here
config = {
    'use_pretrained': False,
    'model_name': 'ResNet34',  # Options: 'ResNet18', 'ResNet34', etc.
    'dataset_name': 'CIFAR10',  # Options: 'CIFAR10', 'CIFAR100'
    'num_epochs': 60,
    'batch_size': 128,
    'learning_rate': 0.001,
    'feature_extract': False,  # Feature extraction or fine-tuning
    'dataset_root': './data',
    'num_workers': 2,
    # Add other configurations as needed
}

