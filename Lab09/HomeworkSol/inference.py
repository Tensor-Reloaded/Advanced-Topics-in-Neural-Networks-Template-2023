import torch
from model import TransformationModel
from utils import test_inference_time, ground_truth_transform, generate_and_save_images, infer_and_compare
from data import CustomDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = TransformationModel()
    model.load_state_dict(torch.load('model_weights.pth'))
    model.to(device)

    # Load CIFAR10 test dataset
    test_dataset = CustomDataset(data_path='./data', train=False, transform=ground_truth_transform)

    # Benchmark with different batch sizes
    batch_sizes = [10, 50, 100, 200]
    for batch_size in batch_sizes:
        print(f"\nBenchmarking with batch size: {batch_size}")
        sequential_time, model_batch_time = test_inference_time(model, device, batch_size)
        print(f"Sequential transforming time: {sequential_time:.4f} seconds")
        print(f"Model batch processing time: {model_batch_time:.4f} seconds")

    # Generate and save images
    print("Generating and saving images...")
    generate_and_save_images(model, test_dataset, num_images=10, device=device)

    # Evaluate model's accuracy
    print("Evaluating model accuracy...")
    model_accuracy = infer_and_compare(model, test_dataset, device)
    print(f"Model accuracy (MSE against ground truth): {model_accuracy[0]:.6f}")


if __name__ == '__main__':
    main()
