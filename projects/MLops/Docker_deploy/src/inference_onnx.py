import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import onnxruntime
import numpy as np


def main():
    # Load CIFAR-100 test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Replace with actual CIFAR-100 normalization values
    ])

    testset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    # Load ONNX model
    onnx_model_path = '../resnet18_cifar100.onnx'
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    correct = 0
    total = 0

    for data in testloader:
        images, labels = data
        # ONNX runtime expects numpy array inputs
        images_np = images.numpy()
        # Run inference
        ort_inputs = {'modelInput': images_np}
        ort_outs = ort_session.run(None, ort_inputs)
        # Convert the output to PyTorch tensor
        outputs = torch.Tensor(ort_outs[0])
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {correct / total}')


if __name__ == '__main__':
    main()
