import os
import sys
import time
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
import onnxruntime
import numpy as np
import psutil

def load_data(base_path=""):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    testset = datasets.CIFAR100(root=f'{base_path}../data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
    return testloader


def pytorch_inference(testloader, base_path=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(torch.load(f'{base_path}../resnet18_cifar100_final.pth', map_location=device))
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    peak_mem = 0
    peak_cpu = 0
    
    current_process = psutil.Process()
    interval = 0.1
    time_now = time.time()
    
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if time.time() - time_now > interval:
                peak_mem = max(peak_mem, current_process.memory_info().rss / 1024 ** 2)
                peak_cpu = max(peak_cpu, current_process.cpu_percent(interval))
                time_now = time.time()
    print(f'Accuracy (PyTorch): {correct / total}')
    print(f'Peak memory usage is {peak_mem} MB')
    print(f'Peak CPU utilization is {peak_cpu} %')

def onnx_inference(testloader, base_path=''):
    onnx_model_path = f'{base_path}../resnet18_cifar100.onnx'
    
    session_options = onnxruntime.SessionOptions()
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = onnxruntime.InferenceSession(onnx_model_path, session_options)

    correct = 0
    total = 0
    
        
    peak_mem = 0
    peak_cpu = 0
    
    current_process = psutil.Process()
    interval = 0.1
    time_now = time.time()
    
    for data in testloader:
        images, labels = data
        images_np = images.numpy()
        ort_inputs = {'modelInput': images_np}
        ort_outs = ort_session.run(None, ort_inputs)
        outputs = torch.Tensor(ort_outs[0])
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if time.time() - time_now > interval:
            peak_mem = max(peak_mem, current_process.memory_info().rss / 1024 ** 2)
            peak_cpu = max(peak_cpu, current_process.cpu_percent(interval))
            time_now = time.time()

    accuracy = correct / total
    print(f'Accuracy (ONNX): {accuracy:.2f}')
    print(f'Peak memory usage is {peak_mem} MB')
    print(f'Peak CPU utilization is {peak_cpu} %')

    return accuracy



def do_inference(model_type, base_path=''):
    testloader = load_data(base_path)
    print(f"Running inference with {model_type} model")
    
    
    if model_type == 'pytorch':
        pytorch_inference(testloader, base_path)
    elif model_type == 'onnx':
        onnx_inference(testloader, base_path)
    else:
        print("Invalid model type specified in environment variable.")

def main():
    start_time = time.time()
    
    # get from args the model type
    model_type = os.environ.get('MODEL_TYPE', 'pytorch')
    
    if len(sys.argv) > 1:
        model_type = sys.argv[1]

    do_inference(model_type)
        
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
