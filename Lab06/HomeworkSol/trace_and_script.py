import time
import torch
from model_utils import ResNet34


# Function to measure the average inference time
def measure_inference_time(model, input_tensor, iterations=100):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Ensure gradients are not computed
        start_time = time.time()
        for _ in range(iterations):
            _ = model(input_tensor)
        end_time = time.time()
    model.train()  # Set the model back to training mode if further training will occur
    return (end_time - start_time) / iterations


# Example input tensor (using CIFAR-10 dimensions)
example_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 color channels, 32x32 image

# Measure performance of the base model
base_model = ResNet34()
base_model_time = measure_inference_time(base_model, example_input)

# Measure performance of the traced model
traced_model = torch.jit.trace(base_model, example_input)
traced_model_time = measure_inference_time(traced_model, example_input)

# Measure performance of the scripted model
scripted_model = torch.jit.script(base_model)
scripted_model_time = measure_inference_time(scripted_model, example_input)

print(f"Base Model Average Inference Time: {base_model_time:.6f} seconds")
print(f"Traced Model Average Inference Time: {traced_model_time:.6f} seconds")
print(f"Scripted Model Average Inference Time: {scripted_model_time:.6f} seconds")
