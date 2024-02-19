import torch
import argparse
from torchvision import models


def trace_model(model_path, output_path):
    # Load the model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 100)

    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set the model to evaluation mode
    model.eval()
    
    # Create a dummy input
    dummy_input = torch.randn(1, 3, 32, 32)

    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save the traced model
    traced_model.save(output_path)
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trace a model")
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("output_path", type=str, help="Path to save the traced model")
    args = parser.parse_args()

    # Trace the model and save it
    trace_model(args.model_path, args.output_path)