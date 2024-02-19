#!/bin/bash

value="$MODEL_TYPE"

PYTORCH_MODEL_PATH=$(realpath ./resnet18_cifar100_final.pth)
TRACED_MODEL_OUTPUT_PATH=$(realpath ./cpp/example-app/traced_model.pt)
CIFAR100_BINARY_TEST_DATASET_PATH=$(realpath ./data/cifar-100-binary/test.bin)
BATCH_SIZE=256

measure_resources() {
    /usr/bin/time -v "$@"
}

if [[ $value == "torch_py" ]]; then
    echo "Running torch_py:"

    cd ./src

    # Measure resources for Python script
    measure_resources python3.11 inference.py pytorch

    cd ..

elif [[ $value == "torch_cpp" ]]; then
    echo "Running torch_cpp"

    python3.11 ./src/convert_to_torchscript.py $PYTORCH_MODEL_PATH $TRACED_MODEL_OUTPUT_PATH

    echo "Traced model saved to $TRACED_MODEL_OUTPUT_PATH"

    cd ./cpp/example-app/build

    # Measure resources for C++ executable
    measure_resources ./example-app $TRACED_MODEL_OUTPUT_PATH $CIFAR100_BINARY_TEST_DATASET_PATH $BATCH_SIZE

    cd ../../..
    
elif [[ $value == "onnx_py" ]]; then
    echo "Running onnx_py"

    cd ./src

    # Measure resources for Python script
    measure_resources python3.11 inference.py onnx

else
    echo "Invalid argument"
fi
