#include <torch/script.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <future>
#include <algorithm>


torch::Tensor normalize(torch::Tensor& input, const std::vector<double>& mean, const std::vector<double>& std) {
    for (size_t i = 0; i < mean.size(); ++i) {
        input[i] = (input[i] - mean[i]) / std[i];
    }
    return input;
}

torch::jit::script::Module loadModel(const std::string& model_path) {
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model.\n";
        exit(1);
    }
    return model;
}


std::pair<std::vector<torch::Tensor>, std::vector<uint8_t>> readDataset(const std::string& dataset_path, int batch_size) {
    std::ifstream file(dataset_path, std::ios::binary);
    if (!file) {
        std::cerr << "Unable to open file " << dataset_path << std::endl;
        exit(1);
    }

    const int num_images = 10000; 
    const int image_size = 3072; // 3x32x32

    std::vector<torch::Tensor> batches;
    std::vector<uint8_t> batch_labels;
    std::vector<uint8_t> data(image_size);

    batches.reserve(num_images / batch_size + 1);

    for (int i = 0; i < num_images; i += batch_size) {
        std::vector<torch::Tensor> image_tensors;


        for (int j = 0; j < batch_size && (i + j) < num_images; ++j) {
            file.ignore(1);

            uint8_t fine_label;
            file.read(reinterpret_cast<char*>(&fine_label), sizeof(fine_label));
            batch_labels.push_back(fine_label);

            file.read(reinterpret_cast<char*>(data.data()), image_size);

            torch::Tensor image_tensor = torch::from_blob(data.data(), {3, 32, 32}, torch::kUInt8).clone();
            image_tensor = image_tensor.to(torch::kFloat).div(255.0); // Convert to float and normalize to 0-1

            image_tensor = normalize(image_tensor, {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
            image_tensors.push_back(image_tensor);
        }

        torch::Tensor batch = torch::stack(image_tensors);
        batches.push_back(batch);
    }

    file.close();

    return {batches, batch_labels};
}


std::pair<int, int> processBatch(const torch::Tensor& batch, const std::vector<uint8_t>& batch_labels, torch::jit::script::Module& model, int start_index, int end_index) {
    torch::Tensor output = model.forward({batch}).toTensor();
    auto predicted = output.argmax(1);

    int correct = 0;
    for (int i = start_index; i < end_index; ++i) {
        if (predicted[i - start_index].item<int>() == static_cast<int>(batch_labels[i])) {
            correct++;
        }
    }

    return {correct, end_index - start_index};
}

double performInference(const std::vector<torch::Tensor>& batches, const std::vector<uint8_t>& batch_labels, torch::jit::script::Module& model, int batch_size) {
    const size_t max_workers = 2;
    std::vector<std::future<std::pair<int, int>>> futures;

    int total_correct = 0;
    int total_count = 0;

    for (size_t batch_index = 0; batch_index < batches.size(); ++batch_index) {
        if (futures.size() >= max_workers) {
            // Wait for the earliest future to complete
            auto done = futures.begin();
            auto result = done->get();
            total_correct += result.first;
            total_count += result.second;
            futures.erase(done);
        }

        const auto& batch = batches[batch_index];
        auto start_index = batch_index * batch_size;
        auto end_index = std::min(static_cast<size_t>(start_index + batch_size), batch_labels.size());

        futures.push_back(std::async(std::launch::async, processBatch, std::ref(batch), std::ref(batch_labels), std::ref(model), start_index, end_index));
    }

    // Wait for the remaining futures
    for (auto& future : futures) {
        auto result = future.get();
        total_correct += result.first;
        total_count += result.second;
    }

    return static_cast<double>(total_correct) / total_count;
}



double performInferenceSingle(const std::vector<torch::Tensor>& batches, const std::vector<uint8_t>& batch_labels, torch::jit::script::Module& model, int batch_size) {

    int correct = 0;
    int total = 0;

    for (size_t batch_index = 0; batch_index < batches.size(); ++batch_index) {
        const auto& batch = batches[batch_index];
        torch::Tensor output = model.forward({batch}).toTensor();
        auto predicted = output.argmax(1); 

        auto start_index = batch_index * batch_size;
        auto end_index = std::min(static_cast<size_t>(start_index + batch_size), batch_labels.size());

        for (int i = start_index; i < end_index; ++i) {
            total++;
            if (predicted[i - start_index].item<int>() == static_cast<int>(batch_labels[i])) {
                correct++;
            }
        }
    }

    return static_cast<double>(correct) / total;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <path-to-model> <path-to-dataset> <batch-size>" << std::endl;
        return 1;
    }

    // time start
    auto start = std::chrono::high_resolution_clock::now();

    std::string model_path = argv[1];
    std::string dataset_path = argv[2];
    int batch_size = std::stoi(argv[3]);

    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Dataset path: " << dataset_path << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;

    torch::jit::script::Module model = loadModel(model_path);
    auto [batches, batch_labels] = readDataset(dataset_path, batch_size);
    double accuracy = performInferenceSingle(batches, batch_labels, model, batch_size);

    // time end
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Accuracy (cpp): " << accuracy << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    return 0;
}
