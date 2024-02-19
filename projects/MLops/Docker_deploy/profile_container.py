import os
import subprocess
import re

from src.inference import do_inference
import csv


def run_command(command, cwd=None):
    return subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)


def benchmark_local_function(command, base_path):
    
    result = run_command(command, base_path)
    local_log = result.stdout.strip() + "\r\n" + result.stderr.strip()

    return extract_process_info(local_log)

def extract_info(log_content):
    time_regex = r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:]+)\.(\d+)'
    max_resident_size_regex = r'Maximum resident set size \(kbytes\): (\d+)'
    cpu_percent_regex = r'Percent of CPU this job got: (\d+)%'
    accuracy_pattern = r"Accuracy.*:\s*([0-9.]+)"

    # Extracting and converting the total execution time to milliseconds
    time_match = re.search(time_regex, log_content)
    if time_match:
        time_parts = time_match.group(1).split(':')
        msec_part = int(time_match.group(2))
        
        if len(time_parts) == 3:  # Format is h:mm:ss
            hours, minutes, seconds = map(int, time_parts)
        elif len(time_parts) == 2:  # Format is m:ss
            hours = 0
            minutes, seconds = map(int, time_parts)
        else:
            hours = minutes = seconds = msec_part = 0

        total_time_msec = round((hours * 3600 + minutes * 60 + seconds) * 1000 + msec_part, 1)
    else:
        total_time_msec = None

    # Extracting and converting the maximum resident set size to megabytes
    max_resident_size_match = re.search(max_resident_size_regex, log_content)
    if max_resident_size_match:
        max_resident_size_kb = int(max_resident_size_match.group(1))
        max_resident_size_mb = round(max_resident_size_kb / 1024, 1)
    else:
        max_resident_size_mb = None

    cpu_percent_match = re.search(cpu_percent_regex, log_content)
    if cpu_percent_match:
        cpu_percent = float(cpu_percent_match.group(1))
        total_cpu_cores = os.cpu_count()
        adjusted_cpu_percent = round((cpu_percent / 100) / total_cpu_cores * 100, 1)
    else:
        adjusted_cpu_percent = None
        
    accuracy_match = re.search(accuracy_pattern, log_content)
    accuracy = round(float(accuracy_match.group(1)), 2) if accuracy_match else None
    
    print(f"[Container]Accuracy: {accuracy}, Execution Time (ms): {total_time_msec}, Memory (MB): {max_resident_size_mb}, Total CPU Usage (%): {adjusted_cpu_percent}")

    return accuracy, total_time_msec, max_resident_size_mb, adjusted_cpu_percent


def extract_process_info(log_content):
    # Regular expression patterns for extracting the required data
    accuracy_pattern = r"Accuracy.*:\s*([0-9.]+)"
    memory_usage_pattern = r"Peak memory usage is\s*([0-9.]+)"
    cpu_utilization_pattern = r"Peak CPU utilization is\s*([0-9.]+)"
    elapsed_time_pattern = r"Elapsed time:\s*([0-9.]+)"

    # Extracting data using regex
    accuracy_match = re.search(accuracy_pattern, log_content)
    memory_usage_match = re.search(memory_usage_pattern, log_content)
    cpu_utilization_match = re.search(cpu_utilization_pattern, log_content)
    elapsed_time_match = re.search(elapsed_time_pattern, log_content)

    # Extracting values as strings and then converting to floats
    accuracy = round(float(accuracy_match.group(1)), 2) if accuracy_match else None
    memory_usage = round(float(memory_usage_match.group(1)), 1) if memory_usage_match else None

    # Adjusting and rounding CPU utilization based on total CPU cores
    cpu_utilization = None
    if cpu_utilization_match:
        cpu_percent = float(cpu_utilization_match.group(1))
        total_cpu_cores = os.cpu_count()
        cpu_utilization = round((cpu_percent / 100) / total_cpu_cores * 100, 1)

    elapsed_time = round(float(elapsed_time_match.group(1)) * 1000, 1) if elapsed_time_match else None  # Convert to milliseconds and round
    
    print(f"[Process]Accuracy: {accuracy}, Peak Memory Usage (MB): {memory_usage}, Peak CPU Utilization (%): {cpu_utilization}, Elapsed Time (ms): {elapsed_time}")

    

    return accuracy, memory_usage, cpu_utilization, elapsed_time

def run_docker_container_and_extract_logs(model_type, image_name="inference_before:latest"):
    # Run the Docker container
    container_id = run_command(f"docker run -d -e MODEL_TYPE={model_type} {image_name}").stdout.strip()

    # Wait for the container to finish
    run_command(f"docker wait {container_id}")

    # Grab the logs
    result = run_command(f"docker logs {container_id}")
    container_log = result.stdout.strip() + "\r\n" + result.stderr.strip()

    print(f"Container Log: {container_log}")

    # Regular expression pattern to match each /usr/bin/time output
    pattern = r'(?smi)Command being timed: ".+?Exit status: \d+'

    # Extracting the outputs into a list
    commands = re.findall(pattern, container_log)
    
    benchmark_results = dict()

    # Process each command section
    for command in commands:
        accuracy, execution_time_msec, memory_mb, cpu_usage_percent = extract_info(command)
        
        accuracy, memory_usage, cpu_utilization, elapsed_time = extract_process_info(container_log)
        
        benchmark_results['container'] = (accuracy, execution_time_msec, memory_mb, cpu_usage_percent)
        benchmark_results['process'] = (accuracy, elapsed_time, memory_usage, cpu_utilization)
        

    # Remove the container
    run_command(f"docker rm {container_id}")
    
    return benchmark_results


def calculate_average(benchmark_results):
    accuracy = [result[0] for result in benchmark_results if result[0] is not None]
    execution_times = [result[1] for result in benchmark_results if result[1] is not None]
    memory_usages = [result[2] for result in benchmark_results if result[2] is not None]
    cpu_usages = [result[3] for result in benchmark_results if result[3] is not None]
    
    if len(accuracy) == 0:
        average_accuracy = None
    else:
        average_accuracy = sum(accuracy) / len(accuracy)
    
    if len(execution_times) == 0:
        average_execution_time = None
    else:
        average_execution_time = sum(execution_times) / len(execution_times)
    
    if len(memory_usages) == 0:
        average_memory_usage = None
    else:
        average_memory_usage = sum(memory_usages) / len(memory_usages)
    
    if len(cpu_usages) == 0:
        average_cpu_usage = None
    else:
        average_cpu_usage = sum(cpu_usages) / len(cpu_usages)
    
    # print(f"Average Execution Time (ms): {average_execution_time}, Average Memory (MB): {average_memory_usage}, Average CPU Usage (%): {average_cpu_usage}")
    
    return average_accuracy, average_execution_time, average_memory_usage, average_cpu_usage

def write_to_csv(columns, entries, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(entries)

if __name__ == '__main__':
    docker_model_types = {"torch_cpp", "torch_py", "onnx_py"}
    local_model_types = {"pytorch", "onnx"}

    container_results = []
    local_results = []

    N = 30

    for model_type in docker_model_types:
        model_results_process = []
        model_results_container = []
        
        
        for i in range(N):
            benchmark_result = run_docker_container_and_extract_logs(model_type)
            
            model_results_process.append(benchmark_result['process'])
            model_results_container.append(benchmark_result['container'])
        
        p_accuracy, p_average_execution_time, p_average_memory_usage, p_average_cpu_usage = calculate_average(model_results_process)
        _, c_average_execution_time, c_average_memory_usage, c_average_cpu_usage = calculate_average(model_results_container)
        
        # append to csv model type, execution time, memory, cpu usage
        container_results.append([model_type, p_accuracy, p_average_execution_time, p_average_memory_usage, p_average_cpu_usage, c_average_execution_time, c_average_memory_usage, c_average_cpu_usage])
    
    
    write_to_csv(["Model Type", "Accuracy", "Average Execution Time (ms) (Process)", "Peak Memory (MB) (Process)", "Average CPU Usage (%) (Process)", "Average Execution Time (ms) (Container)", "Peak Memory (MB) (Container)", "Average CPU Usage (%) (Container)"], \
                 container_results, "./benchmark_container_results.csv")
    
    for model_type in local_model_types:
        local_model_results_process = []
        
        for i in range(N):
            accuracy, memory_usage, cpu_utilization, elapsed_time = benchmark_local_function(f"python inference.py {model_type}", "./src")
            
            local_model_results_process.append((accuracy, elapsed_time, memory_usage, cpu_utilization))
        
        average_accuracy, average_execution_time, average_memory_usage, average_cpu_usage = calculate_average(local_model_results_process)
        
        local_results.append([model_type, average_accuracy, average_execution_time, average_memory_usage, average_cpu_usage])
    
    write_to_csv(["Model Type", "Accuracy", "Average Execution Time (ms)", "Peak Memory (MB)", "Average CPU Usage (%)"], local_results, "./benchmark_local_results.csv")
        
