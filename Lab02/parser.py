import torch


class InputParser:
    def __init__(self, filename=None):
        self.filename = filename

    def parse(self):
        lines = []
        y_true = []

        line_index = -1
        with open(self.filename) as fd:
            while True:
                line = fd.readline().strip()
                if not line or line == "":
                    break

                line_index += 1
                if line_index == 0:
                    continue

                data = [float(value) for value in line.split(",")]
                output = [float(0) if index != data[0] else float(1) for index in range(10)]
                y_true.append(output)
                lines.append(data[1:])

        tensors_list_input = [torch.tensor(line) for line in lines]
        tensor_of_tensors_input = torch.stack(tensors_list_input, dim=0)

        tensors_list_output = [torch.tensor(line) for line in y_true]
        tensor_of_tensors_output = torch.stack(tensors_list_output, dim=0)

        return tensor_of_tensors_input, tensor_of_tensors_output
