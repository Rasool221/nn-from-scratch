import os
import json
import math
import random
from typing import Iterable

from PIL import Image

# { 0: "0", 1: "1", ... }
output_values = {i: str(i) for i in range(10)}

HIDDEN_LAYER_COUNT = 2
HIDDEN_LAYER_SIZE = 20
INPUT_LAYER_SIZE = 784  # 28x28
OUTPUT_LAYER_SIZE = len(output_values)

LEARNING_RATE = 0.02

TRAINING_DATA_DIR = "mnist_images"
TEST_DATA_DIR = "mnist_images_test"


def sigmoid(x: float):
    """Squash any value into the 0-1 range using the classic S-curve."""
    return 1 / (1 + math.exp(-x))


def get_expected_output(expected_output_label: str) -> list[float]:
    """Convert a digit label to one-hot encoded vector for network output comparison."""
    # In a sequence 0-9, the index is just the number itself.
    index_of_label = int(expected_output_label)
    expected_output_vector: list[float] = [0] * OUTPUT_LAYER_SIZE

    if 0 <= index_of_label <= OUTPUT_LAYER_SIZE:
        expected_output_vector[index_of_label] = 1

    return expected_output_vector


def array_sum(arr):
    """Sum elements if it's a list/tuple, otherwise just return the scalar."""
    if isinstance(arr, (list, tuple)):
        total = 0
        for val in arr:
            total += val
        return total
    return arr


def dot(a, b):
    """Dot product of two vectors - multiply corresponding elements and sum."""
    if len(a) != len(b):
        raise ValueError("Arrays must have the same length for dot product")
    return array_sum([a[i] * b[i] for i in range(len(a))])


def d_sigmoid(x):
    """Derivative of sigmoid given the already-activated value. Used in backprop."""
    return x * (1 - x)


class Neuron:
    """Simple container for a neuron's activation state."""

    def __init__(self):
        """Initialize neuron with zero activation."""
        self.activation = 0


class Network:
    """Feedforward neural network with configurable hidden layers for digit classification."""

    def __init__(
        self,
        hidden_layer_count: int = HIDDEN_LAYER_COUNT,
        hidden_layer_size: int = HIDDEN_LAYER_SIZE,
        input_layer_size: int = INPUT_LAYER_SIZE,
        output_layer_size: int = OUTPUT_LAYER_SIZE,
        weights: list[list[float]] = [],
        biases: list[list[float]] = [],
        learning_rate: float = LEARNING_RATE,
    ):
        """Build the network structure and validate that weights/biases fit the architecture."""
        self.layers = []

        self.learning_rate = learning_rate

        self.hidden_layer_count = hidden_layer_count

        if len(weights) != (hidden_layer_count + 1):
            raise ValueError("The specified weights do not fit the model's layers.")

        if len(weights) != len(biases):
            raise ValueError("Custom weights and biases are not the same dimension.")

        for weights_y_index, weights_y in enumerate(weights):
            if weights_y_index == 0:
                if len(weights_y) != input_layer_size * hidden_layer_size:
                    raise ValueError(
                        "Not enough weights provided for the input to the first hidden layer."
                    )
                continue

            if weights_y_index == len(weights) - 1:
                if len(weights_y) != hidden_layer_size * output_layer_size:
                    raise ValueError(
                        "Not enough weights provided for the last hidden layer to output layer."
                    )
                continue

            if len(weights_y) != hidden_layer_size**2:
                raise ValueError(
                    f"Not enough weights provided between hidden layers indexed {weights_y_index - 1} and {weights_y_index}"
                )

        # Creating the input layer.
        self.layers.append([Neuron() for _ in range(input_layer_size)])

        # Creating the hidden layers.
        for _ in range(hidden_layer_count):
            self.layers.append([Neuron() for _ in range(hidden_layer_size)])

        # Creating the output layer.
        self.layers.append([Neuron() for _ in range(output_layer_size)])

        # Setting weights and biases.
        self.weights = weights
        self.biases = biases

    def forward(self, input: list[float]) -> list[float]:
        """Push input through the network and return output layer activations."""
        if len(input) != len(self.layers[0]):
            raise ValueError(
                "The length of the input does not match the length of the input layer."
            )

        # Setting activations of the input layer to the input.
        for i in range(len(input)):
            self.layers[0][i].activation = input[i]

        # Feed forward.
        for layer_index, layer in enumerate(self.layers[1:]):
            weights = self.weights[layer_index]
            prev_layer = self.layers[layer_index]

            for neuron_index, neuron in enumerate(layer):
                weights_for_neuron_lower_bound = int(neuron_index * len(prev_layer))
                weights_for_neuron_upper_bound = int(
                    weights_for_neuron_lower_bound + len(prev_layer)
                )

                neuron_bias = self.biases[layer_index][neuron_index]

                weights_for_neuron = [
                    w
                    for w in weights[
                        weights_for_neuron_lower_bound:weights_for_neuron_upper_bound
                    ]
                ]

                prev_layer_activations = [nueron.activation for nueron in prev_layer]

                activation = sigmoid(
                    array_sum(dot(weights_for_neuron, prev_layer_activations))
                    + neuron_bias
                )

                neuron.activation = activation

        output_layer = self.layers[-1]

        return [float(neuron.activation) for neuron in output_layer]

    def get_error(
        self,
        input: list[float],
        expected_output: list[float],
        actual_output: list[float],
    ) -> list[float]:
        """Calculate mean squared error between expected and actual outputs."""
        if not actual_output:
            actual_output = self.forward(input)

        if len(actual_output) != len(expected_output):
            raise ValueError(
                "The length of the actual output is not the same length of the expected output"
            )

        errors: list[float] = []

        for act_out_index, act_out in enumerate(actual_output):
            exp_out = expected_output[act_out_index]

            errors.append(0.5 * (pow((exp_out - act_out), 2)))

        return errors

    def print_network(self):
        """Print a summary of layer sizes, weight counts, and bias counts."""
        print("*" * 50)

        # Printing activations.
        for index, layer in enumerate(self.layers):
            print(f"Layer[{index}] has size {len(layer)}")

        # Printing weights.
        for index, weight_layer in enumerate(self.weights):
            print(f"Weights[{index}] has size {len(weight_layer)}")

        # Printing biases.
        for index, bias_layer in enumerate(self.biases):
            print(f"Biases[{index}] has size {len(bias_layer)}")

        print("*" * 50)

    def print_activations(self):
        """Dump all neuron activations for debugging purposes."""
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron in enumerate(layer):
                print(
                    f"layer_index={layer_index}, neuron_index={neuron_index}, activation={neuron.activation}"
                )

    def backward(
        self,
        expected_output: list[float],
    ) -> list[list[float]]:
        """Backpropagate error and compute updated weights using gradient descent."""
        new_weights: list[list[float]] = []

        target_output: list[float] = expected_output

        if len(target_output) != len(self.layers[-1]):
            raise ValueError(
                "The expected output is not the same length as the output layer."
            )

        output_layer_index = len(self.layers) - 1

        pd_err_total_wrt_out_arr = []
        pd_o_wrt_in_arr = []

        for layer_index, layer in reversed(list(enumerate(self.layers))):
            if layer_index == 0:
                continue

            pd_err_total_wrt_out_arr_temp = []
            pd_o_wrt_in_arr_temp = []

            prev_layer_index = layer_index - 1
            weight_updates = []

            if layer_index == output_layer_index:
                for neuron_index, neuron in enumerate(layer):
                    pd_err_total_wrt_out = (
                        neuron.activation - target_output[neuron_index]
                    )
                    pd_o_wrt_in = d_sigmoid(neuron.activation)

                    pd_err_total_wrt_out_arr_temp.append(pd_err_total_wrt_out)
                    pd_o_wrt_in_arr_temp.append(pd_o_wrt_in)

                    weights_for_neuron_lower_bound = int(
                        neuron_index * len(self.layers[prev_layer_index])
                    )
                    weights_for_neuron_upper_bound = int(
                        weights_for_neuron_lower_bound
                        + len(self.layers[prev_layer_index])
                    )

                    for weight_index, weight in enumerate(
                        self.weights[prev_layer_index][
                            weights_for_neuron_lower_bound:weights_for_neuron_upper_bound
                        ]
                    ):
                        prev_layer_neuron_for_weight = self.layers[prev_layer_index][
                            weight_index
                        ]

                        pd_in_wrt_w = prev_layer_neuron_for_weight.activation
                        pd_err_total_wrt_w = (
                            pd_in_wrt_w * pd_err_total_wrt_out * pd_o_wrt_in
                        )

                        weight_update = weight - (
                            self.learning_rate * pd_err_total_wrt_w
                        )
                        weight_updates.append(weight_update)
            else:
                next_layer_index = layer_index + 1
                for neuron_index, neuron in enumerate(layer):
                    pd_o_wrt_in = 0

                    weights_for_neuron_lower_bound = int(
                        neuron_index * len(self.layers[next_layer_index])
                    )
                    weights_for_neuron_upper_bound = int(
                        weights_for_neuron_lower_bound
                        + len(self.layers[next_layer_index])
                    )

                    pd_err_wrt_out = 0

                    for next_neuron_index, _ in enumerate(
                        self.layers[next_layer_index]
                    ):
                        pd_err_wrt_out_c = (
                            pd_err_total_wrt_out_arr[next_neuron_index]
                            * pd_o_wrt_in_arr[next_neuron_index]
                        )

                        weight_between_cur_and_next_neuron_index = (
                            next_neuron_index * len(self.layers[layer_index])
                        ) + neuron_index
                        weight_between_cur_and_next_neuron = self.weights[layer_index][
                            weight_between_cur_and_next_neuron_index
                        ]

                        pd_err_wrt_out_c *= weight_between_cur_and_next_neuron
                        pd_err_wrt_out += pd_err_wrt_out_c

                    pd_o_wrt_in = d_sigmoid(neuron.activation)

                    weights_for_neuron_lower_bound = int(
                        neuron_index * len(self.layers[prev_layer_index])
                    )
                    weights_for_neuron_upper_bound = int(
                        weights_for_neuron_lower_bound
                        + len(self.layers[prev_layer_index])
                    )

                    for weight_index, weight in enumerate(
                        self.weights[prev_layer_index][
                            weights_for_neuron_lower_bound:weights_for_neuron_upper_bound
                        ]
                    ):
                        prev_layer_neuron_for_weight = self.layers[prev_layer_index][
                            weight_index
                        ]

                        pd_in_wrt_w = prev_layer_neuron_for_weight.activation

                        pd_err_total_wrt_w = pd_in_wrt_w * pd_o_wrt_in * pd_err_wrt_out

                        weight_update = weight - (
                            self.learning_rate * pd_err_total_wrt_w
                        )
                        weight_updates.append(weight_update)

                    pd_err_total_wrt_out_arr_temp.append(pd_err_wrt_out)
                    pd_o_wrt_in_arr_temp.append(pd_o_wrt_in)

            pd_err_total_wrt_out_arr = pd_err_total_wrt_out_arr_temp
            pd_o_wrt_in_arr = pd_o_wrt_in_arr_temp

            new_weights.insert(0, weight_updates)

        if len(new_weights) != len(self.weights):
            raise ValueError(
                "The new weights length is not the same as the current weight length"
            )

        return new_weights

    def save_network(self, file_name: str = "model.json"):
        """Serialize the network's weights, biases, and config to a JSON file."""
        model_data = {}

        model_data["weights"] = []

        for weight_layer_index, weight_layer in enumerate(self.weights):
            weight_data = {}
            weight_data["weight_layer_index"] = weight_layer_index
            weight_data["weights"] = []

            for weight in weight_layer:
                weight_data["weights"].append(weight)

            model_data["weights"].append(weight_data)

        model_data["biases"] = []

        for bias_layer_index, bais_layer in enumerate(self.biases):
            bias_data = {}
            bias_data["bias_layer_index"] = bias_layer_index
            bias_data["biases"] = []

            for bias in bais_layer:
                bias_data["biases"].append(bias)

            model_data["biases"].append(bias_data)

        model_data["input_layer_size"] = len(self.layers[0])
        model_data["output_layer_size"] = len(self.layers[-1])
        model_data["hidden_layer_size"] = len(
            self.layers[1]
        )  # Same size for all hidden layers
        model_data["learning_rate"] = self.learning_rate
        model_data["hidden_layer_count"] = self.hidden_layer_count

        with open(file_name, "w") as file:
            file.write(json.dumps(model_data))


def load_network(path: str) -> Network:
    """Load a previously saved network from a JSON file."""
    if not os.path.exists(path):
        raise Exception(f"Network path {path} does not exist.")

    network_json = {}

    with open(path, "r") as file:
        network_json = json.loads(file.read())

    network_weights = []
    for layer in network_json["weights"]:
        network_weights.append(layer["weights"])

    network_biases = []
    for layer in network_json["biases"]:
        network_biases.append(layer["biases"])

    network = Network(
        hidden_layer_count=network_json["hidden_layer_count"],
        hidden_layer_size=network_json["hidden_layer_size"],
        input_layer_size=network_json["input_layer_size"],
        output_layer_size=network_json["output_layer_size"],
        weights=network_weights,
        biases=network_biases,
    )

    return network


def get_image_pixels(file_path: str) -> tuple[str, list[float]]:
    """Load an image, normalize pixels to 0-1, and extract the label from filename."""
    image = Image.open(file_path)
    image_pixel_data = image.getdata()

    if image_pixel_data is None:
        raise ValueError("Cannot parse image")

    pixels = list(image_pixel_data)
    pixels = [pixel / 255 for pixel in pixels]
    label = file_path.split("/")[-1].split("_")[1]
    label = label.removesuffix(".png")

    return (label, pixels)


def learning_rate_step_decay(
    epoch: float, initial_rate: float = LEARNING_RATE
) -> float:
    """Calculate decayed learning rate based on epoch number."""
    drop = 0.9
    epochs_drop = 1
    return initial_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))


def initialize_weights_and_biases(
    hidden_layer_count: int = HIDDEN_LAYER_COUNT,
    hidden_layer_size: int = HIDDEN_LAYER_SIZE,
    input_layer_size: int = INPUT_LAYER_SIZE,
    output_layer_size: int = OUTPUT_LAYER_SIZE,
) -> tuple[list[list[float]], list[list[float]]]:
    """Generate random weights and biases for all layers in the network."""
    weight_layers = []
    bias_layers = []

    for layer_index in range(hidden_layer_count + 1):
        weights = []
        biases = []

        # Determine the number of weights and biases for this layer
        if layer_index == 0:
            # Input layer -> first hidden layer
            amount_of_weights = hidden_layer_size * input_layer_size
            amount_of_biases = hidden_layer_size
        elif layer_index == hidden_layer_count:
            # Last hidden layer -> output layer
            amount_of_weights = hidden_layer_size * output_layer_size
            amount_of_biases = output_layer_size
        else:
            # Hidden layer -> hidden layer
            amount_of_weights = hidden_layer_size**2
            amount_of_biases = hidden_layer_size

        for _ in range(amount_of_weights):
            weights.append(random.uniform(-1, 1))

        for _ in range(amount_of_biases):
            biases.append(random.uniform(-1, 1))

        weight_layers.append(weights)
        bias_layers.append(biases)

    return weight_layers, bias_layers


def validate(validation_file_paths: list[str], network: Network) -> dict:
    """Run the network on validation data and print/return accuracy metrics."""
    correct = 0
    total = len(validation_file_paths)
    total_error = 0.0

    print(f"\n{'=' * 50}")
    print(f"Running validation on {total} samples...")
    print(f"{'=' * 50}")

    for file_path in validation_file_paths:
        label, image_pixels = get_image_pixels(file_path)
        expected_output = get_expected_output(label)

        output = network.forward(image_pixels)

        # Get predicted class (index of max activation)
        predicted_index = output.index(max(output))
        predicted_label = output_values[predicted_index]

        # Check if prediction is correct
        if predicted_label == label:
            correct += 1

        # Calculate error for this sample
        error = network.get_error(
            input=image_pixels, expected_output=expected_output, actual_output=output
        )
        total_error += sum(error)

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    avg_error = total_error / total if total > 0 else 0.0

    print(f"\nValidation Results:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Average Error: {avg_error:.6f}")
    print(f"{'=' * 50}\n")

    return {
        "accuracy": accuracy,
        "avg_error": avg_error,
        "total": total,
        "correct": correct,
    }


def get_file_paths():
    """Get file paths for training and validation data sets."""
    training_file_paths = os.listdir(TRAINING_DATA_DIR)
    validation_file_paths = os.listdir(TEST_DATA_DIR)

    training_file_paths = [os.path.join(TRAINING_DATA_DIR, path) for path in training_file_paths]
    validation_file_paths = [os.path.join(TEST_DATA_DIR, path) for path in validation_file_paths]

    return training_file_paths, validation_file_paths


def create_network(learning_rate: float = LEARNING_RATE) -> Network:
    """Create a new network with randomly initialized weights and biases."""
    weights, biases = initialize_weights_and_biases()
    return Network(
        weights=weights,
        biases=biases,
        learning_rate=learning_rate,
    )


def train_epoch(
    network: Network,
    training_file_paths: list[str],
    shuffle: bool = True,
) -> None:
    """Run one full pass through the training data, updating weights after each sample."""
    if shuffle:
        random.shuffle(training_file_paths)

    for sample_path in training_file_paths:
        label, image_pixels = get_image_pixels(sample_path)
        expected_output = get_expected_output(label)

        # Forward pass populates neuron activations
        network.forward(image_pixels)

        # Backward pass computes weight updates
        new_weights = network.backward(expected_output)
        network.weights = new_weights


def train_stochastic(
    epochs: int,
    network: Network | None = None,
    save_path: str = "model.json",
) -> Network:
    """Train for N epochs with SGD, validating after each, then save the model."""
    if network is None:
        network = create_network()

    network.print_network()

    training_file_paths, validation_file_paths = get_file_paths()

    for epoch in range(epochs):
        train_epoch(network, training_file_paths)
        print(f"epoch={epoch + 1}")
        validate(validation_file_paths, network)

    network.save_network(save_path)
    print("done - saved network")

    return network


def test_network(
    network: Network,
    data_dir: str = TRAINING_DATA_DIR,
    log_interval: int = 100,
) -> dict:
    """Evaluate network accuracy on a test dataset with progress logging."""
    test_file_paths = []
    for file in os.listdir(data_dir):
        if file.endswith(".png"):
            test_file_paths.append(os.path.join(data_dir, file))

    correct = 0
    total = len(test_file_paths)

    for index, file_path in enumerate(test_file_paths):
        label, image_pixels = get_image_pixels(file_path)

        output = network.forward(image_pixels)
        predicted_index = output.index(max(output))

        if int(predicted_index) == int(label):
            correct += 1

        if index % log_interval == 0:
            print(f"Progress: {index}/{total}, Correct so far: {correct}")

    accuracy = (correct / total) * 100 if total > 0 else 0.0

    print(f"\nTest Results:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.2f}%")

    return {"correct": correct, "total": total, "accuracy": accuracy}


if __name__ == "__main__":
    train_stochastic(epochs=2)
