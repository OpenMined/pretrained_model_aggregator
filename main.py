from pathlib import Path
import shutil
from syftbox.lib import Client
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import re
import json


# Exception name to indicate the state cannot advance
# as there are some pre-requisites that are not met
class StateNotReady(Exception):
    pass

APP_NAME = "pretrained_model_aggregator"
TEST_DATASET_NAME = "mnist_dataset.pt"
SAMPLE_TEST_DATASET_PATH = Path("./samples/test_data") / TEST_DATASET_NAME

def get_app_private_data(client: Client, app_name: str) -> Path:
    """
    Returns the private data directory of the app
    """
    return client.workspace.data_dir / "private" / app_name

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model_file(path: Path):
    model_files = []
    entries = os.listdir(path)
    pattern = r"^pretrained_mnist_label_[0-9]\.pt$"

    for entry in entries:
        if re.match(pattern, entry):
            model_files.append(entry)

    return model_files[0] if model_files else None


def evaluate_global_model(global_model: nn.Module, dataset_path: Path) -> float:
    global_model.eval()
    # load the saved mnist subset
    images, labels = torch.load(str(dataset_path), weights_only=True)
    # create a tensordataset
    dataset = TensorDataset(images, labels)
    # create a dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def aggregate_model(
    client: Client,participants: list[str], model_output_path: Path
):
    global_model = SimpleNN()
    global_model_state_dict = global_model.state_dict()

    aggregated_model_weights = {}

    n_peers = len(participants)
    aggregated_peers = []
    missing_peers = []
    for user_folder in participants:
        public_folder_path: Path = client.datasites / user_folder / "public"

        model_file = get_model_file(public_folder_path)
        if model_file is None:
            missing_peers.append(user_folder)
            continue
        model_file = public_folder_path / model_file
        aggregated_peers.append(user_folder)

        user_model_state = torch.load(str(model_file), weights_only=True)
        for key in global_model_state_dict.keys():
            # If user model has a different architecture than my global model.
            # Skip it
            if user_model_state.keys() != global_model_state_dict.keys():
                continue

            if aggregated_model_weights.get(key, None) is None:
                aggregated_model_weights[key] = user_model_state[key] * (1 / n_peers)
            else:
                aggregated_model_weights[key] += user_model_state[key] * (1 / n_peers)

    if aggregated_model_weights:
        global_model.load_state_dict(aggregated_model_weights)
        torch.save(global_model.state_dict(), str(model_output_path))
        return global_model, missing_peers
    else:
        return None, None


def advance_pretrained_aggregator(client: Client) -> None:
    """
    Iterates over the running folder and tries to advance it
    It loads in the participants.json file and aggregates the models
    """
    running_folder = client.api_data(APP_NAME) / "running"
    participants_json = running_folder / "participants.json"

    if not participants_json.is_file():
        print("participants.json file not found in the running folder")
        return
    
    with open(participants_json, "r") as f:
        participants = json.load(f)["participants"]

    model_output_path = running_folder / "global_model.pt"
    
    global_model, missing_peers = aggregate_model(
        client,
        participants,
        model_output_path
    )

    if not global_model:
        print("No models found to aggregate in participants:", participants)
        return
    
    # Evaluate the global model
    test_dataset_path = get_app_private_data(client, APP_NAME) / TEST_DATASET_NAME
    accuracy = evaluate_global_model(global_model, test_dataset_path)

    # Write the accuracy to an results.json file
    results = {
        "accuracy": accuracy,
        "participants": participants,
        "missing_peers": missing_peers
    }
    print("Accuracy Results:", results)
    with open(running_folder / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    # If no missing peers, move the global model and results.json to the done folder
    done_folder = client.api_data(APP_NAME) / "done"
    if not missing_peers:
        shutil.move(participants_json, done_folder)
        shutil.move(model_output_path, done_folder)
        shutil.move(running_folder / "results.json", done_folder)


def launch_pretrained_aggregator(client: Client) -> None:
    """
    Iterates over the launch folder and copies the participants.json file
    to the running folder

    We look for the participants.json file in the launch folder
    """
    launch_folder = client.api_data(APP_NAME) / "launch"
    running_folder = client.api_data(APP_NAME) / "running"

    participants_json = launch_folder / "participants.json"
    if participants_json.is_file():
        print("Copying participants.json to running folder")
        shutil.move(participants_json, running_folder)
    




def init_pretrained_aggregator_app(client: Client) -> None:
    """
    Creates the `pretrained_aggregator` app in the `api_data` folder
    with the following structure:
    ```
    api_data
    └── pretrained_aggregator
            └── launch
            └── running
            └── done
    ```
    """
    pretrained_aggregator = client.api_data(APP_NAME)

    for folder in ["launch", "running", "done"]:
        pretrained_aggregator_folder = pretrained_aggregator / folder
        pretrained_aggregator_folder.mkdir(parents=True, exist_ok=True)

    # Create the private data directory for the app
    # This is where the private test data will be stored
    app_pvt_dir = get_app_private_data(client, APP_NAME)
    app_pvt_dir.mkdir(parents=True, exist_ok=True)

    # Copy the test dataset to the private data directory
    test_dataset_path = app_pvt_dir / TEST_DATASET_NAME
    if not test_dataset_path.is_file():
        shutil.copy(SAMPLE_TEST_DATASET_PATH, test_dataset_path)



if __name__ == "__main__":
    client = Client.load()

    # Step 1: Init the Pretrained Aggregator App
    init_pretrained_aggregator_app(client)

    # Step 2: Launch the App
    launch_pretrained_aggregator(client)

    # Step 3: Advance the Pretrained Aggregator App
    # Iterates over the running folder and tries to advance it.
    advance_pretrained_aggregator(client)