# Imports
from PIL import Image
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
from skimage import io
import Resnet_CNN
from Resnet_CNN import ResNet50
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


import neptune.new as neptune

run = neptune.init(
    project="jbeek00/Amazone",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmU5OTRhMi1mNDE1LTQwOTgtOGQ5My0xMzBiMTM4N2UwODgifQ==",
)  # your credentials


from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches

class DeforestationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        with open(csv_file):
            self.annotations = pd.read_csv(csv_file, header=None, names=['file_name', 'loss'])
        self.root_dir = root_dir
        self.transform = transform
        self.all_images = os.listdir(self.root_dir)


    def __len__(self):
#         return len(self.annotations) #23865
       return len(os.listdir(self.root_dir))

    def __getitem__(self, index):
        name = self.all_images[index]
        image = io.imread(os.path.join(self.root_dir, name))[..., :3]

        if self.transform:
            image = self.transform(image)


        name_in_csv = 'Loss Label' + name[11:]
        y_label = torch.tensor(float(self.annotations['loss'][self.annotations['file_name'] == name_in_csv]), dtype=torch.float)

        return (image, y_label)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add batch size and num epochs to command line args
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
args = parser.parse_args()

# Hyperparameters
in_channel = 3
num_classes = 1
learning_rate = 1e-3
batch_size = args.batch_size
num_epochs = args.num_epochs
save_interval = 10

params = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
}
run["parameters"] = params

# Load Data
labels_csv = '/home/jbeek/Labels.csv'
deforestation_train_dir = ('/home/jbeek/BP/TrainSet/')
deforestation_test_dir = ('/home/jbeek/BP/TestSet/')

train_dataset = DeforestationDataset( # switch to test_dataset for test
    csv_file=labels_csv,
    root_dir=deforestation_train_dir, # switch to deforestation_test_dir
    transform=transforms.ToTensor(),
)
test_dataset = DeforestationDataset( # switch to test_dataset for test
    csv_file=labels_csv,
    root_dir=deforestation_test_dir, # switch to deforestation_test_dir
    transform=transforms.ToTensor(),
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Model
model = ResNet50(img_channels=3, num_classes=1) # -- !!
model.to(device)

# Loss and optimizer
# criterion = Regression: mean squared error loss
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Gradient Descent?

# Train Network
for epoch in range(num_epochs):
    losses = []
    test_losses = []
    train_losses = []

    if epoch % save_interval == 0:
        torch.save(model.state_dict(), f"trained_model_{batch_size}_{epoch}.pt")

    # Run training step
    for batch_idx, (data, targets) in enumerate(train_loader):

        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data).squeeze()
        print(scores.shape)
        print(targets.shape)

        loss = criterion(scores, targets)
        print(scores[0])
        print(targets[0])
        losses.append(loss.item())
        rmse = torch.sqrt(loss)
        train_losses.append(rmse.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    train_rmse = sum(train_losses)/len(train_losses)

    # Run step over test set batch to see test set performance
    with torch.no_grad():
        model.eval()
        for x, y, in test_loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x).squeeze()

            # Calculate rmse loss
            loss = criterion(scores, y)
            rmse = torch.sqrt(loss)
            test_losses.append(rmse.item())

    test_rmse = sum(test_losses)/len(test_losses)
    model.train()

    logging.info(msg=batch_size)
    run["train/epoch_loss"].log(sum(losses)/len(losses))
    run["train/train_accuracy"].log(train_rmse)
    run["train/test_accuracy"].log(test_rmse)

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

def check_accuracy_rmse(loader, model):

    model.eval()
    epoch_mean_error = 0
    counter = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)

            epoch_mean_error += torch.abs(scores - y).mean().float().item()
            counter += 1

        epoch_mean_error = epoch_mean_error / counter

    return epoch_mean_error

# model.load_state_dict(torch.load("trained_model.pt"))

model.eval() # -- !!
print("Checking accuracy on Training Set")
train_accuracy = check_accuracy_rmse(train_loader, model)
run["eval/trainset_accuracy"] = train_accuracy

print("Checking accuracy on Test Set")
test_accuracy = check_accuracy_rmse(test_loader, model)
run["eval/testset_accuracy"] = test_accuracy

torch.save(model.state_dict(), f"trained_model_{batch_size}.pt")



run.stop() # stop neptune logging
