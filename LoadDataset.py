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

from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches

class DeforestationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
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

# Hyperparameters
in_channel = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 1
num_epochs = 10

# Load Data
labels_csv = open('/home/jbeek/Labels.csv')
deforestation_train_dir = ('/home/jbeek/BP/Mini_Train/')
deforestation_test_dir = ('/home/jbeek/BP/TestSet/')
train_dataset = DeforestationDataset( # switch to test_dataset for test
    csv_file=labels_csv,
    root_dir=deforestation_train_dir, # switch to deforestation_test_dir
    transform=transforms.ToTensor(),
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


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

    for batch_idx, (data, targets) in enumerate(train_loader):

        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        print(scores.shape)
        print(targets.shape)
        loss = criterion(scores, targets)
        print(scores[0])
        print(targets[0])
        losses.append(loss.item())



        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    logging.info(msg=batch_size)
    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

model.eval() # -- !!
print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)
'''
print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)
'''
