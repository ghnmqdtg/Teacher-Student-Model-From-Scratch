import wandb
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import default_collate
import torchvision
import torchvision.transforms as transforms


def prepare_dataset(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=48,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=48,
        pin_memory=True
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 3 input channel, 6 output channel, 5x5 square convolution
        # nn.Conv2d args: (in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        # 6 input channel, 16 output channel, 5x5 square convolution
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layer 1
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Fully connected layer 2
        self.fc2 = nn.Linear(120, 84)
        # Fully connected layer 3
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten args: (input, start_dim=0, end_dim=-1)
        # Set start_dim=1 so it will flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model_path, train_loader, test_loader):
    print(f'\nStart training on {device}')
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    wandb.watch(model, criterion, log="all")
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            # Get the inputs
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

            # print every 2000 mini-batches
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                wandb.log({'epoch': epoch+1, 'loss': running_loss/2000})
                running_loss = 0.0

        running_loss = 0.0

    torch.save(model.state_dict(), model_path)
    print(f'Finished Training, model saved to {model_path}')

    return model


def predict(model, batch_size):
    print(f'\nStarting prediction using {device}')
    # Get some random testing images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    labels = labels.to(device)
    # Save the grid of images
    img_grid = torchvision.utils.make_grid(images)
    with open('./imgs/test_grid.png', 'wb') as f:
        torchvision.utils.save_image(img_grid, f)
    print('Groundtruth: ', ' '.join(
        f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(batch_size)))


def eval(model):
    print(f'\nStarting evaluation using {device}')
    # Prepare to count predictions for each class
    correct_predictions_by_class = {classname: 0 for classname in classes}
    total_predictions_by_class = {classname: 0 for classname in classes}

    # No gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_predictions_by_class[classes[label]] += 1
                total_predictions_by_class[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_predictions_by_class.items():
        accuracy = 100 * float(correct_count) / \
            total_predictions_by_class[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == '__main__':
    # Start a new run
    wandb.init(project='CIFAR_Net',
               entity='ghnmqdtg',
               name=f'CIFAR-Net-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

    batch_size = 4
    model_path = './cifar_net.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_loader, test_loader, classes = prepare_dataset(batch_size=batch_size)
    model = train(model_path=model_path,
                  train_loader=train_loader, test_loader=test_loader)
    predict(model, batch_size)
    eval(model)
