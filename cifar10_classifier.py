import wandb
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def prepare_dataset(batch_size):
    """
    Prepare CIFAR10 dataset

    Input:
        batch_size: batch size for training and testing
    Output:
        train_loader: training data loader
        test_loader: testing data loader
        classes: class names
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

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
        num_workers=1,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


def load_model(model_path=None):
    """
    Load pre-trained ResNet50 model

    Input:
        model_path: path to the model weights
    Output:
        model: ResNet50 model
    """
    # Load pre-trained ResNet50 model
    model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
    # Modify the last layer to fit CIFAR10 dataset, since it was trained on ImageNet (1000 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    return model


def train(model_path, train_loader, test_loader):
    """
    Train the model

    Input:
        model_path: path to save the model weights
        train_loader: training data loader
        test_loader: testing data loader
    Output:
        model: trained model
    """
    print(f'\nStart training on {device}')
    # Load pre-trained ResNet50 model
    model = load_model()
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Watch model in wandb
    wandb.watch(model, criterion, log="all")
    # Set number of epochs
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in pbar:
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
            # Update tqdm progress bar
            pbar.set_description(
                f'Epoch {epoch+1}/{num_epochs}, loss={running_loss/(i+1):.4f}')

            # print every 100 mini-batches
            if i % 100 == 99:
                wandb.log({'epoch': epoch+1, 'loss': running_loss/100})
                running_loss = 0.0

        # Compute accuracy on test set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                # labels.size() allows you to check out the shape of tensor `labels`
                # labels.size(0) is the number of labels in the batch
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test images: {accuracy:.2f} %')
        wandb.log({'epoch': epoch+1, 'accuracy': accuracy})

    # Save model weights
    torch.save(model.state_dict(), model_path)
    print(f'Finished Training, model saved to {model_path}')

    return model


def eval(model):
    """
    Evaluate the model

    Input:
        model: trained model
    Output:
        None
    """
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
            # Collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_predictions_by_class[classes[label]] += 1
                total_predictions_by_class[classes[label]] += 1

    # Print accuracy for each class
    for classname, correct_count in correct_predictions_by_class.items():
        accuracy = 100 * float(correct_count) / \
            total_predictions_by_class[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == '__main__':
    # Start a new run
    wandb.init(project='CIFAR_Net',
               entity='ghnmqdtg',
               name=f'CIFAR-Net-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

    # CIFAR10 dataset has 50000 training images and 10000 test images
    # Set batch_size to 100, so we have 500 batches for training and 100 batches for testing
    batch_size = 100
    model_path = './cifar10_ResNet50.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    # Prepare dataset
    train_loader, test_loader, classes = prepare_dataset(batch_size=batch_size)
    # Train the model
    model = train(
        model_path=model_path,
        train_loader=train_loader,
        test_loader=test_loader
    )
    # Evaluate the model
    eval(model)
