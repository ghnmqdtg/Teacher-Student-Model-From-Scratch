import wandb
import utils
import config
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


# Define the ResNet50 for CIFAR10 dataset
class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        # Load pre-trained ResNet50 model
        self.model = torchvision.models.resnet50(
            weights='ResNet50_Weights.DEFAULT')
        # Modify the last layer to fit CIFAR10 dataset, since it was trained on ImageNet (1000 classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

# Define the CNN model based on basic CNN
class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input shape (3, 224, 224), output shape (64, 112, 112)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # The num_features of BatchNorm2d should be the same as the num_channels of Conv2d
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Input shape (64, 112, 112), output shape (64, 56, 56)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Input shape (64, 56, 56), output shape (64, 28, 28)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Input shape (64, 28, 28), output shape (24, 14, 14)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Input shape (24, 14, 14), output shape (12, 7, 7)
        self.layer5 = nn.Sequential(
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Input shape (12, 7, 7), output shape (10)
        self.fc = nn.Linear(12 * 7 * 7, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # Flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



def train(model, train_loader, test_loader, save_path):
    """
    Train the model

    Input:
        model: model to be trained
        train_loader: training data loader
        test_loader: testing data loader
        save_path: path to save the model weights
    Output:
        model: trained model
    """
    print(f'\nStart training on {device}')
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
    # Watch model in wandb
    wandb.watch(model, criterion, log="all")
    # Set number of epochs
    num_epochs = config.NUM_EPOCHS_TEACHER
    # Train the model
    model.train()
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

        # Compute loss and accuracy on test set
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_acc = running_corrects / len(test_loader.dataset)
        print(f'Accuracy of the network on the test images: {100 * epoch_acc:.2f} %')
        wandb.log({'epoch': epoch+1, 'loss': epoch_loss, 'accuracy': epoch_acc})

    # Save model weights
    torch.save(model.state_dict(), save_path)
    print(f'Finished Training, model saved to {save_path}')


if __name__ == '__main__':
    # Define a config dictionary object
    wandb_config = {
        "num_epochs": config.NUM_EPOCHS_TEACHER,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "momentum": config.MOMENTUM,
    }
    # Start a new run
    wandb.init(project='Teacher-Student-Model',
               entity='ghnmqdtg',
               config=wandb_config,
               name=f'ResNet50-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

    # CIFAR10 dataset has 50000 training images and 10000 test images
    # Set batch_size to 100, so we have 500 batches for training and 100 batches for testing
    batch_size = config.BATCH_SIZE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    # Prepare dataset
    train_loader, test_loader, classes = utils.prepare_dataset(batch_size=batch_size)
    # Train the CustomResNet50 model
    model = CustomResNet50().to(device)
    train(model=model, train_loader=train_loader, test_loader=test_loader, save_path=config.CIFAR10_RESNET50_PATH)
    wandb.finish()
    
    # Start a new run
    wandb.init(project='Teacher-Student-Model',
               entity='ghnmqdtg',
               config=wandb_config,
               name=f'BasicCNN-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    # Train the BasicCNN model
    model = BasicCNN().to(device)
    train(model=model, train_loader=train_loader, test_loader=test_loader, save_path=config.BASIC_CNN_PATH)
    wandb.finish()