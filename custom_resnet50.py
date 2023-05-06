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
    model = CustomResNet50().to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Watch model in wandb
    wandb.watch(model, criterion, log="all")
    # Set number of epochs
    num_epochs = 5
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

            # Save to wandb every 100 mini-batches
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

        accuracy = correct / total
        print(f'Accuracy of the network on the test images: {100 * accuracy:.2f} %')
        wandb.log({'epoch': epoch+1, 'accuracy': accuracy})

    # Save model weights
    torch.save(model.state_dict(), model_path)
    print(f'Finished Training, model saved to {model_path}')


if __name__ == '__main__':
    # Start a new run
    wandb.init(project='Teacher-Student-Model',
               entity='ghnmqdtg',
               name=f'ResNet50-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

    # CIFAR10 dataset has 50000 training images and 10000 test images
    # Set batch_size to 100, so we have 500 batches for training and 100 batches for testing
    batch_size = config.BATCH_SIZE
    model_path = config.CIFAR10_RESNET50_PATH

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    # Prepare dataset
    train_loader, test_loader, classes = utils.prepare_dataset(batch_size=batch_size)
    # Train the model
    train(model_path=model_path,train_loader=train_loader,test_loader=test_loader)
