import os
import config
import torch
import torchvision
import torchvision.transforms as transforms

def create_folder(path):
    """
    Create a folder if it does not exist

    Input:
        path: path to the folder
    Output:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)


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
        root=config.CIFAR10_DATA_PATH,
        train=True,
        download=True,
        transform=transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root=config.CIFAR10_DATA_PATH,
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
