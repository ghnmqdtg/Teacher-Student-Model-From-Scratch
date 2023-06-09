import wandb
import utils
import config
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


# Define the teacher model based on pretrained ResNet50
class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        # Load pre-trained ResNet50 model
        self.model = torchvision.models.resnet50(
            weights='ResNet50_Weights.DEFAULT')
        # Modify the last layer to fit CIFAR10 dataset, since it was trained on ImageNet (1000 classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)


# Define the student model based on pretrained ResNet18
class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        # Load pre-trained ResNet18 model
        self.model = torchvision.models.resnet18(
            weights='ResNet18_Weights.DEFAULT')
        # Modify the last layer to fit CIFAR10 dataset, since it was trained on ImageNet (1000 classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)


# Define the loss function for knowledge distillation
class DistillationLoss:
    def __init__(self):
        self.student_loss = nn.CrossEntropyLoss()
        # Input of nn.KLDivLoss() should be a distribution in the log space
        # REF: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        self.distillation_loss = nn.KLDivLoss(reduction='batchmean')
        self.temperature = config.TEMPERATURE
        self.alpha = config.ALPHA

    def __call__(self, student_logits, labels, teacher_logits):
        student_loss = self.alpha * \
            self.student_loss(student_logits, labels)
        # nn.LogSoftmax() is used to convert logits to log-probabilities
        distillation_loss = (1 - self.alpha) * self.distillation_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                                                F.softmax(teacher_logits / self.temperature, dim=1))
        return student_loss + distillation_loss


# Define the training function for the teacher model
def train_teacher(teacher_model, train_loader, test_loader, optimizer, criterion):
    """
    Train the teacher model

    Args:
        teacher_model: the teacher model
        train_loader: the training data loader
        test_loader: the test data loader
        optimizer: the optimizer
        criterion: the loss function
    """
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
               name=f'teacher-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    # Load the datasets into dataloaders
    dataloaders = {'train': train_loader, 'val': test_loader}
    # Print the model architecture
    print(f'\nTeacher model architecture:\n{teacher_model}')
    # Compute the total number of trainable parameters in the model
    num_params = sum(p.numel()
                     for p in teacher_model.parameters() if p.requires_grad)
    print(f'The teacher model has {num_params:,} trainable parameters')
    # Print the prompt to start training
    print('Starting training the teacher model...\n')
    # Watch model in wandb
    wandb.watch(teacher_model, criterion, log="all")
    # Set number of epochs
    num_epochs = config.NUM_EPOCHS_TEACHER
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                teacher_model.train()
            else:
                teacher_model.eval()

            running_loss = 0.0
            running_corrects = 0
            pbar = tqdm(enumerate(dataloaders[phase]), total=len(
                dataloaders[phase]))
            for i, (inputs, labels) in pbar:
                # Move inputs and labels to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Set gradient calculation only in 'train' phase
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    outputs = teacher_model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Compute batch loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # Update tqdm progress bar
                pbar.set_description(
                    f'Epoch {epoch+1}/{num_epochs} | {phase} | Loss: {running_loss/(i+1):.4f} | Acc: {running_corrects.double()/(i+1):.4f}')

            if phase == 'train':
                # Comput epoch loss and accuracy
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / \
                    len(dataloaders[phase].dataset)
                wandb.log(
                    {'epoch': epoch+1, 'loss': epoch_loss, 'accuracy': epoch_acc})

    # Save model weights
    save_path = config.TEACHER_PATH
    torch.save(teacher_model.state_dict(), save_path)
    # Finish the run
    wandb.finish()
    print(f'Finished Training, model saved to {save_path}')


# Define the training function for the student model
def train_student(student_model, teacher_model, train_loader, test_loader, optimizer, criterion):
    """
    Train the student model

    Args:
        student_model: the student model
        teacher_model: the teacher model
        train_loader: the training data loader
        test_loader: the test data loader
        optimizer: the optimizer
        criterion: the loss function
    """
    # Define a config dictionary object
    wandb_config = {
        "num_epochs": config.NUM_EPOCHS_STUDENT,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "momentum": config.MOMENTUM,
        "temperature": config.TEMPERATURE,
        "alpha": config.ALPHA,
    }
    # Start a new run
    wandb.init(project='Teacher-Student-Model',
               entity='ghnmqdtg',
                config=wandb_config,
               name=f'student-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    # Load the datasets into dataloaders
    dataloaders = {'train': train_loader, 'val': test_loader}
    # Print the model architecture
    print(f'\nStudent model architecture:\n{student_model}\n')
    # Compute the number of trainable parameters in the student model
    num_params = sum(p.numel()
                     for p in student_model.parameters() if p.requires_grad)
    print(f'The student model has {num_params:,} trainable parameters\n')
    # Print the prompt to start training
    print('Starting training the student model...\n')
    # Watch model in wandb
    wandb.watch(student_model, criterion, log="all")
    # Set number of epochs
    num_epochs = config.NUM_EPOCHS_STUDENT
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                student_model.train()
                teacher_model.eval()
            else:
                student_model.eval()

            running_loss = 0.0
            running_corrects = 0
            pbar = tqdm(enumerate(dataloaders[phase]), total=len(
                dataloaders[phase]))
            for i, (inputs, labels) in pbar:
                # Move inputs and labels to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Set gradient calculation only in 'train' phase
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass of teacher model
                    teacher_logits = teacher_model(inputs)
                    # Forward pass of student model
                    student_logits = student_model(inputs)
                    _, preds = torch.max(student_logits, 1)
                    loss = criterion(
                        student_logits, labels, teacher_logits)
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Compute batch loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # Update tqdm progress bar
                pbar.set_description(
                    f'Epoch {epoch+1}/{num_epochs} | {phase} | Loss: {running_loss/(i+1):.4f} | Acc: {running_corrects.double()/(i+1):.4f}')

            if phase == 'test':
                # Comput epoch loss and accuracy
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / \
                    len(dataloaders[phase].dataset)
                wandb.log(
                    {'epoch': epoch+1, 'loss': epoch_loss, 'accuracy': epoch_acc})

    # Save model weights
    save_path = config.STUDENT_PATH
    torch.save(student_model.state_dict(), save_path)
    # Finish the run
    wandb.finish()
    print(f'Finished Training, model saved to {save_path}')


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'===== Device: {device} =====')
    # CIFAR10 dataset has 50000 training images and 10000 test images
    # Set batch_size to 100, so we have 500 batches for training and 100 batches for testing
    batch_size = config.BATCH_SIZE
    pretrained_weight = config.TEACHER_PATH
    # pretrained_weight = None
    
    # Create folders to save model weights
    utils.create_folder(config.CHECKPOINTS_PATH)

    # Prepare dataset
    train_loader, test_loader, classes = utils.prepare_dataset(
        batch_size=batch_size)

    # If pretrained_weight is None, train the teacher model
    teacher = Teacher().to(device)
    if pretrained_weight is None:
        teacher_optimizer = optim.SGD(
            teacher.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
        teacher_criterion = nn.CrossEntropyLoss()
        train_teacher(teacher_model=teacher, train_loader=train_loader, test_loader=test_loader,
                      optimizer=teacher_optimizer, criterion=teacher_criterion)
    else:
        teacher.load_state_dict(torch.load(pretrained_weight))

    teacher.eval()
    # Train the student model
    student = Student().to(device)
    student_optimizer = optim.SGD(student.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
    student_criterion = DistillationLoss()
    train_student(student_model=student, teacher_model=teacher, train_loader=train_loader, test_loader=test_loader,
                  optimizer=student_optimizer, criterion=student_criterion)
