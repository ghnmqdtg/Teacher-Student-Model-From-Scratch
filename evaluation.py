import utils
import config
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torchvision

from custom_resnet50 import CustomResNet50, BasicCNN
from teacher_student_model import Teacher, Student


def eval(model, name='val'):
    """
    Evaluate the model

    Input:
        model: trained model
    Output:
        None
    """
    utils.create_folder(f'./imgs/{name}')
    y_true = []
    y_pred = []
    # Prepare to count predictions for each class
    correct_predictions_by_class = {classname: 0 for classname in classes}
    total_predictions_by_class = {classname: 0 for classname in classes}

    # No gradients needed
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            # Move inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true += labels.tolist()
            y_pred += preds.tolist()
            # Collect the correct preds for each class
            for label, pred in zip(labels, preds):
                if label == pred:
                    correct_predictions_by_class[classes[label]] += 1
                total_predictions_by_class[classes[label]] += 1

    # Print accuracy for each class
    for classname, correct_count in correct_predictions_by_class.items():
        accuracy = 100 * float(correct_count) / \
            total_predictions_by_class[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    # Print overall accuracy
    overall_accuracy = 100 * \
        sum(correct_predictions_by_class.values()) / \
        sum(total_predictions_by_class.values())
    print(f'\nOverall accuracy: {overall_accuracy:.1f} %\n')

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=classes, yticklabels=classes)
    plt.savefig(f'./imgs/{name}/confusion_matrix.png')
    plt.close()
    print(f'Confusion matrix saved to ./imgs/{name}/confusion_matrix.png')

    # Compute precision, recall, and F1 score
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True)

    # Extract precision, recall, and F1 scores from classification report
    precision = [report[label]['precision'] for label in classes]
    recall = [report[label]['recall'] for label in classes]
    f1_score = [report[label]['f1-score'] for label in classes]

    # Create a dataframe to store the scores
    df_scores = pd.DataFrame(
        {'Precision': precision, 'Recall': recall, 'F1-score': f1_score}, index=classes)

    # Plot the scores using bar charts
    sns.set_style('whitegrid')
    ax = df_scores.plot(kind='bar', figsize=(8, 6))
    ax.set_title('Classification Report')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'./imgs/{name}/classification_report.png')
    plt.close()
    print(f'Classification report saved to ./imgs/{name}/classification_report.png')


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n===== Device: {device} =====')
    batch_size = config.BATCH_SIZE
    
    # Create folders to save model weights
    utils.create_folder(config.CHECKPOINTS_PATH)

    # Prepare dataset
    _, test_loader, classes = utils.prepare_dataset(
        batch_size=batch_size)

    # Evaluate the custom_resnet50
    print('\n===== Evaluating the custom_resnet50 =====')
    custom_resnet50 = CustomResNet50()
    custom_resnet50 = custom_resnet50.to(device)
    custom_resnet50.load_state_dict(torch.load(config.CIFAR10_RESNET50_PATH), strict=False)
    eval(custom_resnet50, 'custom_resnet50')

    # Evaluate the basic_cnn
    print('\n===== Evaluating the basic_cnn =====')
    basic_cnn = BasicCNN()
    basic_cnn = basic_cnn.to(device)
    basic_cnn.load_state_dict(torch.load(config.BASIC_CNN_PATH))
    eval(basic_cnn, 'basic_cnn')

    # TODO: Check why the result is not good here, its structure is basically the same as custom_resnet50
    # TODO: It should be able to achieve 96% accuracy
    # Evaluate the teacher model
    print('\n===== Evaluating the teacher model =====')
    teacher = Teacher(pretrained_weight=None)
    teacher = teacher.to(device)
    eval(teacher, 'teacher')

    # Evaluate the student model
    print('\n===== Evaluating the student model =====')
    student = Student()
    student = student.to(device)
    student.load_state_dict(torch.load(config.STUDENT_PATH))
    eval(student, 'student')
