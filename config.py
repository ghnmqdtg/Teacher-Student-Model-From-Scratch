# Hyperparameters
BATCH_SIZE = 100
NUM_EPOCHS_TEACHER = 5
NUM_EPOCHS_STUDENT = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
TEMPERATURE = 5
ALPHA = 0.25

# Paths
CIFAR10_DATA_PATH = './data'
CHECKPOINTS_PATH = './checkpoints'
IMGS_PATH = './imgs'
CIFAR10_RESNET50_PATH = './checkpoints/custom_resnet50.pt'
BASIC_CNN_PATH = './checkpoints/basic_cnn.pt'
TEACTHER_PATH = './checkpoints/teacher_model.pt'
STUDENT_PATH = './checkpoints/student_model.pt'