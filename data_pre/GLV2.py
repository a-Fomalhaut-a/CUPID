import torch
from torchvision import transforms, datasets
import sys
sys.path.append('..')
from config import data_all

class AddNoise(object):
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def __call__(self, img):
        img_tensor = transforms.functional.to_tensor(img)
        noise = torch.rand_like(img_tensor) * self.noise_level
        noisy_img = img_tensor + noise
        return transforms.functional.to_pil_image(noisy_img)


# Applying Transforms to the Data
para = data_all['GLV2']
SIZE = para['fea_dim']
COLOR_DEVIATION = 0.01
root = para['initial_data_road']

image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=SIZE),
        AddNoise(0.01),
        transforms.ColorJitter(brightness=(1.0 - COLOR_DEVIATION, 1.0 + COLOR_DEVIATION),
                               contrast=(1.0 - COLOR_DEVIATION, 1.0 + COLOR_DEVIATION),
                               saturation=(1.0 - COLOR_DEVIATION, 1.0 + COLOR_DEVIATION),
                               hue=(-1 * COLOR_DEVIATION, COLOR_DEVIATION)),
        transforms.RandomRotation(degrees=0),
        transforms.RandomAffine(degrees=0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]),
    'validation': transforms.Compose([
        transforms.Resize(size=SIZE),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=SIZE),
        transforms.ToTensor(),
    ])
}

def preprocess():
    # Set train and valid directory paths
    train_directory = root + 'train'
    test_directory = root + 'test'
    validation_directory = root + 'validation'

    # Load Data from folders
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'validation': datasets.ImageFolder(root=validation_directory, transform=image_transforms['validation']),
        'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
    }

    # Size of Data, to be used for calculating Average Loss and Accuracy
    train_data_size = len(data['train'])
    validation_data_size = len(data['validation'])
    test_data_size = len(data['test'])

    print("train_data_size:{}".format(train_data_size))
    print("validation_data_size:{}".format(validation_data_size))
    print("test_data_size:{}".format(test_data_size))

    return data

