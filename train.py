# Standard imports
from collections import OrderedDict

# Data manipulation imports
import json

# Computer vision imports
import torch
from torch import nn
from torchvision import models, transforms, datasets

# Data loading and preprocessing imports
from parsers import make_train_parser


def make_optimizer(model, lr=0.001):
    '''Creates and returns an Adam optimizer with a learning rate of `lr` for the `model`.'''
    return torch.optim.Adam(model.classifier.parameters(), lr=lr)

def create_model(architecture, hidden_size=1024, p_dropout=0.2):
    '''Loads and returns a model from the available torchvision models.
       `architecture` must be the name of one of the models provided by torchvision.
    '''
    if architecture in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
        model = vars(models)[architecture](pretrained=True)
    else:
        raise KeyError(str(e) + ' is not a valid architecture.')
        
    for param in model.features.parameters():
        param.requires_grad_(False)

    classifier = nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(in_features=25088, out_features=hidden_size, bias=True)),
        ('ReLU_1', nn.ReLU()),
        ('Dropout', nn.Dropout(p=p_dropout)),
        ('fc2', nn.Linear(in_features=hidden_size, out_features=102, bias=True)),
        ('LogSoftmax', nn.LogSoftmax(dim=1))
    ]
    ))
    
    model.add_module('classifier', classifier)
#     print(model)
    return model


def make_dataloaders(train_dir, test_dir, valid_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=8)
    }

    return dataloaders, image_datasets


def evaluate(model, dataloader, criterion, device):
    '''Evaluates the loss and accuracy of the model on the provided dataloader.'''
    model.to(device)
    model.eval()
    with torch.no_grad():
        loss = 0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model.forward(images)
                loss += criterion(outputs, labels)
                predicted = outputs.max(dim=1)[1]    
                correct += (predicted==labels).sum().item()
                total += (predicted==labels).size()[0]

        loss = loss/(i+1)
        accuracy = correct/total
        return loss, accuracy

def create_train_return_model(architecture, hidden_size, p_dropout,
                              learning_rate, epochs, criterion,
                              train_loader, valid_loader, device):
    model = create_model(architecture, hidden_size, p_dropout)
    model.to(device)
    optimizer = make_optimizer(model, learning_rate)
    
    print('Starting training on {}...'.format(device))
    for e in range(epochs):
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1)%10==0:
                valid_loss, valid_accuracy = \
                evaluate(model, valid_loader, criterion, device)            
                print(("\rEpoch {}/{}, batch #{}, training loss: {:.4f}, "
                      + "validation loss: {:.4f}, validation accuracy: {:.2f}")\
                      .format(e+1, epochs, i+1, loss, valid_loss, valid_accuracy))
            else:
                print('\rEpoch {}/{}, batch #{}, training loss: {:.4f}'\
                      .format(e+1, epochs, i+1, loss), end='')
            
    return model

if __name__ == '__main__':
    parser = make_train_parser()
    args = parser.parse_args()

    data_dir = args.data_directory
    save_dir = args.save_dir
    arch = args.arch
    hidden_units = args.hidden_units
    learning_rate = args.learning_rate
    epochs = args.epochs

    if args.gpu:
        device='cuda'
    else:
        device='cpu'

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    dataloaders, image_datasets = make_dataloaders(train_dir, test_dir, valid_dir)
    
    criterion = nn.NLLLoss()

    model = create_train_return_model(
        arch, hidden_units, 0.2, learning_rate, epochs, criterion,
        dataloaders['train'], dataloaders['valid'], device)

    class_to_idx = image_datasets['train'].class_to_idx
#     idx_to_class = dict([(class_to_idx[cls], cls) for cls in class_to_idx])
#     idx_to_name = dict([(idx, cat_to_name[idx_to_class[idx]]) for idx in idx_to_class])

    checkpoint = {
        'architecture': arch,
        'hidden_size': hidden_units,
        'p_dropout': 0.2,
        'learning_rate': learning_rate,
        'class_to_idx': class_to_idx,
#         'idx_to_class': idx_to_class,
#         'idx_to_name': idx_to_name,
#         'class_to_name': cat_to_name,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, save_dir + '/checkpoint.pth')