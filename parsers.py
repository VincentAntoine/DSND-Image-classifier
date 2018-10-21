import argparse
import os 


def make_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', action='store', 
                        help='Path to training directory.')
    parser.add_argument('--save_dir', action='store', dest='save_dir',
                        default=os.path.abspath(os.curdir),
                        help='Path to the directory where the model checkpoint will be saved.')
    parser.add_argument('--arch', action='store', dest='arch',
                        default='vgg13', help='Model architecture - allowed values are "vgg11", "vgg13", "vgg16", "vgg19". Default is vgg13.')
    parser.add_argument('--hidden_units', action='store', dest='hidden_units',
                        default=1024, type=int,
                        help='Number of hidden units in the classifier hidden layer.')
    parser.add_argument('--learning_rate', action='store', dest='learning_rate',
                        default=0.001, type=float,
                        help='Learning rate.')
    parser.add_argument('--epochs', action='store', dest='epochs',
                        default=3, type=int,
                        help='Number of epochs in the training phase.')
    parser.add_argument('--gpu', action='store_true', dest='gpu',
                        default=False, help='Activate GPU computing.')
    return parser

def make_predict_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', action='store', 
                        help='path to the image to classify.')
    parser.add_argument('checkpoint', action='store', 
                        help='path to the checkpoint of the model to load to classify the image.')
    parser.add_argument('--top_k', action='store', type=int, default=1, 
                        help='Number of classes to return, by decreasing order or probability.')
    parser.add_argument('--category_names', action='store', 
                        default=None, help='path to a json file containing a dictionary with classe as keys and flower names as values')
    parser.add_argument('--gpu', action='store_true', dest='gpu',
                        default=False, help='Activate GPU computing.')
    return parser