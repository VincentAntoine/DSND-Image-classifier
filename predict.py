import re
import json 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL
import torch

from parsers import make_predict_parser
from preprocessing import process_image
from train import create_model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = create_model(architecture=checkpoint['architecture'],
                         hidden_size=checkpoint['hidden_size'], 
                         p_dropout=checkpoint['p_dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = dict([(model.class_to_idx[cls], cls) for cls in model.class_to_idx])
    return model


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)    
    return ax


def predict(image_path, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print('Start loading checkpoint...', end='')
    model = load_checkpoint(model)
    print('checkpoint loaded.')
    model.to('cuda')
    model.eval()

    print('Opening image...', end='')
    pil_image = PIL.Image.open(image_path)
    print('image opened')
    processed_image = process_image(pil_image)
    tensor = torch.from_numpy(processed_image)
    tensor = tensor.float()
    tensor.resize_(1, 3, 224, 224)
    tensor = tensor.to('cuda')
    with torch.no_grad():
        probs, classes = torch.exp(model(tensor)).topk(top_k)
    
    probs, classes = probs.to('cpu'), classes.to('cpu')
    probs, classes = list(probs.numpy()[0]), list(classes.numpy()[0])
    return probs, [model.idx_to_class[idx] for idx in classes]


def sanity_check(image_path, checkpoint, top_k, cat_to_name):
    probs, classes = predict(image_path, checkpoint, top_k=top_k)

    # Find correct image class from the name of the folder containing the image
    regex = re.compile(r'/(\d+)/')
    correct_class, = regex.search(image_path).groups()

    # Find real flower names if a cat_to_name file is provided
    if cat_to_name is not None:
        with open(cat_to_name, 'r') as f:
            cat_to_name = json.load(f)
            classes = [cat_to_name[cls] for cls in classes]
            correct_class = cat_to_name[correct_class]
    print('-' * 50)
    print('Correct class: ', correct_class)
    print('-' * 50)
    print('Predicted classes and probabilities:')
    print('\n'.join(['{}: {:.3f}'.format(c, p) for p, c in zip(probs, classes)]))

    # Create plot
    fig, axs = plt.subplots(2, 1, figsize=(5, 8))
    
    ax = axs[0]
    ax.imshow(PIL.Image.open(image_path))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title(correct_class)
    
    ax = axs[1]
    ax.barh(classes, probs)
    fig.savefig('Classified picture.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    parser = make_predict_parser()
    args = parser.parse_args()
    print(args)
    image = args.image
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names

    if args.gpu:
        device='cuda'
    else:
        device='cpu'

    sanity_check(image, checkpoint, top_k, category_names)
    