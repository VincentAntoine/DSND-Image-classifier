import numpy as np


def resize(pil_image, size=256):
    ''' Returns a scaled version of the input image such that the length 
        of the shortest side equals the `size` parameter, preserving the
        aspect ratio of the image.
    '''
    initial_size = np.array(pil_image.size)
    shortest_side = initial_size.min()
    resize_ratio = size/shortest_side
    new_size = (resize_ratio * initial_size).round().astype(int)
    return pil_image.resize(new_size)


def center_crop(pil_image, size=224):
    ''' Returns a square and centered crop of the input
        image of dimensions size * size. 
    '''
    w, h= pil_image.size
    l = int((w - size) / 2)
    r = l + size
    t = int((h - size) / 2)
    b = t + size
    return pil_image.crop((l, t, r, b))


def normalize_tranpose(pil_image, means=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    ''' Normalizes the input pil_image by first dividing by 255 to bring
        the values between 0 and 1, then subtracts `means` and divides by `std`.
        Return the corresponding np_image transposed in a way that the color channel is first.
    '''
    means=np.array(means)
    std=np.array(std)
    np_image = np.array(pil_image)
    scaled_image = ((np_image/255) - means)/std
    return scaled_image.transpose(2, 0, 1)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    return normalize_tranpose(center_crop(resize(image)))
