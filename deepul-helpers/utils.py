import os
from os.path import join, dirname, exists
import matplotlib.pyplot as plt
import numpy as np
import pickle


def savefig(fname, show_figure=True):
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()


def save_training_plot(train_losses, test_losses, title, fname):
    plt.figure()
    n_epochs = len(test_losses)-1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs+1)

    plt.plot(x_train, train_losses, label='train_loss')
    plt.plot(x_test, test_losses, label='test_loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    savefig(fname)


def save_scatter_2d(data, title, fname):
    plt.figure()
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1])
    savefig(fname)


def save_distribution_1d(data, distribution, title, fname):
    d = len(distribution)

    plt.figure()
    plt.hist(data, bins=np.arange(d) - 0.5, label='train data', density=True)

    x = np.linspace(-0.5, d - 0.5, 1000)
    y = distribution.repeat(1000 // d)
    plt.plot(x, y, label='learned distribution')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    savefig(fname)


def save_distribution_2d(true_dist, learned_dist, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(true_dist)
    ax1.set_title('True Distribution')
    ax1.axis('off')
    ax2.imshow(learned_dist)
    ax2.set_title('Learned Distribution')
    ax2.axis('off')
    savefig(fname)


def show_samples(samples, fname=None, nrow=10, title='Samples'):
    #samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, imgs_per_row=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img)
    plt.axis('off')

    if fname is not None:
        savefig(fname)
    else:
        plt.show()


def load_pickled_data(fname, include_labels=False):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    train_data, test_data = data['train'], data['test']
    if 'mnist.pkl' in fname or 'shapes.pkl' in fname:
        # Binarize MNIST and shapes dataset
        train_data = (train_data > 127.5).astype('uint8')
        test_data = (test_data > 127.5).astype('uint8')
    if 'celeb.pkl' in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]
    if include_labels:
        return train_data, test_data, data['train_labels'], data['test_labels']
    return train_data, test_data


def get_data_dir(hw_number):
    return join('deepul', 'homeworks', f'hw{hw_number}', 'data')


def quantize(images, n_bits):
    images = np.floor(images / 256. * 2 ** n_bits)
    return images.astype('uint8')

def make_grid(images, imgs_per_row):
    if len(images.shape) == 2:  # single image H x W
        images = np.expand_dims(images, axis=-1)
    if len(images.shape) == 3:  # single image
        if images.shape[-1]== 1: # if single-channel, convert to 3-channel
            images = images.repeat(3, axis=-1)
        images = np.expand_dims(images, axis=0)

    if len(images.shape) == 4 and images.shape[-1] == 1:  # single-channel images
        images = images.repeat(3, axis=-1)

    assert len(images.shape) ==4 , 'images should be of shape (bs, h, w, c)'
   
    if images.shape[0] == 1:
        return np.squeeze(images, axis=0)

    num_imgs = images.shape[0]
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_row_imgs = min(num_imgs, imgs_per_row)

    if num_imgs > imgs_per_row:
        n_columns = int(np.ceil(num_imgs/n_row_imgs))
    else:
        n_columns = 1

    grid = np.zeros([n_columns*img_h, n_row_imgs*img_w, 3], dtype=images.dtype)
    for idx in range(num_imgs):
        x = (idx % n_row_imgs)  * img_w
        y = (idx // n_columns)  * img_h
        grid[y : y + img_h, x : x + img_w, :] = images[idx]
    return grid 