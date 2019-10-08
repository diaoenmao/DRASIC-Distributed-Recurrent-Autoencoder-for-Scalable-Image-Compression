import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms


def process_dataset(dataset, type):
    img, label = dataset.data.numpy(), dataset.targets
    if type == 'random':
        indices = np.random.permutation(img.shape[0])
        processed_img = img[indices]
        processed_img = np.split(processed_img, 10, axis=0)
        for i in range(10):
            processed_img[i] = processed_img[i].reshape(-1).astype(np.float32)
        processed_img = np.stack(processed_img, axis=1)
    elif type == 'label':
        label = np.array(label)
        processed_img = [None for _ in range(10)]
        for i in range(10):
            processed_img[i] = img[label == i]
        min_size = min([len(processed_img[i]) for i in range(len(processed_img))])
        for i in range(10):
            processed_img[i] = processed_img[i][:min_size]
        for i in range(10):
            processed_img[i] = processed_img[i].reshape(-1).astype(np.float32)
        processed_img = np.stack(processed_img, axis=1)
    else:
        raise ValueError('Not valid type')
    return processed_img


def main():
    type = 'random'
    fig_format = 'png'
    dataset = torchvision.datasets.MNIST(root='.', train=True, download=True)
    img = process_dataset(dataset, type)
    corr_random = np.corrcoef(img, rowvar=False)
    ax = sns.heatmap(corr_random, vmin=-1, vmax=1, linewidths=.5, annot=True)
    plt.show()
    # plt.savefig('./output/fig/corr_{}}.{}'.format(type, fig_format), bbox_inches='tight', pad_inches=0, dpi=300)
    return


if __name__ == "__main__":
    main()