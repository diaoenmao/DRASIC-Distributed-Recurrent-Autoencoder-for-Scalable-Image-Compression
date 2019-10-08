import config

config.init()
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data import fetch_dataset

parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], help=\'\')'.format(k))
args = vars(parser.parse_args())
for k in config.PARAM:
    if config.PARAM[k] != args[k]:
        exec('config.PARAM[\'{0}\'] = {1}'.format(k, args[k]))


def process_dataset_label(dataset):
    img, label = dataset.img, dataset.label
    label = np.array(label)
    processed_img = [None for _ in range(dataset.classes_size)]
    for i in range(dataset.classes_size):
        processed_img[i] = img[label == i]
    min_size = min([len(processed_img[i]) for i in range(len(processed_img))])
    for i in range(dataset.classes_size):
        processed_img[i] = processed_img[i][:min_size]
    for i in range(dataset.classes_size):
        processed_img[i] = processed_img[i].reshape(-1).astype(np.float32)
    processed_img = np.stack(processed_img, axis=1)
    return processed_img


def process_dataset_random(dataset):
    img, label = dataset.img, dataset.label
    indices = np.random.permutation(img.shape[0])
    processed_img = img[indices]
    processed_img = np.split(processed_img, dataset.classes_size, axis=0)
    for i in range(dataset.classes_size):
        processed_img[i] = processed_img[i].reshape(-1).astype(np.float32)
    processed_img = np.stack(processed_img, axis=1)
    return processed_img


def CorrMtx(df, dropDuplicates=True):
    # Your dataset is already a correlation matrix.
    # If you have a dateset where you need to include the calculation
    # of a correlation matrix, just uncomment the line below:
    # df = df.corr()

    # Exclude duplicate correlations by masking uper right values
    if dropDuplicates:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # Set background color / chart style
    sns.set_style(style='white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Draw correlation plot with or without duplicates
    if dropDuplicates:
        sns.heatmap(df, mask=mask, cmap=cmap,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    else:
        sns.heatmap(df, cmap=cmap,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.show()
    return


def pearson_corr(X, Y):
    N = X.shape[0]
    X_mean, Y_mean = X.mean(axis=0), Y.mean(axis=0)
    cov = (X - X_mean).T @ (Y - Y_mean) / N
    sigma_X, sigma_Y = np.sqrt(X_mean ** 2 / N), np.sqrt(X_mean ** 2 / N)
    sigma_XY = sigma_X.reshape(-1, 1) @ sigma_Y.reshape(1, -1)
    corr = cov / (sigma_XY + 1e-10)
    corr[sigma_XY < 1e-1] = 0
    return corr


def main():
    fig_format = 'png'
    dataset = fetch_dataset(data_name=config.PARAM['data_name']['train'])['train']
    # img = process_dataset_label(dataset)
    img_random = process_dataset_random(dataset)
    img_label = process_dataset_label(dataset)
    corr_random = np.corrcoef(img_random, rowvar=False)
    corr_label = np.corrcoef(img_label, rowvar=False)
    ax = sns.heatmap(corr_random, vmin=-1, vmax=1, linewidths=.5, annot=True)
    # plt.show()
    plt.savefig('./output/fig/corr_random.{}'.format(fig_format), bbox_inches='tight', pad_inches=0, dpi=300)
    return


if __name__ == "__main__":
    main()