import matplotlib.pyplot as plt


def visualize(**images, savepath=None):
    """
    plot images in a row
    :param images: what to plot paired with desired title [title=array]
    :param savepath: where to save the resulting plot if desired [str]
    """
    n = len(images)
    if 'mask' in images.keys():
        n += images['mask'].shape[2]-1
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        if name == 'mask':
            for jj in range(image.shape[2]):
                plt.subplot(1, n, i + 1 + jj)
                plt.xticks([])
                plt.yticks([])
                plt.title(f'Mask channel: {jj}'.title(), fontsize=18)
                plt.imshow(image[:, :, jj])
        elif name == 'ROI':
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title(), fontsize=18)
            plt.imshow(image, cmap='gray') 
        else:
            plt.imshow(image)
            if savepath:
                plt.savefig(savepath)
    plt.show()