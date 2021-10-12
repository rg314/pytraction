import os

import matplotlib.pyplot as plt


def get_img_from_seg(path_to_file):
    path, file = os.path.split(path_to_file)
    img_name = (
        f"{os.sep}".join(path.split(os.sep)[:-1]).split("_")[0]
        + os.sep
        + file.replace("man_seg", "t")
    )
    if os.path.exists(img_name):
        return img_name
    else:
        msg = "Raw image not found"
        raise ValueError(msg)


# helper function for data visualization
def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()
