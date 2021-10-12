#
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from pytraction.net.formats import DATASETS


class CellTrackPreprocessor:
    """
    Class used to preprocess datasets and det file for datasets. Here we assume that that each GT track
    is the GT for seg and tracking. Datasets from http://celltrackingchallenge.net/2d-datasets/
    """

    def __init__(
        self,
        data_path,
        datasets,
        gt_type,
        start_frame=0,
        gt_enc=None,
        gt_standard="GT",
        ext=".tif",
        shuffle=False,
    ):
        """
        :param data_path: data path for importing data from celltrackingchallenge.net/2d-datasets/
        :param datasets: dataset types as a list i.e. ['BF-C2DL-HSC','DIC-C2DH-HeLa']
        :param gt_standard: 'GT'
        :param gt_type: 'SEG'
        :param embeddings: default is None but should be some 'EMB' folder stored in 'GT' or 'ST' i.e. '01_GT/EMB
        :param ext: default is '.tif' based on cell tracking datasets. This only applies to mask format.
        :param shuffle: False, if you want to shuffle the file list

        Notes on ground truth:
        Gold reference tracking annotation (gold tracking truth) with complete cell instance coverage (unless explicitly stated) but poor cell region information;
        gt_standard: 'GT' gt_type: 'TRA'

        Gold reference segmentation annotation (gold segmentation truth) with very limited cell instance coverage but good cell region information;
        gt_standard: 'GT' gt_type: 'SEG'

        Silver reference segmentation annotation (silver segmentation truth) with good cell instance coverage but limited cell region information; so far available for nine datasets
        gt_standard: 'ST' gt_type: 'SEG'

        """
        self.data_path = data_path
        self.datasets = datasets
        self.shuffle = shuffle

        self.imgs = []
        self.masks = []
        self.embeddings = []

        self.gt_standard = gt_standard
        self.gt_type = gt_type
        self.gt_enc = gt_enc
        self.ext = ext
        self.start_frame = start_frame

        # load in the dataset by default
        self._load_dataset()

        # if shuffle then shuffle the file names
        if shuffle:
            self._shuffle()

    def _load_dataset(self):
        """load all datasets using formats objects"""
        for dataset in self.datasets:
            if dataset in DATASETS:
                dataset_obj = DATASETS[dataset](
                    self.data_path,
                    gt_standard=self.gt_standard,
                    gt_type=self.gt_type,
                    ext=self.ext,
                )
                img_files, mask_files = dataset_obj.return_img_mask_files()
                self.imgs += img_files
                self.masks += mask_files

            else:
                msg = f"Dataset: {dataset} has not been implimented"
                raise NotImplementedError(msg)

    def __len__(self):
        assert len(self.masks) == len(self.imgs)
        return len(self.masks)

    def __getitem__(self, idx):
        if idx > self.__len__():
            msg = f"list index out of range {idx} of shape {self.__len__()}"
            raise IndexError(msg)
        return self.imgs[idx], self.masks[idx]

    def _shuffle(self):
        if self.embeddings:
            zipped = list(zip(self.masks, self.imgs, self.embeddings))
            random.shuffle(zipped)
            self.masks, self.imgs, self.embeddings = zip(*zipped)
        else:
            zipped = list(zip(self.masks, self.imgs))
            random.shuffle(zipped)
            self.masks, self.imgs = zip(*zipped)

    def imread(self, idx):
        """load image for idx as np.array"""
        img, mask = self.__getitem__(idx)
        return cv2.imread(img), cv2.imread(mask, -1)

    def imshow(self, idx):
        """load image for idx as np.array"""

        img, mask = self.imread(idx)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(mask, vmax=np.max(mask))
        plt.title(self.__getitem__(idx)[1])
        return fig

    def get_video_sequence(
        self, dataset, play=False, gif=False, scale=100, mask_func=None
    ):
        """
        Get video sequence of dataset and play using default player
        :param dataset: target dataset to get sequence
        :param scale: multiplier to make greys in mask visable
        """
        if not isinstance(dataset, str):
            msg = "Please enter dataset as string"
            raise TypeError(msg)

        from cv2 import VideoWriter, VideoWriter_fourcc

        ctp = CellTrackPreprocessor(
            self.data_path,
            [dataset],
            start_frame=self.start_frame,
            gt_type=self.gt_type,
            gt_standard=self.gt_standard,
            ext=self.ext,
        )
        # get example image
        img, mask = ctp.imread(0)

        # Parameters for adding text to video sequence
        font = cv2.FONT_HERSHEY_SIMPLEX
        topcenteroftext = (10, 500)
        bottomLeftCornerOfText = (500, 50)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        # dimentions of video
        width = img.shape[1]
        height = img.shape[0]
        FPS = 4
        seeconds = 10

        # create video codec and videowriter object
        fourcc = VideoWriter_fourcc(*"MJPG")
        video = VideoWriter("tmp_seq.avi", fourcc, float(FPS), (width * 2, height))

        # iterate via frames
        for i in range(len(ctp)):
            img, mask = ctp.imread(i)
            img_name, mask_name = ctp[i]
            if mask_func != None:
                mask = mask_func(mask)
            mask = mask.astype("uint8")
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            # join img and mask together
            img = np.concatenate((img, mask * scale), axis=1)

            # add text to image
            cv2.putText(
                img,
                f"{i}",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )
            cv2.putText(
                img,
                f"{mask_name.split('data')[1]}",
                topcenteroftext,
                font,
                fontScale,
                fontColor,
                lineType,
            )
            sys.stdout.write(f"\rProcessed frame {i} out of {len(ctp)}")
            sys.stdout.flush()
            video.write(img)
        video.release()

        if play:
            os.system("xdg-open tmp_seq.avi")

        if gif and (sys.platform == "linux" or platform == "linux2"):
            path = f"{dataset}_{self.gt_standard}_{self.gt_type}.gif"
            os.system(f"ffmpeg -i tmp_seq.avi {path}")
            if not os.path.exists(path):
                msg = f"System: {sys.platform}. For linux users please ffmpeg 'sudo apt-get install ffmpeg -y' to output .gif \
                    and check 'ffmpeg -i tmp_seq.avi tmp_seq.gif'"
                raise OSError(msg)

        return "tmp_seq.avi"

    def new_mask_ground_truth_name(self, image_path, custom):
        """Create new file name for custom segmentation. Only permitted for

        :param image_path: image path to TRA file
        :param custom: custom name new gt type

        """
        if self.gt_type != "TRA":
            msg = f"Warning you can only use TRA gt_type and are using type {self.gt_type}"
            raise ValueError(msg)

        gt_standard = "GT"
        gt_type = custom
        new_path = image_path.replace(
            f"{self.gt_standard}/{self.gt_type}/man_track",
            f"{gt_standard}/{gt_type}/man_seg",
        )

        path, basename = os.path.split(new_path)

        if not os.path.exists(path):
            os.makedirs(path)

        return new_path
