import cv2
import numpy as np
import torch

from pytraction.net.dataloader import get_preprocessing


def get_mask(img, model, pre_fn, device="cuda"):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.asarray(img)[:, :, :3]

    image = pre_fn(image=img)["image"][:1, :, :]
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    pr_mask = model.predict(x_tensor)

    return pr_mask.squeeze().cpu().numpy().round()
