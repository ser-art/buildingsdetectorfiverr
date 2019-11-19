import torch
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose, Resize

import numpy as np
from PIL import Image

import cv2

from solaris.nets.zoo import XDXD_SpaceNet4_UNetVGG16


input_shape = (3, 512, 512)


def load_model(weights_url):
    state_dict = torch.load(weights_url)
    model = XDXD_SpaceNet4_UNetVGG16()

    model.load_state_dict(state_dict=state_dict)
    return model.cuda()


# png/jpeg, return pytorch tensor
def load_img(img_url):
    img = Image.open(img_url)
    img = np.array(img)
    img = Image.fromarray(img[:, :, : input_shape[0]])
    return img


def predict_mask(model, img, threshold=0.3):
    img_transform = Compose([
        Resize((input_shape[1], input_shape[2])),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    input_img = torch.unsqueeze(img_transform(img), dim=0).cuda()

    prediction = torch.sigmoid(model(input_img)).data[0].cpu().numpy()[0]
    return np.uint8(prediction > threshold)*(2**8 - 1)


def save_img_with_mask(save_url, img_url, mask, output_size=None):
    img = load_img(img_url)
    if not output_size:
        output_size = img.size
    img = img.resize((input_shape[1], input_shape[2]))
    img_with_mask = Image.fromarray(cv2.bitwise_or(
        np.array(img), np.array(img), mask=255-mask)).resize(output_size)
    img_with_mask.save(save_url)
    print("Image saved!")
    return img_with_mask
