import random

import torch
from PIL import Image


def get_cutout_holes(height, width, min_holes=8, max_holes=32, min_height=16, max_height=128, min_width=16,
                     max_width=128):
    holes = []
    for _n in range(random.randint(min_holes, max_holes)):
        hole_height = random.randint(min_height, max_height)
        hole_width = random.randint(min_width, max_width)
        y1 = random.randint(0, height - hole_height)
        x1 = random.randint(0, width - hole_width)
        y2 = y1 + hole_height
        x2 = x1 + hole_width
        holes.append((x1, y1, x2, y2))
    return holes


def generate_random_mask(image):
    mask = torch.zeros_like(image[:1])
    holes = get_cutout_holes(mask.shape[1], mask.shape[2])
    for (x1, y1, x2, y2) in holes:
        mask[:, y1:y2, x1:x2] = 1.
    if random.uniform(0, 1) < 0.25:
        mask.fill_(1.)
    masked_image = image * (mask < 0.5)
    return mask, masked_image


def image_to_tensor(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = image.convert('RGB')
    image = torch.tensor(list(image.getdata())).float()
    image = image.view(3, 256, 256) / 255.
    return image

def tensor_to_image(tensor):
    image = tensor * 255.
    image = image.permute(1, 2, 0).byte()
    image = Image.fromarray(image.numpy())
    return image

def generate_masked_image(image, mask):
    return image * (mask < 0.5)


msk = Image.open(self.mask_files[idx]).convert('RGB')

path = '/content/waifu_diffusion_inpainting/images/i2.png'

def extract_image_number(path):
    return int(path.split('/')[-1].split('.')[0][1:])

print(extract_image_number(path))