import random

import torch
from PIL import Image
from torchvision.transforms import transforms


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
    image = Image.open(image_path).convert('RGB')
    # Define a transform to convert PIL
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    return transform(image)


def tensor_to_image(tensor):
    # define a transform to convert a tensor to PIL image
    transform = transforms.ToPILImage()
    # convert the tensor to PIL image using above transform
    return transform(tensor)

def generate_masked_image(image, mask):
    return image * (mask < 0.5)


mask = image_to_tensor('/Users/TheBirwinator/PycharmProjects/waifu-diffusion/minidataset/m0.png')
print('MASK SHAPE PRE-BINARIZATION: ' + str(mask.shape))
masks_sum = mask.sum(axis=0)
binarized_mask = torch.where(masks_sum > 0, 1.0, 0.).unsqueeze(0)
print('MASK SHAPE POST-BINARIZATION: ' + str(binarized_mask.shape))

img = image_to_tensor('/Users/TheBirwinator/PycharmProjects/waifu-diffusion/minidataset/i0.png')
tensor_to_image(generate_masked_image(img, binarized_mask)).show()
tensor_to_image(img).show()



def extract_image_number(path):
    return int(path.split('/')[-1].split('.')[0][1:])