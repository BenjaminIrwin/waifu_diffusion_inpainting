import glob
import os
import random
import re
import shutil
from typing import Generator, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps

from trainer.mask_gen import tensor_to_image


class Resize():
    def __init__(self, is_resizing: bool, is_not_migrating: bool) -> None:
        if not is_resizing:
            self.resize = self.__no_op
            return

        if not is_not_migrating:
            self.resize = self.__migration
            dataset_path = os.path.split('/minidataset')
            self.__directory = os.path.join(
                dataset_path[0],
                f'{dataset_path[1]}_cropped'
            )
            os.makedirs(self.__directory, exist_ok=True)
            return print(f"Resizing: Performing migration to '{self.__directory}'.")

        self.resize = self.__no_migration

    def __no_migration(self, image_path: str, w: int, h: int):
        return ImageOps.fit(
            Image.open(image_path),
            (w, h),
            bleed=0.0,
            centering=(0.5, 0.5),
            method=Image.Resampling.LANCZOS
        ).convert(mode='RGB')

    def __migration(self, image_path: str, w: int, h: int):
        filename = re.sub('\.[^/.]+$', '', os.path.split(image_path)[1])

        image = ImageOps.fit(
            Image.open(image_path),
            (w, h),
            bleed=0.0,
            centering=(0.5, 0.5),
            method=Image.Resampling.LANCZOS
        ).convert(mode='RGB')

        image.save(
            os.path.join(f'{self.__directory}', f'{filename}.jpg'),
            optimize=True
        )

        try:
            shutil.copy(
                os.path.join('/minidataset', f'{filename}.txt'),
                os.path.join(self.__directory, f'{filename}.txt'),
                follow_symlinks=False
            )
        except (FileNotFoundError):
            f = open(
                os.path.join(self.__directory, f'{filename}.txt'),
                'w',
                encoding='UTF-8'
            )
            f.close()

        return image

    def __no_op(self, image_path: str, w: int, h: int):
        return Image.open(image_path)

class Validation():
    def __init__(self, is_skipped: bool, is_extended: bool) -> None:
        if is_skipped:
            self.validate = self.__no_op
            return print("Validation: Skipped")

        if is_extended:
            self.validate = self.__extended_validate
            return print("Validation: Extended")

        self.validate = self.__validate
        print("Validation: Standard")

    def __validate(self, fp: str) -> bool:
        try:
            Image.open(fp)
            return True
        except:
            print(f'WARNING: Image cannot be opened: {fp}')
            return False

    def __extended_validate(self, fp: str) -> bool:
        try:
            Image.open(fp).load()
            return True
        except (OSError) as error:
            if 'truncated' in str(error):
                print(f'WARNING: Image truncated: {error}')
                return False
            print(f'WARNING: Image cannot be opened: {error}')
            return False
        except shutil.Error as error:
            print(f'WARNING: Image cannot be opened: {error}')
            return False

    def __no_op(self, fp: str) -> bool:
        return True


class ImageStore:
    def __init__(self, data_dir: str) -> None:
        print('ImageStore: Initializing')
        self.data_dir = data_dir
        self.resizer = Resize(False, True).resize
        self.validator = Validation(False, False).validate

        print('ImageStore: Loading images from ' + self.data_dir)
        self.image_files = []
        [self.image_files.extend(glob.glob(f'{data_dir}' + '/i*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
        print(f'ImageStore: Found {len(self.image_files)} images')
        print(self.image_files)

        print('ImageStore: Loading masks from ' + self.data_dir)
        self.mask_files = []
        [self.mask_files.extend(glob.glob(f'{data_dir}' + '/m*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
        print(f'ImageStore: Found {len(self.mask_files)} masks')
        print(self.mask_files)

        self.clean()

    def __len__(self) -> int:
        return len(self.image_files)

    # iterator returns images as PIL images and their index in the store
    # def entries_iterator(self) -> Generator[Tuple[Image, int], None, None]:
    #     for f in range(len(self)):
    #         yield Image.open(self.image_files[f]), f

    # get image by index
    def get_image_and_mask(self, idx):
        print('ImageStore: Getting image and mask')
        img = Image.open(self.image_files[idx]).convert('RGB')
        msk = Image.open(self.mask_files[idx]).convert('RGB')
        return img, msk

    # gets caption by removing the extension from the filename and replacing it with .txt
    def get_caption(self, idx) -> str:
        filename = re.sub('\.[^/.]+$', '', self.image_files[idx]) + '.txt'
        with open(filename, 'r', encoding='UTF-8') as f:
            return f.read()

    def extract_input_num(self, path):
        return int(path.split('/')[-1].split('.')[0][1:])

    def clean(self) -> None:

        print('ImageStore: Sorting images and masks')

        clean_images = []
        clean_masks = []

        # Create a dictionary that maps image numbers to filenames
        image_dict = {}
        for img in self.image_files:
            # If image is valid
            if self.validator(img):
                # Extract the number from the filename
                img_num = self.extract_input_num(img)
                image_dict[img_num] = img

        # Create a dictionary that maps mask numbers to filenames
        mask_dict = {}
        for msk in self.mask_files:
            # If mask is valid
            if self.validator(msk):
                # Extract the number from the filename
                msk_num = self.extract_input_num(msk)
                mask_dict[msk_num] = msk

        # Iterate over the keys (numbers) in the image dictionary
        for img_num in image_dict:
            # Check if the number exists in the mask dictionary
            if img_num in mask_dict:
                # If it does, add the corresponding filenames to the clean lists
                print('FOUND MATCHING MASK AND IMAGE: ' + str(img_num))
                clean_images.append(image_dict[img_num])
                clean_masks.append(mask_dict[img_num])

        print(clean_images)
        print(clean_masks)

        self.image_files = clean_images
        print(self.image_files)
        self.mask_files = clean_masks
        print(self.mask_files)

def generate_masked_image(image, mask):
    return image * (mask < 0.5)

class InpaintDataset(torch.utils.data.Dataset):
    def __init__(self, store: ImageStore,
                 ucg: float = 0.1):
        self.store = store
        self.ucg = ucg

        self.transforms = torchvision.transforms.Compose([
            # torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.store)

    def __getitem__(self, idx):
        return_dict = {'image_pixel_values': None, 'mask_pixel_values': None, 'masked_image_pixel_values': None, 'input_ids': None}

        image, mask = self.store.get_image_and_mask(idx)

        state = torch.get_rng_state()
        return_dict['image_pixel_values'] = self.transforms(image)
        print('SHAPE OF IMAGE: ' + str(return_dict['image_pixel_values'].shape))

        torch.set_rng_state(state)
        return_dict['mask_pixel_values'] = self.transforms(mask)
        return_dict['mask_pixel_values'] = (return_dict['mask_pixel_values'] >= 0.5).type(return_dict['mask_pixel_values'].type())
        print('SHAPE OF MASK: ' + str(return_dict['mask_pixel_values'].shape))

        torch.set_rng_state(state)
        masked_image = generate_masked_image(return_dict['image_pixel_values'], return_dict['mask_pixel_values'])
        print('SHAPE OF MASKED IMAGE: ' + str(masked_image.shape))
        tensor_to_image(masked_image).show()

        return_dict['masked_image_pixel_values'] = self.transforms(masked_image.numpy())

        # TODO: Do we still need this?
        if random.random() > self.ucg:
            caption_file = self.store.get_caption(idx)
        else:
            caption_file = ''

        return_dict['input_ids'] = caption_file
        return return_dict

    def collate_fn(self, examples):

        pixel_values = [example["image_pixel_values"] for example in examples]
        mask_values = [example["mask_pixel_values"] for example in examples]
        masked_image_values = [example["masked_image_pixel_values"] for example in examples]

        pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
        mask_values = torch.stack(mask_values).to(memory_format=torch.contiguous_format).float()
        masked_image_values = torch.stack(masked_image_values).to(memory_format=torch.contiguous_format).float()

        batch = {
            "image_pixel_values": pixel_values,
            "mask_pixel_values": mask_values,
            "masked_image_pixel_values": masked_image_values
        }

        return batch

# load dataset
# TODO: add mask support here
store = ImageStore('/minidataset')
dataset = InpaintDataset(store)

print(f'STORE_LEN: {len(store)}')

# TODO: add mask support here (in collate_fn)
train_dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    collate_fn=dataset.collate_fn
)

for _, batch in enumerate(train_dataloader):
    print('BATCH')
    print(batch['image_pixel_values'])
    print(batch['mask_pixel_values'])
    print(batch['masked_image_pixel_values'])