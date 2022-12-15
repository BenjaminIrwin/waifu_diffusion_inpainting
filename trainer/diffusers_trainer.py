import argparse
import socket
import torch
import torchvision
import transformers
import diffusers
import os
import glob
import random
import tqdm
import resource
import psutil
import pynvml
import wandb
import gc
import time
import itertools
import numpy as np
import json
import re
import traceback
import shutil

from torch.nn.functional import interpolate

try:
    pynvml.nvmlInit()
except pynvml.nvml.NVMLError_LibraryNotFound:
    pynvml = None

from typing import Iterable, Tuple
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, PNDMScheduler, DDIMScheduler, \
    StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.optimization import get_scheduler
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageOps

from typing import Dict, List, Generator, Tuple
from scipy.interpolate import interp1d

torch.backends.cuda.matmul.allow_tf32 = True

# defaults should be good for everyone
bool_t = lambda x: x.lower() in ['true', 'yes', '1']
parser = argparse.ArgumentParser(description='Stable Diffusion Finetuner')
parser.add_argument('--model', type=str, default=None, required=True,
                    help='The name of the model to use for finetuning. Could be HuggingFace ID or a directory')
parser.add_argument('--resume', type=str, default=None,
                    help='The path to the checkpoint to resume from. If not specified, will create a new run.')
parser.add_argument('--run_name', type=str, default=None, required=True, help='Name of the finetune run.')
parser.add_argument('--dataset', type=str, default=None, required=True,
                    help='The path to the dataset to use for finetuning.')
parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--use_ema', type=bool_t, default='False', help='Use EMA for finetuning')
parser.add_argument('--resize', type=bool_t, default='False', help='Resize input')
parser.add_argument('--ucg', type=float, default=0.1,
                    help='Percentage chance of dropping out the text condition per batch. Ranges from 0.0 to 1.0 where 1.0 means 100% text condition dropout.')  # 10% dropout probability
parser.add_argument('--gradient_checkpointing', dest='gradient_checkpointing', type=bool_t, default='False',
                    help='Enable gradient checkpointing')
parser.add_argument('--use_8bit_adam', dest='use_8bit_adam', type=bool_t, default='False',
                    help='Use 8-bit Adam optimizer')
parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1')
parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2')
parser.add_argument('--adam_weight_decay', type=float, default=1e-2, help='Adam weight decay')
parser.add_argument('--adam_epsilon', type=float, default=1e-08, help='Adam epsilon')
parser.add_argument('--lr_scheduler', type=str, default='cosine',
                    help='Learning rate scheduler [`cosine`, `linear`, `constant`]')
parser.add_argument('--lr_scheduler_warmup', type=float, default=0.05,
                    help='Learning rate scheduler warmup steps. This is a percentage of the total number of steps in the training run. 0.1 means 10 percent of the total number of steps.')
parser.add_argument('--seed', type=int, default=42,
                    help='Seed for random number generator, this is to be used for reproduceability purposes.')
parser.add_argument('--output_path', type=str, default='./output', help='Root path for all outputs.')
parser.add_argument('--save_steps', type=int, default=500, help='Number of steps to save checkpoints at.')
parser.add_argument('--resolution', type=int, default=512,
                    help='Image resolution to train against. Lower res images will be scaled up to this resolution and higher res images will be scaled down.')
parser.add_argument('--shuffle', dest='shuffle', type=bool_t, default='True', help='Shuffle dataset')
parser.add_argument('--hf_token', type=str, default=None, required=False,
                    help='A HuggingFace token is needed to download private models for training.')
parser.add_argument('--project_id', type=str, default='diffusers', help='Project ID for reporting to WandB')
parser.add_argument('--fp16', dest='fp16', type=bool_t, default='False', help='Train in mixed precision')
parser.add_argument('--image_log_steps', type=int, default=500, help='Number of steps to log images at.')
parser.add_argument('--image_log_amount', type=int, default=4, help='Number of images to log every image_log_steps')
parser.add_argument('--image_log_inference_steps', type=int, default=50,
                    help='Number of inference steps to use to log images.')
parser.add_argument('--image_log_scheduler', type=str, default="PNDMScheduler",
                    help='Number of inference steps to use to log images.')
parser.add_argument('--clip_penultimate', type=bool_t, default='False',
                    help='Use penultimate CLIP layer for text embedding')
parser.add_argument('--use_xformers', type=bool_t, default='False', help='Use memory efficient attention')
parser.add_argument('--wandb', dest='enablewandb', type=bool_t, default='True',
                    help='Enable WeightsAndBiases Reporting')
parser.add_argument('--inference', dest='enableinference', type=bool_t, default='True',
                    help='Enable Inference during training (Consumes 2GB of VRAM)')
parser.add_argument('--extended_validation', type=bool_t, default='False',
                    help='Perform extended validation of images to catch truncated or corrupt images.')
parser.add_argument('--no_migration', type=bool_t, default='False',
                    help='Do not perform migration of dataset while the `--resize` flag is active. Migration creates an adjacent folder to the dataset with <dataset_dirname>_cropped.')
parser.add_argument('--skip_validation', type=bool_t, default='False',
                    help='Skip validation of images, useful for speeding up loading of very large datasets that have already been validated.')
parser.add_argument('--extended_mode_chunks', type=int, default=0,
                    help='Enables extended mode for tokenization with given amount of maximum chunks. Values < 2 disable.')
parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")

args = parser.parse_args()


def setup():
    torch.distributed.init_process_group("nccl", init_method="env://")


def cleanup():
    torch.distributed.destroy_process_group()


def get_rank() -> int:
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def generate_masked_image(image, mask):
    return image * (mask < 0.5)


def binary_mask_to_tensor(image) -> torch.Tensor:
    # Convert the PIL image to a NumPy array
    np_image = np.array(image)

    # Convert the NumPy array to a Pytorch tensor
    tensor = torch.from_numpy(np_image)

    # Reshape the tensor to the desired shape
    tensor = tensor.view(1, 256, 256)

    return tensor


def get_gpu_ram() -> str:
    """
    Returns memory usage statistics for the CPU, GPU, and Torch.

    :return:
    """
    gpu_str = ""
    torch_str = ""
    try:
        cudadev = torch.cuda.current_device()
        nvml_device = pynvml.nvmlDeviceGetHandleByIndex(cudadev)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_device)
        gpu_total = int(gpu_info.total / 1E6)
        gpu_free = int(gpu_info.free / 1E6)
        gpu_used = int(gpu_info.used / 1E6)
        gpu_str = f"GPU: (U: {gpu_used:,}mb F: {gpu_free:,}mb " \
                  f"T: {gpu_total:,}mb) "
        torch_reserved_gpu = int(torch.cuda.memory.memory_reserved() / 1E6)
        torch_reserved_max = int(torch.cuda.memory.max_memory_reserved() / 1E6)
        torch_used_gpu = int(torch.cuda.memory_allocated() / 1E6)
        torch_max_used_gpu = int(torch.cuda.max_memory_allocated() / 1E6)
        torch_str = f"TORCH: (R: {torch_reserved_gpu:,}mb/" \
                    f"{torch_reserved_max:,}mb, " \
                    f"A: {torch_used_gpu:,}mb/{torch_max_used_gpu:,}mb)"
    except AssertionError:
        pass
    cpu_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1E3 +
                     resource.getrusage(
                         resource.RUSAGE_CHILDREN).ru_maxrss / 1E3)
    cpu_vmem = psutil.virtual_memory()
    cpu_free = int(cpu_vmem.free / 1E6)
    return f"CPU: (maxrss: {cpu_maxrss:,}mb F: {cpu_free:,}mb) " \
           f"{gpu_str}" \
           f"{torch_str}"


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


class Resize():
    def __init__(self, is_resizing: bool, is_not_migrating: bool) -> None:
        if not is_resizing:
            self.resize = self.__no_op
            return

        if not is_not_migrating:
            self.resize = self.__migration
            dataset_path = os.path.split(args.dataset)
            self.__directory = os.path.join(
                dataset_path[0],
                f'{dataset_path[1]}_cropped'
            )
            os.makedirs(self.__directory, exist_ok=True)
            return print(f"Resizing: Performing migration to '{self.__directory}'.")

        self.resize = self.__no_migration

    def __no_migration(self, image_path: str, w: int, h: int) -> Image:
        return ImageOps.fit(
            Image.open(image_path),
            (w, h),
            bleed=0.0,
            centering=(0.5, 0.5),
            method=Image.Resampling.LANCZOS
        ).convert(mode='RGB')

    def __migration(self, image_path: str, w: int, h: int) -> Image:
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
                os.path.join(args.dataset, f'{filename}.txt'),
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

    def __no_op(self, image_path: str, w: int, h: int) -> Image:
        return Image.open(image_path)


class ImageStore:
    def __init__(self, data_dir: str) -> None:
        print('ImageStore: Initializing')
        self.data_dir = data_dir
        self.resizer = Resize(args.resize, args.no_migration).resize
        self.validator = Validation(args.skip_validation, args.extended_validation).validate

        print('ImageStore: Loading images from ' + self.data_dir)
        self.image_files = []
        [self.image_files.extend(glob.glob(f'{data_dir}' + '/i*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]

        print('ImageStore: Loading masks from ' + self.data_dir)
        self.mask_files = []
        [self.mask_files.extend(glob.glob(f'{data_dir}' + '/m*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]

        self.clean()

    def __len__(self) -> int:
        return len(self.image_files)

    # iterator returns images as PIL images and their index in the store
    # def entries_iterator(self) -> Generator[Tuple[Image, int], None, None]:
    #     for f in range(len(self)):
    #         yield Image.open(self.image_files[f]), f

    # get image by index
    def get_image_and_mask(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        msk = Image.open(self.mask_files[idx]).convert('RGB')
        return img, msk

    # gets caption by removing the extension from the filename and replacing it with .txt
    def get_caption(self, idx) -> str:
        # filename = re.sub('\.[^/.]+$', '', self.image_files[idx]) + '.txt'
        # with open(filename, 'r', encoding='UTF-8') as f:
        #     return f.read()
        return 'architectural render with people ' + re.sub('\.[^/.]+$', '', self.image_files[idx])

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
                clean_images.append(image_dict[img_num])
                clean_masks.append(mask_dict[img_num])


        self.image_files = clean_images
        self.mask_files = clean_masks


class InpaintDataset(torch.utils.data.Dataset):
    def __init__(self, store: ImageStore, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: torch.device,
                 ucg: float = 0.1):
        self.store = store
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.ucg = ucg

        if type(self.text_encoder) is torch.nn.parallel.DistributedDataParallel:
            self.text_encoder = self.text_encoder.module

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
        return_dict['image_pixel_values'] = self.transforms(image).to(self.device)

        torch.set_rng_state(state)
        transformed_mask = self.transforms(mask).to(self.device)
        masks_sum = transformed_mask.sum(axis=0)
        binarized_mask = torch.where(masks_sum > 0, 1.0, 0.).unsqueeze(0)

        return_dict['mask_pixel_values'] = binarized_mask

        torch.set_rng_state(state)

        # return_dict['masked_image_pixel_values'] = self.transforms(numpy).to(self.device)
        return_dict['masked_image_pixel_values'] = generate_masked_image(return_dict['image_pixel_values'], return_dict['mask_pixel_values'])


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
        names = [example["input_ids"] for example in examples]

        pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
        mask_values = torch.stack(mask_values).to(memory_format=torch.contiguous_format).float()
        masked_image_values = torch.stack(masked_image_values).to(memory_format=torch.contiguous_format).float()

        if args.extended_mode_chunks < 2:
            max_length = self.tokenizer.model_max_length - 2
            input_ids = [self.tokenizer([example['input_ids']], truncation=True, return_length=True,
                                        return_overflowing_tokens=False, padding=False, add_special_tokens=False,
                                        max_length=max_length).input_ids for example in examples if example is not None]
        else:
            max_length = self.tokenizer.model_max_length
            max_chunks = args.extended_mode_chunks
            input_ids = [self.tokenizer([example['input_ids']], truncation=True, return_length=True,
                                        return_overflowing_tokens=False, padding=False, add_special_tokens=False,
                                        max_length=(max_length * max_chunks) - (max_chunks * 2)).input_ids[0] for
                         example in examples if example is not None]

        tokens = input_ids

        if args.extended_mode_chunks < 2:
            for i, x in enumerate(input_ids):
                for j, y in enumerate(x):
                    input_ids[i][j] = [self.tokenizer.bos_token_id, *y,
                                       *np.full((self.tokenizer.model_max_length - len(y) - 1),
                                                self.tokenizer.eos_token_id)]

            if args.clip_penultimate:
                input_ids = [self.text_encoder.text_model.final_layer_norm(
                    self.text_encoder(torch.asarray(input_id).to(self.device), output_hidden_states=True)[
                        'hidden_states'][-2])[0] for input_id in input_ids]
            else:
                input_ids = [self.text_encoder(torch.asarray(input_id).to(self.device),
                                               output_hidden_states=True).last_hidden_state[0] for input_id in
                             input_ids]
        else:
            max_standard_tokens = max_length - 2
            max_chunks = args.extended_mode_chunks
            max_len = np.ceil(max(len(x) for x in input_ids) / max_standard_tokens).astype(
                int).item() * max_standard_tokens
            if max_len > max_standard_tokens:
                z = None
                for i, x in enumerate(input_ids):
                    if len(x) < max_len:
                        input_ids[i] = [*x, *np.full((max_len - len(x)), self.tokenizer.eos_token_id)]
                batch_t = torch.tensor(input_ids)
                chunks = [batch_t[:, i:i + max_standard_tokens] for i in range(0, max_len, max_standard_tokens)]
                for chunk in chunks:
                    chunk = torch.cat((torch.full((chunk.shape[0], 1), self.tokenizer.bos_token_id), chunk,
                                       torch.full((chunk.shape[0], 1), self.tokenizer.eos_token_id)), 1)
                    if z is None:
                        if args.clip_penultimate:
                            z = self.text_encoder.text_model.final_layer_norm(
                                self.text_encoder(chunk.to(self.device), output_hidden_states=True)['hidden_states'][
                                    -2])
                        else:
                            z = self.text_encoder(chunk.to(self.device), output_hidden_states=True).last_hidden_state
                    else:
                        if args.clip_penultimate:
                            z = torch.cat((z, self.text_encoder.text_model.final_layer_norm(
                                self.text_encoder(chunk.to(self.device), output_hidden_states=True)['hidden_states'][
                                    -2])), dim=-2)
                        else:
                            z = torch.cat((z, self.text_encoder(chunk.to(self.device),
                                                                output_hidden_states=True).last_hidden_state), dim=-2)
                input_ids = z
            else:
                for i, x in enumerate(input_ids):
                    input_ids[i] = [self.tokenizer.bos_token_id, *x,
                                    *np.full((self.tokenizer.model_max_length - len(x) - 1),
                                             self.tokenizer.eos_token_id)]
                if args.clip_penultimate:
                    input_ids = self.text_encoder.text_model.final_layer_norm(
                        self.text_encoder(torch.asarray(input_ids).to(self.device), output_hidden_states=True)[
                            'hidden_states'][-2])
                else:
                    input_ids = self.text_encoder(torch.asarray(input_ids).to(self.device),
                                                  output_hidden_states=True).last_hidden_state
        input_ids = torch.stack(tuple(input_ids))

        batch = {
            'names': names,
            "input_ids": input_ids,
            'tokens': tokens,
            "image_pixel_values": pixel_values,
            "mask_pixel_values": mask_values,
            "masked_image_pixel_values": masked_image_values
        }

        return batch


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    # From CompVis LitEMA implementation
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

        del self.collected_params
        gc.collect()

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]


def main():
    rank = get_rank()
    world_size = get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)

        mode = 'disabled'
        if args.enablewandb:
            mode = 'online'
        if args.hf_token is not None:
            os.environ['HF_API_TOKEN'] = args.hf_token
            args.hf_token = None
        run = wandb.init(project=args.project_id, name=args.run_name, config=vars(args),
                         dir=args.output_path + '/wandb', mode=mode)

        # Inform the user of host, and various versions -- useful for debugging issues.
        print("RUN_NAME:", args.run_name)
        print("HOST:", socket.gethostname())
        print("CUDA:", torch.version.cuda)
        print("TORCH:", torch.__version__)
        print("TRANSFORMERS:", transformers.__version__)
        print("DIFFUSERS:", diffusers.__version__)
        print("MODEL:", args.model)
        print("FP16:", args.fp16)
        print("RESOLUTION:", args.resolution)

    if args.hf_token is not None:
        print(
            'It is recommended to set the HF_API_TOKEN environment variable instead of passing it as a command line argument since WandB will automatically log it.')
    else:
        try:
            args.hf_token = os.environ['HF_API_TOKEN']
            print("HF Token set via enviroment variable")
        except Exception:
            print("No HF Token detected in arguments or enviroment variable, setting it to none (as in string)")
            args.hf_token = "none"

    device = torch.device('cuda')

    print("DEVICE:", device)

    # setup fp16 stuff
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print('RANDOM SEED:', args.seed)

    if args.resume:
        args.model = args.resume

    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder='tokenizer', use_auth_token=args.hf_token)
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder='text_encoder', use_auth_token=args.hf_token)
    vae = AutoencoderKL.from_pretrained(args.model, subfolder='vae', use_auth_token=args.hf_token)
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder='unet', use_auth_token=args.hf_token)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.use_xformers:
        unet.set_use_memory_efficient_attention_xformers(True)

    # "The “safer” approach would be to move the model to the device first and create the optimizer afterwards."
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    # move models to device
    vae = vae.to(device, dtype=weight_dtype)
    unet = unet.to(device, dtype=torch.float32)
    text_encoder = text_encoder.to(device, dtype=weight_dtype if not args.train_text_encoder else torch.float32)

    unet = torch.nn.parallel.DistributedDataParallel(
        unet,
        device_ids=[rank],
        output_device=rank,
        gradient_as_bucket_view=True
    )

    if args.train_text_encoder:
        text_encoder = torch.nn.parallel.DistributedDataParallel(
            text_encoder,
            device_ids=[rank],
            output_device=rank,
            gradient_as_bucket_view=True
        )

    if args.use_8bit_adam:  # Bits and bytes is only supported on certain CUDA setups, so default to regular adam if it fails.
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except:
            print('bitsandbytes not supported, using regular Adam optimizer')
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    """
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )
    """

    optimizer_parameters = unet.parameters() if not args.train_text_encoder else itertools.chain(unet.parameters(),
                                                                                                 text_encoder.parameters())

    # Create distributed optimizer
    from torch.distributed.optim import ZeroRedundancyOptimizer
    optimizer = ZeroRedundancyOptimizer(
        optimizer_parameters,
        optimizer_class=optimizer_cls,
        parameters_as_bucket_view=True,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model,
        subfolder='scheduler',
        use_auth_token=args.hf_token,
    )

    # load dataset
    # TODO: add mask support here
    store = ImageStore(args.dataset)
    dataset = InpaintDataset(store, tokenizer, text_encoder, device, ucg=args.ucg)

    print(f'DATASET SIZE: {len(store)}')

    # TODO: add mask support here (in collate_fn)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

    # Migrate dataset
    if args.resize and not args.no_migration:
        for _, batch in enumerate(train_dataloader):
            continue
        print(
            f"Completed resize and migration to '{args.dataset}_cropped' please relaunch the trainer without the --resize argument and train on the migrated dataset.")
        exit(0)

    # create ema
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())

    print(get_gpu_ram())

    num_steps_per_epoch = len(train_dataloader)
    progress_bar = tqdm.tqdm(range(args.epochs * num_steps_per_epoch), desc="Total Steps", leave=False)
    global_step = 0

    if args.resume:
        target_global_step = int(args.resume.split('_')[-1])
        print(f'resuming from {args.resume}...')

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(args.lr_scheduler_warmup * num_steps_per_epoch * args.epochs),
        num_training_steps=args.epochs * num_steps_per_epoch,
        # last_epoch=(global_step // num_steps_per_epoch) - 1,
    )

    def save_checkpoint(global_step):
        if rank == 0:
            if args.use_ema:
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            pipeline = StableDiffusionPipeline(
                text_encoder=text_encoder if type(
                    text_encoder) is not torch.nn.parallel.DistributedDataParallel else text_encoder.module,
                vae=vae,
                unet=unet.module,
                tokenizer=tokenizer,
                scheduler=PNDMScheduler.from_pretrained(args.model, subfolder="scheduler",
                                                        use_auth_token=args.hf_token),
                safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )
            print(f'saving checkpoint to: {args.output_path}/{args.run_name}_{global_step}')
            pipeline.save_pretrained(f'{args.output_path}/{args.run_name}_{global_step}')

            if args.use_ema:
                ema_unet.restore(unet.parameters())

    # train!
    try:
        loss = torch.tensor(0.0, device=device, dtype=weight_dtype)
        print('Initiating training...')
        for epoch in range(args.epochs):
            print(f'Epoch {epoch + 1}/{args.epochs}')
            unet.train()
            if args.train_text_encoder:
                text_encoder.train()
            for _, batch in enumerate(train_dataloader):
                print(f'Batch {_ + 1}/{num_steps_per_epoch}')
                if args.resume and global_step < target_global_step:
                    if rank == 0:
                        progress_bar.update(1)
                    global_step += 1
                    continue
                print(batch['names'])
                b_start = time.perf_counter()
                # TODO: do we need with_torch_no_grad here?
                with torch.no_grad():
                    latent_dist = vae.encode(batch["image_pixel_values"].to(dtype=weight_dtype)).latent_dist
                    masked_latent_dist = vae.encode(batch["masked_image_pixel_values"].to(dtype=weight_dtype)).latent_dist
                    latents = latent_dist.sample() * 0.18215
                    masked_image_latents = masked_latent_dist.sample() * 0.18215
                    mask = interpolate(batch["mask_pixel_values"], scale_factor=1 / 8)


                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                latent_model_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)

                # Get the embedding for conditioning
                encoder_hidden_states = batch['input_ids']

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                if not args.train_text_encoder:
                    with unet.join():
                        # Predict the noise residual and compute loss
                        with torch.autocast('cuda', enabled=args.fp16):
                            # NOTE: this is where the new additional channels added are used
                            noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                        # backprop and update
                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                else:
                    with unet.join(), text_encoder.join():
                        # Predict the noise residual and compute loss
                        with torch.autocast('cuda', enabled=args.fp16):
                            noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                        # backprop and update
                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        # Update EMA
                if args.use_ema:
                    ema_unet.step(unet.parameters())

                # perf
                b_end = time.perf_counter()
                seconds_per_step = b_end - b_start
                steps_per_second = 1 / seconds_per_step
                rank_images_per_second = args.batch_size * steps_per_second
                world_images_per_second = rank_images_per_second * world_size
                samples_seen = global_step * args.batch_size * world_size

                # get global loss for logging
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss = loss / world_size

                if rank == 0:
                    progress_bar.update(1)
                    global_step += 1
                    logs = {
                        "train/loss": loss.detach().item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/samples_seen": samples_seen,
                        "perf/rank_samples_per_second": rank_images_per_second,
                        "perf/global_samples_per_second": world_images_per_second,
                    }
                    progress_bar.set_postfix(logs)
                    run.log(logs, step=global_step)

                if global_step % args.save_steps == 0 and global_step > 0:
                    save_checkpoint(global_step)
                if args.enableinference:
                    if global_step % args.image_log_steps == 0 and global_step > 0:
                        if rank == 0:
                            # get prompt from random batch
                            prompt = tokenizer.decode(batch['tokens'][random.randint(0, len(batch['tokens']) - 1)])

                            if args.image_log_scheduler == 'DDIMScheduler':
                                print('using DDIMScheduler scheduler')
                                scheduler = DDIMScheduler.from_pretrained(args.model, subfolder="scheduler",
                                                                          use_auth_token=args.hf_token)
                            else:
                                print('using PNDMScheduler scheduler')
                                scheduler = PNDMScheduler.from_pretrained(args.model, subfolder="scheduler",
                                                                          use_auth_token=args.hf_token)

                            pipeline = StableDiffusionPipeline(
                                text_encoder=text_encoder if type(
                                    text_encoder) is not torch.nn.parallel.DistributedDataParallel else text_encoder.module,
                                vae=vae,
                                unet=unet.module,
                                tokenizer=tokenizer,
                                scheduler=scheduler,
                                safety_checker=None,  # disable safety checker to save memory
                                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
                            ).to(device)
                            # inference
                            if args.enablewandb:
                                images = []
                            else:
                                saveInferencePath = args.output_path + "/inference"
                                os.makedirs(saveInferencePath, exist_ok=True)
                            with torch.no_grad():
                                with torch.autocast('cuda', enabled=args.fp16):
                                    for _ in range(args.image_log_amount):
                                        if args.enablewandb:
                                            images.append(
                                                wandb.Image(pipeline(
                                                    prompt, num_inference_steps=args.image_log_inference_steps
                                                ).images[0],
                                                            caption=prompt)
                                            )
                                        else:
                                            from datetime import datetime
                                            images = \
                                            pipeline(prompt, num_inference_steps=args.image_log_inference_steps).images[
                                                0]
                                            filenameImg = str(time.time_ns()) + ".png"
                                            filenameTxt = str(time.time_ns()) + ".txt"
                                            images.save(saveInferencePath + "/" + filenameImg)
                                            with open(saveInferencePath + "/" + filenameTxt, 'a') as f:
                                                f.write('Used prompt: ' + prompt + '\n')
                                                f.write('Generated Image Filename: ' + filenameImg + '\n')
                                                f.write('Generated at: ' + str(global_step) + ' steps' + '\n')
                                                f.write('Generated at: ' + str(
                                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '\n')

                            # log images under single caption
                            if args.enablewandb:
                                run.log({'images': images}, step=global_step)

                            # cleanup so we don't run out of memory
                            del pipeline
                            gc.collect()
    except Exception as e:
        print(
            f'Exception caught on rank {rank} at step {global_step}, saving checkpoint...\n{e}\n{traceback.format_exc()}')
        pass

    save_checkpoint(global_step)

    cleanup()

    print(get_gpu_ram())
    print('Done!')


if __name__ == "__main__":
    setup()
    main()
