import csv
import gc
import io
import json
import math
import os
import random
from contextlib import contextmanager, nullcontext
from random import shuffle
from threading import Thread

import albumentations
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

from .utils import (VIDEO_READER_TIMEOUT, Camera, VideoReader_contextmanager,
                    custom_meshgrid, get_random_mask, get_relative_pose,
                    get_video_reader_batch, padding_image, process_pose_file,
                    process_pose_params, ray_condition, resize_frame,
                    resize_image_with_target_area)


class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]


class ImageVideoDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        enable_inpaint=False,
        return_file_name=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        
        if data_info.get('type', 'image')=='video':
            video_id, text = data_info['file_path'], data_info['text']

            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
            return pixel_values, text, 'video', video_dir
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            if random.random() < self.text_drop_ratio:
                text = ''
            return image, text, 'image', image_path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type, file_path = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


def _compute_batch_index(
    num_frames,
    video_sample_n_frames,
    video_sample_stride,
    video_length_drop_start,
    video_length_drop_end,
):
    min_sample_n_frames = min(
        video_sample_n_frames,
        int(num_frames * (video_length_drop_end - video_length_drop_start) // video_sample_stride),
    )
    if min_sample_n_frames == 0:
        raise ValueError("No Frames in video.")

    video_length = int(video_length_drop_end * num_frames)
    clip_length = min(video_length, (min_sample_n_frames - 1) * video_sample_stride + 1)
    start_idx = (
        random.randint(int(video_length_drop_start * video_length), video_length - clip_length)
        if video_length != clip_length
        else 0
    )
    return np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)


class ImageVideoControlDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.1, 
        video_length_drop_end=0.9,
        enable_inpaint=False,
        enable_camera_info=False,
        return_file_name=False,
        enable_subject_info=False,
        padding_subject_info=True,
        align_frames_to_control=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.enable_camera_info = enable_camera_info
        self.enable_subject_info = enable_subject_info
        self.padding_subject_info = padding_subject_info

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        self.align_frames_to_control = align_frames_to_control

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        if self.enable_camera_info:
            self.video_transforms_camera = transforms.Compose(
                [
                    transforms.Resize(min(self.video_sample_size)),
                    transforms.CenterCrop(self.video_sample_size)
                ]
            )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))
    
    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        video_id, text = data_info['file_path'], data_info['text']

        if data_info.get('type', 'image')=='video':
            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            control_video_id = data_info.get('normal_file_path', data_info.get('control_file_path'))
            depth_video_id = data_info.get('depth_file_path')
            motion_video_id = data_info.get('motion_file_path', data_info.get('motion_vector_file_path'))
            mask_video_id = data_info.get('mask_file_path', data_info.get('alpha_file_path'))
            uv_video_id = data_info.get('uv_file_path')

            def _resolve_path(path):
                if path is None:
                    return None
                if self.data_root is None:
                    return path
                return os.path.join(self.data_root, path)

            control_video_path = _resolve_path(control_video_id)
            depth_video_path = _resolve_path(depth_video_id)
            motion_video_path = _resolve_path(motion_video_id)
            mask_video_path = _resolve_path(mask_video_id)
            uv_video_path = _resolve_path(uv_video_id)

            control_video_id = control_video_path
            control_pixel_values = None
            depth_pixel_values = None
            motion_pixel_values = None
            gbuffer_mask_pixel_values = None
            uv_pixel_values = None
            control_camera_values = None
            gbuffer_paths = {
                "depth": depth_video_path,
                "normal": control_video_path,
                "motion": motion_video_path,
                "mask": mask_video_path,
                "uv": uv_video_path,
            }

            def _read_video_values(video_path, batch_index, error_name):
                with VideoReader_contextmanager(video_path, num_threads=1) as local_video_reader:
                    try:
                        sample_args = (local_video_reader, batch_index)
                        video_values = func_timeout(
                            VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                        )
                        resized_frames = []
                        for i in range(len(video_values)):
                            frame = video_values[i]
                            resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                            resized_frames.append(resized_frame)
                        video_values = np.array(resized_frames)
                    except FunctionTimedOut:
                        raise ValueError(f"Read {idx} timeout.")
                    except Exception as e:
                        raise ValueError(f"Failed to extract frames from {error_name} video. Error is {e}.")

                    if not self.enable_bucket:
                        video_values = torch.from_numpy(video_values).permute(0, 3, 1, 2).contiguous()
                        video_values = video_values / 255.
                        video_values = self.video_transforms(video_values)
                    return video_values

            with VideoReader_contextmanager(video_dir, num_threads=1) as video_reader:
                if self.align_frames_to_control and control_video_path is not None:
                    depth_reader_context = (
                        VideoReader_contextmanager(depth_video_path, num_threads=1)
                        if depth_video_path is not None
                        else nullcontext(None)
                    )
                    with VideoReader_contextmanager(control_video_path, num_threads=1) as control_video_reader, depth_reader_context as depth_video_reader:
                        control_num_frames = len(control_video_reader)
                        if control_num_frames == 0:
                            raise ValueError("No Frames in control video.")
                        num_frames_for_sampling = min(len(video_reader), control_num_frames)
                        if depth_video_path is not None:
                            depth_num_frames = len(depth_video_reader)
                            if depth_num_frames == 0:
                                raise ValueError("No Frames in depth video.")
                            num_frames_for_sampling = min(num_frames_for_sampling, depth_num_frames)
                        for optional_path, optional_name in (
                            (motion_video_path, "motion"),
                            (mask_video_path, "mask"),
                            (uv_video_path, "uv"),
                        ):
                            if optional_path is None:
                                continue
                            with VideoReader_contextmanager(optional_path, num_threads=1) as optional_reader:
                                optional_num_frames = len(optional_reader)
                                if optional_num_frames == 0:
                                    raise ValueError(f"No Frames in {optional_name} video.")
                                num_frames_for_sampling = min(num_frames_for_sampling, optional_num_frames)

                        batch_index = _compute_batch_index(
                            num_frames_for_sampling,
                            self.video_sample_n_frames,
                            self.video_sample_stride,
                            self.video_length_drop_start,
                            self.video_length_drop_end,
                        )

                        try:
                            sample_args = (video_reader, batch_index)
                            pixel_values = func_timeout(
                                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                            )
                            resized_frames = []
                            for i in range(len(pixel_values)):
                                frame = pixel_values[i]
                                resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                                resized_frames.append(resized_frame)
                            pixel_values = np.array(resized_frames)

                            sample_args = (control_video_reader, batch_index)
                            control_pixel_values = func_timeout(
                                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                            )
                            resized_frames = []
                            for i in range(len(control_pixel_values)):
                                frame = control_pixel_values[i]
                                resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                                resized_frames.append(resized_frame)
                            control_pixel_values = np.array(resized_frames)

                            if depth_video_reader is not None:
                                sample_args = (depth_video_reader, batch_index)
                                depth_pixel_values = func_timeout(
                                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                                )
                                resized_frames = []
                                for i in range(len(depth_pixel_values)):
                                    frame = depth_pixel_values[i]
                                    resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                                    resized_frames.append(resized_frame)
                                depth_pixel_values = np.array(resized_frames)
                        except FunctionTimedOut:
                            raise ValueError(f"Read {idx} timeout.")
                        except Exception as e:
                            raise ValueError(f"Failed to extract frames from video. Error is {e}.")
                else:
                    num_frames_for_sampling = len(video_reader)
                    batch_index = _compute_batch_index(
                        num_frames_for_sampling,
                        self.video_sample_n_frames,
                        self.video_sample_stride,
                        self.video_length_drop_start,
                        self.video_length_drop_end,
                    )

                    try:
                        sample_args = (video_reader, batch_index)
                        pixel_values = func_timeout(
                            VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                        )
                        resized_frames = []
                        for i in range(len(pixel_values)):
                            frame = pixel_values[i]
                            resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                            resized_frames.append(resized_frame)
                        pixel_values = np.array(resized_frames)
                    except FunctionTimedOut:
                        raise ValueError(f"Read {idx} timeout.")
                    except Exception as e:
                        raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)

                if random.random() < self.text_drop_ratio:
                    text = ''

            if self.enable_camera_info:
                if control_video_id.lower().endswith('.txt'):
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0])
                        control_camera_values = torch.from_numpy(control_camera_values).permute(0, 3, 1, 2).contiguous()
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)
                        control_camera_values = self.video_transforms_camera(control_camera_values)
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0], return_poses=True)
                        control_camera_values = torch.from_numpy(np.array(control_camera_values)).unsqueeze(0).unsqueeze(0)
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)[0][0]
                        control_camera_values = np.array([control_camera_values[index] for index in batch_index])
                else:
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)
                        control_camera_values = None
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)
                        control_camera_values = None
            else:
                if control_video_id is not None and control_pixel_values is None:
                    with VideoReader_contextmanager(control_video_id, num_threads=1) as control_video_reader:
                        try:
                            sample_args = (control_video_reader, batch_index)
                            control_pixel_values = func_timeout(
                                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                            )
                            resized_frames = []
                            for i in range(len(control_pixel_values)):
                                frame = control_pixel_values[i]
                                resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                                resized_frames.append(resized_frame)
                            control_pixel_values = np.array(resized_frames)
                        except FunctionTimedOut:
                            raise ValueError(f"Read {idx} timeout.")
                        except Exception as e:
                            raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                        if not self.enable_bucket:
                            control_pixel_values = torch.from_numpy(control_pixel_values).permute(0, 3, 1, 2).contiguous()
                            control_pixel_values = control_pixel_values / 255.
                            del control_video_reader
                        else:
                            control_pixel_values = control_pixel_values

                        if not self.enable_bucket:
                            control_pixel_values = self.video_transforms(control_pixel_values)
                else:
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)

                if depth_video_path is not None and depth_pixel_values is None:
                    depth_pixel_values = _read_video_values(depth_video_path, batch_index, "depth")
                elif depth_video_path is None:
                    depth_pixel_values = control_pixel_values.copy() if self.enable_bucket else control_pixel_values.clone()

                if motion_video_path is not None:
                    motion_pixel_values = _read_video_values(motion_video_path, batch_index, "motion")
                if mask_video_path is not None:
                    gbuffer_mask_pixel_values = _read_video_values(mask_video_path, batch_index, "mask")
                if uv_video_path is not None:
                    uv_pixel_values = _read_video_values(uv_video_path, batch_index, "uv")
                control_camera_values = None

            gbuffer_pixel_values = {
                "depth": depth_pixel_values,
                "normal": control_pixel_values,
                "motion": motion_pixel_values,
                "mask": gbuffer_mask_pixel_values,
                "uv": uv_pixel_values,
            }
            gbuffer_available = {
                key: 1.0 if gbuffer_paths[key] is not None else 0.0
                for key in gbuffer_pixel_values
            }
            # Keep historical metadata trainable: depth used to fall back to normal.
            if depth_video_path is None:
                gbuffer_available["depth"] = 0.0
            if control_video_path is not None:
                gbuffer_available["normal"] = 1.0
            is_synthetic = float(
                str(data_info.get("domain", data_info.get("data_domain", ""))).lower()
                in {"synthetic", "render", "rendered", "sim", "simulation"}
                or any(gbuffer_available[key] > 0 for key in ("motion", "mask", "uv"))
            )
            
            if self.enable_subject_info:
                if not self.enable_bucket:
                    visual_height, visual_width = pixel_values.shape[-2:]
                else:
                    visual_height, visual_width = pixel_values.shape[1:3]

                subject_id = data_info.get('object_file_path', [])
                shuffle(subject_id)
                subject_images = []
                for i in range(min(len(subject_id), 4)):
                    subject_path = subject_id[i]
                    if self.data_root is not None:
                        subject_path = os.path.join(self.data_root, subject_path)
                    subject_image = Image.open(subject_path)
                    width, height = subject_image.size
                    total_pixels = width * height

                    if self.padding_subject_info:
                        img = padding_image(subject_image, visual_width, visual_height)
                    else:
                        img = resize_image_with_target_area(subject_image, 1024 * 1024)

                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    subject_images.append(np.array(img))
                if self.padding_subject_info:
                    subject_image = np.array(subject_images)
                else:
                    subject_image = subject_images
            else:
                subject_image = None

            return pixel_values, control_pixel_values, depth_pixel_values, subject_image, control_camera_values, text, "video", gbuffer_pixel_values, gbuffer_available, is_synthetic
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)

            if random.random() < self.text_drop_ratio:
                text = ''

            control_image_id = data_info['control_file_path']

            if self.data_root is None:
                control_image_id = control_image_id
            else:
                control_image_id = os.path.join(self.data_root, control_image_id)

            control_image = Image.open(control_image_id).convert('RGB')
            if not self.enable_bucket:
                control_image = self.image_transforms(control_image).unsqueeze(0)
            else:
                control_image = np.expand_dims(np.array(control_image), 0)
            
            if self.enable_subject_info:
                if not self.enable_bucket:
                    visual_height, visual_width = image.shape[-2:]
                else:
                    visual_height, visual_width = image.shape[1:3]

                subject_id = data_info.get('object_file_path', [])
                shuffle(subject_id)
                subject_images = []
                for i in range(min(len(subject_id), 4)):
                    subject_path = subject_id[i]
                    if self.data_root is not None:
                        subject_path = os.path.join(self.data_root, subject_path)
                    subject_image = Image.open(subject_path).convert('RGB')
                    width, height = subject_image.size
                    total_pixels = width * height

                    if self.padding_subject_info:
                        img = padding_image(subject_image, visual_width, visual_height)
                    else:
                        img = resize_image_with_target_area(subject_image, 1024 * 1024)

                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    subject_images.append(np.array(img))
                if self.padding_subject_info:
                    subject_image = np.array(subject_images)
                else:
                    subject_image = subject_images
            else:
                subject_image = None

            gbuffer_pixel_values = {
                "depth": control_image,
                "normal": control_image,
                "motion": None,
                "mask": None,
                "uv": None,
            }
            gbuffer_available = {"depth": 0.0, "normal": 1.0, "motion": 0.0, "mask": 0.0, "uv": 0.0}
            return image, control_image, control_image, subject_image, None, text, 'image', gbuffer_pixel_values, gbuffer_available, 0.0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, control_pixel_values, depth_pixel_values, subject_image, control_camera_values, name, data_type, gbuffer_pixel_values, gbuffer_available, is_synthetic = self.get_batch(idx)

                sample["pixel_values"] = pixel_values
                sample["control_pixel_values"] = control_pixel_values
                sample["depth_pixel_values"] = depth_pixel_values
                sample["gbuffer_pixel_values"] = gbuffer_pixel_values
                sample["gbuffer_available"] = gbuffer_available
                sample["is_synthetic"] = is_synthetic
                sample["subject_image"] = subject_image
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx

                if self.enable_camera_info:
                    sample["control_camera_values"] = control_camera_values

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class ImageVideoSafetensorsDataset(Dataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))

        self.data_root = data_root
        self.dataset = dataset
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.data_root is None:
            path = self.dataset[idx]["file_path"]
        else:
            path = os.path.join(self.data_root, self.dataset[idx]["file_path"])
        state_dict = load_file(path)
        return state_dict


class TextDataset(Dataset):
    def __init__(self, ann_path, text_drop_ratio=0.0):
        print(f"loading annotations from {ann_path} ...")
        with open(ann_path, 'r') as f:
            self.dataset = json.load(f)
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        self.text_drop_ratio = text_drop_ratio

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                item = self.dataset[idx]
                text = item['text']

                # Randomly drop text (for classifier-free guidance)
                if random.random() < self.text_drop_ratio:
                    text = ''

                sample = {
                    "text": text,
                    "idx": idx
                }
                return sample

            except Exception as e:
                print(f"Error at index {idx}: {e}, retrying with random index...")
                idx = np.random.randint(0, self.length - 1)
