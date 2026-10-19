import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = ('.jpeg', '.jpg', '.png', '.tif', '.tiff')
MAX_CELL_BOX_SIZE_PX = 100.0

class FourCropDataset(Dataset):
    """Inference only: fixed 1024 resize and four 512 crops, no augmentation."""
    def __init__(self, records, input_size=512, resize_size=1024,
                 augment=False, random_crop=False, four_crop=True, cache_images=False):
        if (input_size, resize_size, augment, random_crop, four_crop, cache_images) != (512, 1024, False, False, True, False):
            raise ValueError('Only fixed 1024→4×512 evaluation is supported')
        self.records = list(records)
        self.input_size, self.resize_size = input_size, resize_size

    def __len__(self):
        return 4 * len(self.records)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        record_index, quadrant = divmod(index, 4)
        image, labels = self._load_resized_pair(record_index)
        image, labels = self._crop(image, labels, quadrant)
        return (torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                torch.from_numpy(labels[:, 0].copy()),
                torch.from_numpy(labels[:, 1:5].copy()), torch.zeros(len(labels)))

    @staticmethod
    def _mediauto_box(cell, image_width, image_height):
        coordinates = cell.get('local_coordinates')
        if coordinates:
            points = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
            (x0, y0) = points.min(axis=0)
            (x1, y1) = points.max(axis=0)
        elif 'local_x' in cell and 'local_y' in cell:
            bbox = cell.get('bbox', {})
            width = float(bbox.get('width', 0.0))
            height = float(bbox.get('height', 0.0))
            (cx, cy) = (float(cell['local_x']), float(cell['local_y']))
            (x0, y0, x1, y1) = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
        else:
            return None
        if x1 - x0 > MAX_CELL_BOX_SIZE_PX or y1 - y0 > MAX_CELL_BOX_SIZE_PX:
            return None
        (x0, x1) = np.clip([x0, x1], 0, image_width)
        (y0, y1) = np.clip([y0, y1], 0, image_height)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _load_resized_pair(self, index):
        (image_path, label_path) = self.records[index]
        with Image.open(image_path) as pil_image:
            (image_width, image_height) = pil_image.size
            image = np.asarray(pil_image.convert('RGB'))
        labels = self.load_labels(label_path, image_size=(image_width, image_height))
        if image.shape[:2] != (self.resize_size, self.resize_size):
            interpolation = cv2.INTER_AREA if max(image.shape[:2]) > self.resize_size else cv2.INTER_LINEAR
            image = cv2.resize(image, (self.resize_size, self.resize_size), interpolation=interpolation)
        return (np.ascontiguousarray(image), labels)

    def _crop(self, image, labels, tile_index=None):
        left, top = ((0, 0), (512, 0), (0, 512), (512, 512))[tile_index]
        (right, bottom) = (left + self.input_size, top + self.input_size)
        cropped = image[top:bottom, left:right]
        if not len(labels):
            return (cropped, labels)
        boxes = labels[:, 1:5].copy()
        cx = boxes[:, 0] * self.resize_size
        cy = boxes[:, 1] * self.resize_size
        width = boxes[:, 2] * self.resize_size
        height = boxes[:, 3] * self.resize_size
        keep = (cx >= left) & (cx < right) & (cy >= top) & (cy < bottom)
        labels = labels[keep].copy()
        if not len(labels):
            return (cropped, labels)
        (cx, cy, width, height) = (cx[keep], cy[keep], width[keep], height[keep])
        x0 = np.clip(cx - width / 2, left, right) - left
        y0 = np.clip(cy - height / 2, top, bottom) - top
        x1 = np.clip(cx + width / 2, left, right) - left
        y1 = np.clip(cy + height / 2, top, bottom) - top
        labels[:, 1] = (x0 + x1) / (2 * self.input_size)
        labels[:, 2] = (y0 + y1) / (2 * self.input_size)
        labels[:, 3] = (x1 - x0) / self.input_size
        labels[:, 4] = (y1 - y0) / self.input_size
        return (cropped, labels)
