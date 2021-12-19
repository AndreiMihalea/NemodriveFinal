import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

__all__ = ['SegmentationDataset']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size_w=832, base_size_h=256, crop_size_w=640,
                 crop_size_h=256):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size_w = base_size_w
        self.base_size_h = base_size_h
        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

    def _val_sync_transform(self, img, mask, path_mask):
        outsize_w = self.crop_size_w
        outsize_h = self.crop_size_h
        short_size_w, short_size_h = outsize_w, outsize_h
        # if w > h:
        #     oh = short_size
        #     ow = int(1.0 * w * oh / h)
        # else:
        #     ow = short_size
        #     oh = int(1.0 * h * ow / w)
        img = img.resize((short_size_w, short_size_h), Image.BILINEAR)
        mask = mask.resize((short_size_w, short_size_h), Image.NEAREST)
        path_mask = path_mask.resize((short_size_w, short_size_h), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize_w) / 2.))
        y1 = int(round((h - outsize_h) / 2.))
        img = img.crop((x1, y1, x1 + outsize_w, y1 + outsize_h))
        mask = mask.crop((x1, y1, x1 + outsize_w, y1 + outsize_h))
        path_mask = path_mask.crop((x1, y1, x1 + outsize_w, y1 + outsize_h))
        # final transform
        img, mask, path_mask = self._img_transform(img), self._mask_transform(mask), self._img_transform(path_mask)
        return img, mask, path_mask

    def _sync_transform(self, img, mask, path_mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            path_mask = path_mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size_w = self.crop_size_w
        crop_size_h = self.crop_size_h
        # random scale (short edge)
        w, h = img.size
        random_scale = random.randint(int(self.base_size_w * 0.5), int(self.base_size_w * 2.0))
        short_size_w = random_scale
        short_size_h = int(random_scale * h / w)
        # if h > w:
        #     ow = short_size
        #     oh = int(1.0 * h * ow / w)
        # else:
        #     oh = short_size
        #     ow = int(1.0 * w * oh / h)
        img = img.resize((short_size_w, short_size_h), Image.BILINEAR)
        mask = mask.resize((short_size_w, short_size_h), Image.NEAREST)
        path_mask = path_mask.resize((short_size_w, short_size_h), Image.NEAREST)
        # pad crop
        if short_size_w < crop_size_w:
            padw = crop_size_w - short_size_w if short_size_w < crop_size_w else 0
            img = ImageOps.expand(img, border=(0, 0, padw, 0), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, 0), fill=0)
            path_mask = ImageOps.expand(path_mask, border=(0, 0, padw, 0), fill=0)
        if short_size_h < crop_size_h:
            padh = crop_size_h - short_size_h if short_size_h < crop_size_h else 0
            img = ImageOps.expand(img, border=(0, 0, 0, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, 0, padh), fill=0)
            path_mask = ImageOps.expand(path_mask, border=(0, 0, 0, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size_w)
        y1 = random.randint(0, h - crop_size_h)
        img = img.crop((x1, y1, x1 + crop_size_w, y1 + crop_size_h))
        mask = mask.crop((x1, y1, x1 + crop_size_w, y1 + crop_size_h))
        path_mask = path_mask.crop((x1, y1, x1 + crop_size_w, y1 + crop_size_h))
        # random rotation
        if random.random() < 0.5:
            random_rot = random.randint(-20, 20)
            img = img.rotate(random_rot, Image.BILINEAR)
            mask = mask.rotate(random_rot, Image.NEAREST)
            path_mask = path_mask.rotate(random_rot, Image.NEAREST)

        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask, path_mask = self._img_transform(img), self._mask_transform(mask), self._img_transform(path_mask)
        return img, mask, path_mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    def _path_mask_transform(self, img):
        return np.array(img).astype('float')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
