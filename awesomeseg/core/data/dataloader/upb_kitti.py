"""Pascal VOC Semantic Segmentation Dataset."""
import os
import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from torchvision import transforms

from PIL import Image
from .segbase import SegmentationDataset


class UPBSegmentation(SegmentationDataset):
    """Pascal VOC Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/VOCdevkit'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = VOCSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'labels'
    NUM_CLASS = 1 # 1 for soft labels

    def __init__(self, root='/mnt/storage/workspace/andreim/nemodrive/upb_self_supervised_labels', split='train', mode=None, transform=None,
                 **kwargs):
        super(UPBSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        _voc_root = os.path.join(root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'HardLabels')#os.path.join(_voc_root, 'JPEGImages')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        _path_mask_dir = os.path.join(_voc_root, 'SoftLabels')#os.path.join(_voc_root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        if split == 'train':
            _split_f = os.path.join(_splits_dir, 'train')
        elif split == 'val':
            _split_f = os.path.join(_splits_dir, 'val') #'val_upb.txt'; with info has the files that are synched with the steering info files
        elif split == 'test':
            _split_f = os.path.join(_splits_dir, 'test_short')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        self.masks = []
        self.path_masks = []
        self.cmds = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                file_name = line.split(',')[0]
                cmd = line.split(',')[1]
                _image = os.path.join(_image_dir, file_name)
                assert os.path.isfile(_image)
                _path_mask = os.path.join(_path_mask_dir, file_name.replace('/', '\\')) # doar filename pt eval
                assert os.path.isfile(_path_mask)
                self.images.append(_image)
                self.path_masks.append(_path_mask)
                self.cmds.append(cmd)
                _mask = os.path.join(_mask_dir, file_name.replace('/', '\\')) # doar filename pt eval
                assert os.path.isfile(_mask)
                self.masks.append(_mask)

        if split != 'test':
            assert (len(self.images) == len(self.masks))
        print('Found {} images in the folder {}'.format(len(self.images), _voc_root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        # print(self.cmds[index])
        # img.show()
        # time.sleep(8)
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index]).quantize(self.num_class + 1) # 1 for train or 2 for eval
        path_mask = Image.open(self.path_masks[index]).convert('RGB')
        # path_mask = np.load(self.path_masks[index], allow_pickle=True)
        # path_mask = Image.fromarray(path_mask)
        # mask.show()
        # time.sleep(5)
        # synchronized transform
        if self.mode == 'train':
            img, mask, path_mask = self._sync_transform(img, mask, path_mask)
        elif self.mode == 'val':
            img, mask, path_mask = self._val_sync_transform(img, mask, path_mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
            path_mask = transforms.ToTensor()(path_mask)

        path_mask = path_mask[1].unsqueeze(0)

        if path_mask.max() != 0:
            path_mask = path_mask / path_mask.max()

        return img, mask, path_mask, self.images[index]# os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category names."""
        return ('path', 'rest')


class UPBImageSampler(Sampler):
    def __init__(self, image_data, prob_weights):
        self.image_data = image_data

        # Get dataset length in terms of video frames and start frame for each video
        self.start_frames = []
        self.len = len(image_data)

        self.seen = 0
        self.samples_cmd = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        self.samples_idx = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self._population = [0, 1, 2, 3, 4, 5]
        self._weights = prob_weights
        self._split_samples()

        for key in self.samples_cmd.keys():
            np.random.shuffle(self.samples_cmd[key])

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        # added this while because samples_cmd[sample_type] could be empty
        while True:
            sample_type = np.random.choice(self._population, p=self._weights)
            if self.samples_cmd[sample_type]:
                break
        idx = self.samples_cmd[sample_type][self.samples_idx[sample_type]]

        self.samples_idx[sample_type] += 1
        if self.samples_idx[sample_type] >= len(self.samples_cmd[sample_type]):
            self.samples_idx[sample_type] = 0
            np.random.shuffle(self.samples_cmd[sample_type])

        self.seen += 1
        if self.seen >= self.len:
            for key in self.samples_cmd.keys():
                np.random.shuffle(self.samples_cmd[key])
                self.samples_idx[key] = 0
            self.seen = 0
            raise StopIteration

        return idx

    def _split_samples(self):
        index = 0
        for j in range(len(self.image_data)):
            cmd = self.image_data[j]
            self.samples_cmd[cmd].append(index)
            index += 1


if __name__ == '__main__':
    dataset = UPBSegmentation()
