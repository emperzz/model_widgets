from torch.utils.data import DataLoader
import numpy as np
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import torchvision.transforms as transforms
import os
import sys

def _adjust_entry_name(name):
    if sys.platform == 'win32':
        return name[2:] if '._' == name[:2] else name
    return name

class TinyImageNetFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        train:bool = True,
        target_label:bool = True,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, train = train)

        self.loader = loader
        self.extensions = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
        self.target_label = target_label

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        
        self.imgs = self.samples
        
    def find_classes(self, directory):
        with open(os.path.join(self.root, 'wnids.txt'), 'r') as fo:
            classes = sorted([c.strip('\n') for c in fo.readlines()])
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
        
    @staticmethod
    def class_to_boxes(root, class_name = None):
        if class_name is None:
            class_name = 'val_annotations.txt'
        
        with open(os.path.join(root, class_name), 'r') as fo:
            data = fo.readlines()
            
        return {
            c.split('\t')[0] : list(map(lambda x : int(x), c.split('\t')[-4:]))
            for c in data
        }

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        train: bool,
    ) -> List[Tuple[str, int]]:
        
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
            
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        # both_none = extensions is None and is_valid_file is None
        # both_something = extensions is not None and is_valid_file is not None
        # if both_none or both_something:
        #     raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")


        def is_valid_file(x: str) -> bool:
            return x.lower().endswith((".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"))

        if train:
            instances = self._make_train_dataset(directory, class_to_idx, is_valid_file)
        else:
            instances = self._make_val_dataset(directory, class_to_idx, is_valid_file)
        return instances
    
    def _make_train_dataset(
        self,
        directory,
        class_to_idx,
        is_valid_file
    ):
        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, 'train', target_class)
            
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    fname = _adjust_entry_name(fname)
                    path = os.path.join(root, fname)
                    if 'boxes' in fname:
                        class_to_boxes = self.class_to_boxes(root, fname)

                    if is_valid_file(path):
                        # 先转化为array，list在dataloader会输出成len([...tensor(batch_size) ...]) = 4(框的四角)
                        item = path, (class_index, np.array(class_to_boxes[fname])) 
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if self.extensions is not None:
                msg += f"Supported extensions are: {self.extensions if isinstance(self.extensions, str) else ', '.join(self.extensions)}"
            raise FileNotFoundError(msg)
            
        return instances
            
    def _make_val_dataset(
        self,
        directory,
        class_to_idx,
        is_valid_file
    ):
        instances = []
        available_classes = set()
        
        with open(os.path.join(directory, 'val', 'val_annotations.txt'), 'r') as fo:
            val_ann ={c.split('\t')[0]:c.split('\t')[1] for c in fo.readlines()}
        class_to_boxes = self.class_to_boxes(os.path.join(directory, 'val'))

        target_dir = os.path.join(directory, 'val', 'images')
        for entry in os.scandir(target_dir):
            fname = _adjust_entry_name(entry.name)
            path = os.path.join(target_dir, fname)
            if is_valid_file(path):
                target_class = val_ann[_adjust_entry_name(entry.name)]
                item = path, (class_to_idx[target_class], np.array(class_to_boxes[fname]))
                instances.append(item)
                
                if target_class not in available_classes:
                    available_classes.add(target_class)
                    
        empty_classes = set(class_to_idx.keys()) - available_classes
        
        return instances
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        if self.target_label:
            target = target[0]
        else:
            target = target[1]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def get_tiny_imagenet_dataloader(
    root, 
    batch_size, 
    num_workers = 0,
    transform = transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor()]),
    target_transform = None):
    train_iter =\
        DataLoader(
            TinyImageNetFolder(root, train = True,
                transform = transform, target_transform= target_transform),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True)
    
    test_iter =\
        DataLoader(
            TinyImageNetFolder(root, train = False,
                transform = transform, target_transform= target_transform),
                batch_size=batch_size,
                num_workers=num_workers)

    return train_iter, test_iter

if __name__ == '__main__':
    train_iter, test_iter = get_tiny_imagenet_dataloader('E:/Data/tiny-imagenet-200', batch_size = 128, num_workers = 0)
    for x, y in train_iter:
        print(x.shape, y.shape)
        break