from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import os

class TinyImageNetFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        train:bool = True,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, train = train)

        self.loader = loader
        self.extensions = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

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
            _, class_to_idx = find_classes(directory)
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
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
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
            
        target_dir = os.path.join(directory, 'val', 'images')
        for entry in os.scandir(target_dir):
            path = os.path.join(target_dir, entry.name)
            if is_valid_file(path):
                target_class = val_ann[entry.name]
                item = path, class_to_idx[target_class]
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
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

