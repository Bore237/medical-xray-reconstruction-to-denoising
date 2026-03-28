from monai.data import CacheDataset, DataLoader
import matplotlib.pyplot as plt
from itertools import islice
import numpy as np
from tqdm import tqdm
import os
from typing import Any

class  Read_data():
    def __init__(self, root, transform, extension : str = 'png',
                batch_size : int = 16, num_workers : int = 0, 
                cache_rate : float = 0.01, title_loader : str = "Load data") -> None:

        self.root = root 
        self.transform = transform
        self.extension = extension
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.title_loader = title_loader
        self.paths_list = self._get_files(self.extension)
        self.files = []
        self.image_data = None

    def _get_files(self, extension : str = ".png") -> list:
        files = []
        for root, _, filenames in tqdm(os.walk(self.root), desc=self.title_loader):
            for name in filenames:
                if name.endswith(extension) and not name.startswith("._"):
                    files.append(os.path.join(root, name).replace("\\","/"))
        return files

    def __len__(self) -> int:
        return len(self.paths_list)

    def get_files(self) -> list:
        return self.paths_list
        
    def get_files_d(self, limit: Any = None) -> list:
        if limit == None:
            limit = len(self.paths_list)
        self.files = [{'image': f, 'label': f} for f in self.paths_list[:limit]]
        return self.files

    def load_dataset(self, start_idx:int =0, end_idx: Any = None) -> CacheDataset:
        if self.files == []:
            self.get_files_d()
        files_load = self.files
        if end_idx is not None:
            files_load = self.files[start_idx : end_idx]
        self.image_data = CacheDataset(data = files_load, transform = self.transform,
                                    cache_rate = self.cache_rate, num_workers = self.num_workers)
        
        return self.image_data

    def loader_dataset(self, batch_size : int = 0) -> DataLoader:
        if batch_size > 0:
            self.batch_size = batch_size
        if self.image_data is None:
            self.image_data = self.load_dataset()
        self.image_loader = DataLoader(dataset = self.image_data, batch_size= self.batch_size, 
                                shuffle= True, num_workers = self.num_workers)
        return self.image_loader

    def show_image(self, nrows : int = 2, ncols : int = 2, figsize: tuple= (8,  4), 
                batch: int = 0, dict_label: str = 'image', title:str = "Train Data Batch 0") -> None:
        fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
        display_data = next(islice(self.image_loader, batch, None))
        if dict_label:
            display_data = display_data[dict_label]
        for i in range(nrows):
            for j in range(ncols):
                idx = np.random.randint(0, len(display_data))
                axes[i, j].imshow(display_data[idx].squeeze(dim=0), cmap = 'gray')
                axes[i, j].axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
