# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader
from .av1_dataset import Av1Dataset, collate_fn


class ArgoverseV1DataModule(LightningDataModule):

    def __init__(self,
                 data_root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 test: bool = False,
                 dataset: dict = {}) -> None:
        super(ArgoverseV1DataModule, self).__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers   
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.dataset_cfg = dataset
        self.test = test

    # def prepare_data(self) -> None:
    #     Av1Dataset(self.data_root, 'train', self.train_transform, **self.dataset_cfg)
    #     Av1Dataset(self.data_root, 'val', self.val_transform, **self.dataset_cfg)

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = Av1Dataset(self.data_root, 'train', **self.dataset_cfg)
            self.val_dataset = Av1Dataset(self.data_root, 'val', **self.dataset_cfg)
        else:
            self.test_dataset = Av1Dataset(self.data_root, 'test', **self.dataset_cfg)

    def train_dataloader(self):
        return TorchDataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return TorchDataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, collate_fn=collate_fn)

    def test_dataloader(self):
        return TorchDataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, collate_fn=collate_fn)