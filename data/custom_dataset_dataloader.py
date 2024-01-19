import torch.utils.data
from . import single_dataset
from .image_folder import make_dataset_with_labels, make_dataset

class CustomDatasetDataLoader(object):
    def name(self):
        return 'CustomDatasetDataLoader'

    def __init__(self, dataset_type, train, batch_size, 
		dataset_root="", transform=None, classnames=None,
		paths=None, labels=None, num_workers=0, **kwargs):

        self.train = train
        self.dataset = getattr(single_dataset, dataset_type)()
        self.dataset.initialize(root=dataset_root, 
                        transform=transform, classnames=classnames, 
			paths=paths, labels=labels, **kwargs)

        self.classnames = classnames
        self.batch_size = batch_size

        dataset_len = len(self.dataset)
        cur_batch_size = min(dataset_len, batch_size)
        assert cur_batch_size != 0, \
            'Batch size should be nonzero value.'

        if self.train:
            drop_last = True
            sampler = torch.utils.data.RandomSampler(self.dataset)
            batch_sampler = torch.utils.data.BatchSampler(sampler, 
	    			self.batch_size, drop_last)
        else:
            drop_last = False
            sampler = torch.utils.data.SequentialSampler(self.dataset)
            batch_sampler = torch.utils.data.BatchSampler(sampler, 
	    			self.batch_size, drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                         batch_sampler=batch_sampler,
                         num_workers=int(num_workers))

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class SingleSourceDataLoader(object):
    def name(self):
        return 'SingleSourceDataLoader'

    def __init__(self, dataset_type, batch_size, 
		dataset_root="", transform_train=None, transform_test=None, classnames=None,
		paths=None, labels=None, num_workers=0, **kwargs):

        self.data_paths, self.data_labels = make_dataset_with_labels(
				dataset_root, classnames)
        assert(len(self.data_paths) == len(self.data_labels)), \
            'The number of images (%d) should be equal to the number of labels (%d).' % \
            (len(self.data_paths), len(self.data_labels))
        index = list(range(len(self.data_paths)))
        import random
        import numpy as np
        random.shuffle(index)
        self.data_paths = np.array(self.data_paths)
        self.data_labels = np.array(self.data_labels)

        self.data_paths = self.data_paths[index]
        self.data_labels = self.data_labels[index]

        self.data_paths = list(self.data_paths)
        self.data_labels = list(self.data_labels)

        l = len(self.data_paths)
        train_size = int(0.9*l)
        
        self.data_paths_train = self.data_paths[:train_size]
        self.data_labels_train = self.data_labels[:train_size]
        
        self.data_paths_test = self.data_paths[train_size:]
        self.data_labels_test = self.data_labels[train_size:]

        self.dataset_train = getattr(single_dataset, dataset_type)()
        self.dataset_train.initializeSingle(root=dataset_root, 
			paths=self.data_paths_train, labels=self.data_labels_train,
            transform=transform_train)

        self.dataset_test = getattr(single_dataset, dataset_type)()
        self.dataset_test.initializeSingle(root=dataset_root, 
			paths=self.data_paths_test, labels=self.data_labels_test,
            transform=transform_test)

        self.classnames = classnames
        self.batch_size = batch_size

        drop_last = True
        sampler = torch.utils.data.RandomSampler(self.dataset_train)
        batch_sampler = torch.utils.data.BatchSampler(sampler, 
                self.batch_size, drop_last)
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, 
                        batch_sampler=batch_sampler,
                        num_workers=int(num_workers))
        
        drop_last = False
        sampler = torch.utils.data.SequentialSampler(self.dataset_test)
        batch_sampler = torch.utils.data.BatchSampler(sampler, 
                self.batch_size, drop_last)
        self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, 
                        batch_sampler=batch_sampler,
                        num_workers=int(num_workers))

