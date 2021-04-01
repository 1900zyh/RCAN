import bisect 
import warnings 
from importlib import import_module
from utils.dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets, args):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train
        self.scale = datasets[0].scale
        
        n_patches = args.batch_size * args.test_every
        n_images = self.cumulative_sizes[-1]
        if n_images == 0:
            self.repeat = 0
        else:
            self.repeat = max(n_patches // n_images, 1)
    
    def __len__(self): 
        return self.cumulative_sizes[-1] * self.repeat
    
    def __getitem__(self, idx): 
        idx = idx % self.cumulative_sizes[-1]
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                if d.find('DIV2K-Q') > 0: 
                    module_name = 'DIV2KJPEG'
                    m = import_module('data.div2kjpeg')
                else: 
                    module_name = 'DIV2K'
                    m = import_module('data.div2k')
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = MSDataLoader(
                args,
                MyConcatDataset(datasets, args),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                # num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
