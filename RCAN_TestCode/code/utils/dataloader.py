import threading
import random
import itertools
import warnings

import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
from torch.utils.data import BatchSampler
from torch.utils.data import _utils
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, \
    _MultiProcessingDataLoaderIter, _BaseDataLoaderIter

from torch.utils.data._utils import collate
from torch.utils.data._utils import signal_handling
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data._utils import ExceptionWrapper
from torch.utils.data._utils import IS_WINDOWS
from torch.utils.data._utils.worker import ManagerWatchdog

from torch._six import queue

def _ms_loop(dataset, index_queue, data_queue, done_event, collate_fn, scale, seed, init_fn, worker_id):
    try:
        collate._use_shared_memory = True
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        random.seed(seed)
        torch.manual_seed(seed)

        data_queue.cancel_join_thread()

        if init_fn is not None:
            init_fn(worker_id)

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            if r is None:
                assert done_event.is_set()
                return
            elif done_event.is_set():
                continue

            idx, batch_indices = r
            try:
                idx_scale = 0
                if len(scale) > 1 and dataset.train:
                    idx_scale = random.randrange(0, len(scale))
                    dataset.set_scale(idx_scale)

                samples = collate_fn([dataset[i] for i in batch_indices])
                samples.append(idx_scale)
            except Exception:
                data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
            else:
                data_queue.put((idx, samples))
                del samples

    except KeyboardInterrupt:
        pass


class _MSDataLoaderIter(_MultiProcessingDataLoaderIter):

    def __init__(self, loader):
        self.scale = loader.scale
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._pin_memory = loader.pin_memory and torch.cuda.is_available()
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        # self.batch_sampler = loader.batch_sampler
        # self.sample_iter = iter(self.batch_sampler)

# =================================================================
        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            index_queue = multiprocessing_context.Queue()
            # index_queue.cancel_join_thread()
            w = multiprocessing.Process(
                target=_ms_loop,
                args=(
                    self._dataset,
                    index_queue,
                    self._worker_result_queue,
                    self._workers_done_event,
                    self._collate_fn ,
                    self.scale,
                    self._base_seed + i,
                    self._worker_init_fn,
                    i
                )
            )
            w.daemon = True
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            self._data_queue  = queue.Queue()
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(
                    self._worker_result_queue,
                    self._data_queue ,
                    torch.cuda.current_device(),
                    self._pin_memory_thread_done_event
                )
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        _utils.signal_handling._set_worker_pids(
            id(self), tuple(w.pid for w in self._workers)
        )
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

        # for _ in range(2 * self.num_workers):
        #     # self._put_indices()
        #     self._try_put_index()


# class MSDataLoader(DataLoader):

#     def __init__(self, cfg, *args, **kwargs):
#         super(MSDataLoader, self).__init__(
#             *args, **kwargs, num_workers=cfg.n_threads
#         )
#         self.scale = cfg.scale

#     def _get_iterator(self):
#         if self.num_workers == 0:
#             return _SingleProcessDataLoaderIter(self)
#         else:
#             return _MSDataLoaderIter(self)


class MSDataLoader(DataLoader):
    def __init__(self, cfg, *args, **kwargs):
        super(MSDataLoader, self).__init__(
            *args, **kwargs, num_workers=cfg.n_threads, collate_fn=self.collate_fn
        )

    def collate_fn(self, sample): 
        sample = _utils.collate.default_collate(sample)
        if len(self.dataset.scale) > 1 and self.dataset.train:
            idx_scale = random.randrange(0, len(self.dataset.scale))
            self.dataset.set_scale(idx_scale)
        return sample
