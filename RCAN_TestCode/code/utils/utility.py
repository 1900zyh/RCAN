import os
import cv2 
import math
import time
import datetime
import warnings
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

# from utils.scheduler import GradualWarmupScheduler


class timer():
    def __init__(self):
        self.acc = 0
        self.t0 = torch.cuda.Event(enable_timing=True)
        self.t1 = torch.cuda.Event(enable_timing=True)
        self.tic()

    def tic(self):
        # self.t0 = time.time()
        self.t0.record()

    def toc(self, restart=False):
        # diff = time.time() - self.t0
        # if restart: self.t0 = time.time()
        self.t1.record()
        torch.cuda.synchronize()
        diff = self.t0.elapsed_time(self.t1) /1000.
        if restart: self.tic()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = {
            'psnr': torch.Tensor(), 
            'ssim': torch.Tensor()
        }
        # self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join(args.dir_exp, args.save)
        else:
            self.dir = os.path.join(args.dir_exp, args.load)
            if os.path.exists(self.dir):
                try: 
                    self.log = torch.load(self.get_path('metric_log.pt'))
                    print('[**] Continue log from epoch {}...'.format(len(self.log['psnr'])))
                except: 
                    pass
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_metric('psnr', epoch)
        self.plot_metric('ssim', epoch)
        trainer.optimizer.save(self.dir)
        # torch.save(self.log, self.get_path('psnr_log.pt'))
        torch.save(self.log, self.get_path('metric_log.pt'))

    def add_log(self, name, log):
        self.log[name] = torch.cat([self.log[name], log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_metric(self, metric_name, epoch):
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                data = self.log[metric_name][:, idx_data, idx_scale].numpy()
                axis = np.linspace(1, len(data), len(data))
                plt.plot(
                    axis,
                    data,
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel(metric_name.upper())
            plt.grid(True)
            plt.savefig(self.get_path(f'{d}_{metric_name}.pdf'))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if not self.args.remove_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

    
def calc_ssim(sr, hr, scale, rgb_range, dataset=None):
    if dataset and dataset.dataset.benchmark:
        shave = scale
    else:
        shave = scale + 6
    
    pixel_range = 255 / rgb_range
    sr = sr.data.squeeze().float().cpu().numpy() * pixel_range
    sr = np.transpose(sr,(1,2,0))
    hr = hr.data.squeeze().cpu().numpy() * pixel_range
    hr = np.transpose(hr,(1,2,0))
    
    sr_y = np.dot(sr,[65.738,129.057,25.064])/255.0+16.0
    hr_y = np.dot(hr,[65.738,129.057,25.064])/255.0+16.0
    if not sr.shape == hr.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = sr.shape[:2]
    sr_y = sr_y[shave:h-shave, shave:w-shave]
    hr_y = hr_y[shave:h-shave, shave:w-shave]

    if sr_y.ndim == 2:
        return ssim(sr_y, hr_y)
    elif sr.ndim == 3:
        if sr.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(sr, hr))
            return np.array(ssims).mean()
        elif sr.shape[2] == 1:
            return ssim(np.squeeze(sr), np.squeeze(hr))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(sr, hr):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(sr, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(hr, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(sr**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(hr**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(sr * hr, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
    

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'ADAMW': 
        optimizer_class = optim.AdamW 
        kwargs_optimizer['weight_decay'] = 1e-4
        

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    if args.scheduler == 'StepLR': 
        scheduler_class = lrs.MultiStepLR
        kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    elif args.scheduler == 'CosLR': 
        scheduler_class = lrs.CosineAnnealingLR
        kwargs_scheduler = {'T_max': args.T_max, 'eta_min': args.eta_min}
        if args.warm_up: 
            kwargs_scheduler = {
                'multiplier': 1, 'total_epoch': args.warmup_epoch,
                'after_scheduler': scheduler_class, 'after_scheduler_kwargs': kwargs_scheduler}
            scheduler_class = GradualWarmupScheduler

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            try: 
                self.load_state_dict(torch.load(self.get_dir(load_dir)))
                with warnings.catch_warnings(record=True) as w:
                    if epoch > 1:
                        for _ in range(epoch): self.scheduler.step()
                print(f'[**] Load optimizer from {load_dir}')
            except: 
                pass

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

