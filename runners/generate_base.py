import torch
import logging
import torch.distributed as dist
import numpy as np
import os

import torchvision.utils
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from model.ema import ExponentialMovingAverage
from utils.util import set_seeds, make_dir
from denoiser import EDMDenoiser, VPSDEDenoiser, VESDEDenoiser, VDenoiser
from samplers import ddim_sampler, edm_sampler
from model.ncsnpp import NCSNpp


def get_model(config, local_rank):
    if config.model.denoiser_name == 'edm':
        if config.model.denoiser_network == 'song':
            model = EDMDenoiser(
                NCSNpp(**config.model.network).to(config.setup.device))
        else:
            raise NotImplementedError
    elif config.model.denoiser_name == 'vpsde':
        if config.model.denoiser_network == 'song':
            model = VPSDEDenoiser(config.model.beta_min, config.model.beta_max - config.model.beta_min,
                                  config.model.scale, NCSNpp(**config.model.network).to(config.setup.device))
        else:
            raise NotImplementedError
    elif config.model.denoiser_name == 'vesde':
        if config.model.denoiser_network == 'song':
            model = VESDEDenoiser(
                NCSNpp(**config.model.network).to(config.setup.device))
        else:
            raise NotImplementedError
    elif config.model.denoiser_name == 'v':
        if config.model.denoiser_network == 'song':
            model = VDenoiser(
                NCSNpp(**config.model.network).to(config.setup.device))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    model = DDP(model, device_ids=[local_rank])
    state = torch.load(config.model.ckpt, map_location=config.setup.device)
    model.load_state_dict(state['model'], strict=False)
    # print(model)
    # logging.info(model.load_state_dict(state['model'], strict=True))
    if config.model.use_ema:
        ema = ExponentialMovingAverage(
            model.parameters(), decay=config.model.ema_rate)
        ema.load_state_dict(state['ema'])
        ema.copy_to(model.parameters())

    model.eval()
    return model


def sample_batch(sample_dir, counter, max_samples, sampling_fn, sampling_shape, device, labels, n_classes):
    x = torch.randn(sampling_shape, device=device)
    with torch.no_grad():
        if labels is None:
            if n_classes is not None:
                raise ValueError(
                    'Need to set labels for class-conditional sampling.')

            x = sampling_fn(x)
        else:
            if isinstance(labels, int):
                if labels == n_classes:
                    labels = torch.randint(
                        n_classes, (sampling_shape[0],)).to(x.device)
                elif sampling_shape[0] % labels == 0:
                    labels = torch.tensor(
                        [[i] * labels for i in range(n_classes)], device=x.device).view(-1)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            x = sampling_fn(x, labels)

        x = (x / 2. + .5).clip(0., 1.)
        #torchvision.utils.save_image(x.detach().cpu(), os.path.join(
        #        sample_dir, 'batch_{}.png'.format(counter)), padding=0, nrow=8)
        x = x.cpu().permute(0, 2, 3, 1) * 255.
        x = x.numpy().astype(np.uint8)

        if x.shape[3] == 1:
            x = x[:, :, :, 0]

    c = 0
    for img in x:
        if counter < max_samples:
            if labels is not None:
                label = labels[c].item()
                Image.fromarray(img).save(os.path.join(
                    sample_dir, str(label).zfill(6), str(counter).zfill(6) + '.png'))
            else:
                Image.fromarray(img).save(os.path.join(
                    sample_dir, str(counter).zfill(6) + '.png'))
            c += 1
            counter += 1

    return counter, labels


def sample_batch_show(sample_dir, counter, max_samples, sampling_fn, sampling_shape, device, labels, n_classes):
    x = torch.randn(sampling_shape, device=device)
    with torch.no_grad():
        if labels is None:
            if n_classes is not None:
                raise ValueError(
                    'Need to set labels for class-conditional sampling.')

            x = sampling_fn(x)
        else:
            if isinstance(labels, int):
                if labels == n_classes:
                    labels = torch.randint(
                        n_classes, (sampling_shape[0],)).to(x.device)
                elif sampling_shape[0] % labels == 0:
                    # labels = torch.tensor(
                    #     [[i for i in range(n_classes)] * labels], device=x.device).view(-1)
                    labels = torch.tensor(
                       [[i] * labels for i in range(n_classes)], device=x.device).view(-1)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            x = sampling_fn(x, labels)

        x = (x / 2. + .5).clip(0., 1.)
        torchvision.utils.save_image(x.detach().cpu(), os.path.join(
                sample_dir, 'batch_{}.png'.format(counter)), padding=2, nrow=10)
        counter += 1

    return counter, labels


def evaluation(config, workdir):
    if config.model.ckpt is None:
        raise ValueError('Need to specify a checkpoint.')

    set_seeds(config.setup.global_rank, config.test.seed)
    torch.cuda.device(config.setup.local_rank)
    config.setup.device = 'cuda:%d' % config.setup.local_rank

    sample_dir = os.path.join(workdir, 'samples/')
    if config.setup.global_rank == 0:
        make_dir(sample_dir)
    dist.barrier()

    model = get_model(config, config.setup.local_rank)

    sampling_shape = (config.sampler.batch_size,
                      config.data.num_channels,
                      config.data.resolution,
                      config.data.resolution)

    def sampler(x, y=None):
        if config.sampler.type == 'ddim':
            return ddim_sampler(x, y, model, **config.sampler)
        elif config.sampler.type == 'edm':
            return edm_sampler(x, y, model, **config.sampler)
        else:
            raise NotImplementedError

    counter = (config.test.n_samples //
               (sampling_shape[0] * config.setup.global_size) + 1) * sampling_shape[0] * config.setup.global_rank

    if config.setup.global_rank == 0 and config.sampler.labels is not None:
        for i in range(config.data.n_classes):
            os.mkdir(os.path.join(sample_dir, str(i).zfill(6)))

    all_labels = []
    for _ in range(config.test.n_samples // (sampling_shape[0] * config.setup.global_size) + 1):
        counter, labels = sample_batch(sample_dir, counter, config.test.n_samples, sampler,
                                       sampling_shape, config.setup.device, config.sampler.labels, config.data.n_classes)
        all_labels.append(labels)

    if config.sampler.labels is not None:
        all_labels = torch.cat(all_labels)
        all_labels_across_all_gpus = [torch.empty_like(all_labels).to(
            config.setup.device) for _ in range(config.setup.global_size)]
        dist.all_gather(all_labels_across_all_gpus, all_labels)
        all_labels_across_all_gpus = torch.cat(all_labels_across_all_gpus)[
            :config.test.n_samples].to('cpu')
        if config.setup.global_rank == 0:
            torch.save(all_labels_across_all_gpus,
                       os.path.join(sample_dir, 'all_labels.pt'))
        dist.barrier()


def sample_show(config, workdir):
    if config.model.ckpt is None:
        raise ValueError('Need to specify a checkpoint.')

    set_seeds(config.setup.global_rank, config.test.seed)
    torch.cuda.device(config.setup.local_rank)
    config.setup.device = 'cuda:%d' % config.setup.local_rank

    sample_dir = os.path.join(workdir, 'samples/')
    if config.setup.global_rank == 0:
        make_dir(sample_dir)
    dist.barrier()

    model = get_model(config, config.setup.local_rank)

    sampling_shape = (config.sampler.batch_size,
                      config.data.num_channels,
                      config.data.resolution,
                      config.data.resolution)

    def sampler(x, y=None):
        if config.sampler.type == 'ddim':
            return ddim_sampler(x, y, model, **config.sampler)
        elif config.sampler.type == 'edm':
            return edm_sampler(x, y, model, **config.sampler)
        else:
            raise NotImplementedError

    counter = (config.test.n_samples //
               (sampling_shape[0] * config.setup.global_size) + 1) * sampling_shape[0] * config.setup.global_rank

    if config.setup.global_rank == 0 and config.sampler.labels is not None:
        for i in range(config.data.n_classes):
            os.mkdir(os.path.join(sample_dir, str(i).zfill(6)))

    for _ in range(64):
        counter, labels = sample_batch_show(sample_dir, counter, config.test.n_samples, sampler,
                                       sampling_shape, config.setup.device, config.sampler.labels, config.data.n_classes)


def test_fid(config, workdir):
    from utils.util import compute_fid
    from dnnlib.util import open_url
    import pickle

    if config.model.ckpt is None:
        raise ValueError('Need to specify a checkpoint.')

    set_seeds(config.setup.global_rank, config.test.seed)
    torch.cuda.device(config.setup.local_rank)
    config.setup.device = 'cuda:%d' % config.setup.local_rank

    sample_dir = os.path.join(workdir, 'samples/')
    if config.setup.global_rank == 0:
        make_dir(sample_dir)
    dist.barrier()

    model = get_model(config, config.setup.local_rank)
    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        inception_model = pickle.load(f).to(config.setup.device)
    
    def sampler(x, y=None):
        if config.sampler.type == 'ddim':
            return ddim_sampler(x, y, model, **config.sampler)
        elif config.sampler.type == 'edm':
            return edm_sampler(x, y, model, **config.sampler)
        else:
            raise NotImplementedError
    fid_sampling_shape = (config.sampler.batch_size, config.data.num_channels,
                          config.data.resolution, config.data.resolution)
    
    with torch.no_grad():
        fids = compute_fid(config.test.n_samples, config.setup.global_size, fid_sampling_shape, sampler, inception_model,
                                            config.data.fid_stats, config.setup.device, config.data.n_classes)
    
    logging.info(fids[0])
    dist.barrier()


def reconstruction_show(config, workdir):
    from score_losses import EDMLoss, VPSDELoss, VESDELoss, VLoss
    from stylegan3.dataset import ImageFolderDataset

    if config.model.ckpt is None:
        raise ValueError('Need to specify a checkpoint.')

    set_seeds(config.setup.global_rank, config.test.seed)
    torch.cuda.device(config.setup.local_rank)
    config.setup.device = 'cuda:%d' % config.setup.local_rank

    sample_dir = os.path.join(workdir, 'samples/')
    if config.setup.global_rank == 0:
        make_dir(sample_dir)
    dist.barrier()

    if config.loss.version == 'edm':
        loss_fn = EDMLoss(**config.loss).get_loss
    elif config.loss.version == 'vpsde':
        loss_fn = VPSDELoss(**config.loss).get_loss
    elif config.loss.version == 'vesde':
        loss_fn = VESDELoss(**config.loss).get_loss
    elif config.loss.version == 'v':
        loss_fn = VLoss(**config.loss).get_loss
    else:
        raise NotImplementedError
    
    if config.data.name.startswith('fmnist'):
        dataset = torchvision.datasets.FashionMNIST(
            root='/bigtemp/fzv6en/datasets/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    elif config.data.name.startswith('mnist'):
        dataset = torchvision.datasets.MNIST(
            root='/bigtemp/fzv6en/datasets/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    else:
        dataset = ImageFolderDataset(
            config.data.path, config.data.resolution, **config.data.dataset_params)
    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=True, batch_size=config.train.batch_size)

    model = get_model(config, config.setup.local_rank)

    for x, y in dataset_loader:

        if config.data.name.startswith('mnist') or config.data.name.startswith('fmnist'):
            x = (x.to(config.setup.device).to(
                        torch.float32) * 2. - 1.)
        else:
            x = (x.to(config.setup.device).to(
                torch.float32) / 127.5 - 1.)

        if config.data.n_classes is None:
            y = None
        else:
            if config.data.one_hot:
                train_y = torch.argmax(train_y, dim=1)
            y = train_y.to(config.setup.device)
            if y.dtype == torch.float32:
                y = y.long()

        x_rec = loss_fn.reconstruction(model, x, y)

