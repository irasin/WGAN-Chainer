import argparse
import os
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer.dataset import iterator
from chainer import cuda, Variable
from chainer import datasets, training, optimizers
from chainer.training import extensions

from PIL import Image


def backward_linear(x, l):
    y = F.matmul(x, l.W)
    return y


def backward_convolution(x_in, x, l):
    y = F.deconvolution_2d(x, l.W, None, l.stride, l.pad, (x_in.data.shape[2], x_in.data.shape[3]))
    return y


def backward_deconvolution(x_in, x, l):
    y = F.convolution_2d(x, l.W, None, l.stride, l.pad)
    return y


def backward_relu(x_in, x):
    y = (x_in.data > 0) * x
    return y


def backward_leaky_relu(x_in, x, a):
    y = (x_in.data > 0) * x + a * (x_in.data < 0) * x
    return y


def backward_sigmoid(x_in, g):
    y = F.sigmoid(x_in)
    return g * y * (1 - y)


class Generator(chainer.Chain):
    """wgan-gp for the mnist dataset"""
    """(batch_size, n_hidden/z_dim) -> (batch_size, 1, 28, 28) """

    def __init__(self, n_hidden=100, bottom_width=3, ch=512):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            self.l0 = L.Linear(in_size=n_hidden, out_size=bottom_width*bottom_width*ch)
            # output size (batch_size, 3*3*512)

            self.dc1 = L.Deconvolution2D(in_channels=ch, out_channels=ch//2, ksize=2, stride=2, pad=1)
            # output size (batch_size, 256, 4, 4)

            self.dc2 = L.Deconvolution2D(in_channels=ch//2, out_channels=ch//4, ksize=2, stride=2, pad=1)
            # output size (batch_size, 128, 6, 6)

            self.dc3 = L.Deconvolution2D(in_channels=ch//4, out_channels=ch//8, ksize=2, stride=2, pad=1)
            # output size (batch_size, 64, 10, 10)

            self.dc4 = L.Deconvolution2D(in_channels=ch//8, out_channels=1, ksize=3, stride=3, pad=1)
            # output size (batch_size, 1, 28, 28)

            self.bn1 = L.BatchNormalization(size=ch)
            self.bn2 = L.BatchNormalization(size=ch//2)
            self.bn3 = L.BatchNormalization(size=ch//4)
            self.bn4 = L.BatchNormalization(size=ch//8)

    def __call__(self, z):
        h = self.l0(z)
        h = F.reshape(h, (len(z), self.ch, self.bottom_width, self.bottom_width))
        h = F.relu(self.bn1(h))
        h = F.relu(self.bn2(self.dc1(h)))
        h = F.relu(self.bn3(self.dc2(h)))
        h = F.relu(self.bn4(self.dc3(h)))
        x = F.sigmoid(self.dc4(h))
        # x = F.tanh(self.dc4(h))
        return x


class Critic(chainer.Chain):
    """wgan-mnist-gp-mnist for the mnist dataset"""
    """(batch_size, 1, 28, 28) -> (1) """

    def __init__(self, ch=512):
        super(Critic, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_channels=1, out_channels=64, ksize=3, stride=3, pad=1)
            # output size (batch_size , 64, 10, 10)

            self.c1 = L.Convolution2D(in_channels=ch // 8, out_channels=128, ksize=2, stride=2, pad=1)
            # output size (batch_size, 128, 6, 6)

            self.c2 = L.Convolution2D(in_channels=ch // 4, out_channels=256, ksize=2, stride=2, pad=1)
            # output size (batch_size, 256, 4, 4)

            self.c3 = L.Convolution2D(in_channels=ch // 2, out_channels=512, ksize=2, stride=2, pad=1)
            # output size (batch_size, 512, 3, 3)
            self.l4 = L.Linear(in_size=512*3*3, out_size=1)
            # output size (batch_size, 1)

    def __call__(self, x):
        self.x = x
        self.h0 = F.leaky_relu(self.c0(self.x))
        self.h1 = F.leaky_relu((self.c1(self.h0)))
        self.h2 = F.leaky_relu((self.c2(self.h1)))
        self.h3 = F.leaky_relu((self.c3(self.h2)))
        return self.l4(self.h3)

    def differentiable_backward(self, x):
        g = backward_linear(x, self.l4)
        g = F.reshape(g, (x.shape[0], 512, 3, 3))
        g = backward_leaky_relu(self.h3, g, 0.2)
        g = backward_convolution(self.h2, g, self.c3)
        g = backward_leaky_relu(self.h2, g, 0.2)
        g = backward_convolution(self.h1, g, self.c2)
        g = backward_leaky_relu(self.h1, g, 0.2)
        g = backward_convolution(self.h0, g, self.c1)
        g = backward_leaky_relu(self.h0, g, 0.2)
        g = backward_convolution(self.x, g, self.c0)
        return g


def to_tuple(x):
    if hasattr(x, '__getitem__'):
        return tuple(x)
    return x,


class UniformNoiseGenerator(object):
    def __init__(self, low, high, size):
        self.low = low
        self.high = high
        self.size = to_tuple(size)

    def __call__(self, batch_size):
        return np.random.uniform(self.low, self.high, (batch_size,) + self.size).astype(np.float32)


class GaussianNoiseGenerator(object):
    def __init__(self, loc, scale, size):
        self.loc = loc
        self.scale = scale
        self.size = to_tuple(size)

    def __call__(self, batch_size):
        return np.random.normal(self.loc, self.scale, (batch_size,) + self.size).astype(np.float32)


class RandomNoiseIterator(iterator.Iterator):
    def __init__(self, noise_generator, batch_size):
        self.noise_generator = noise_generator
        self.batch_size = batch_size

    def __next__(self):
        return self.noise_generator(self.batch_size)


class WGANGPUpdater(training.StandardUpdater):
    def __init__(self, iterator, noise_iterator, optimizer_generator,
                 optimizer_critic, batch_size, lam, device=-1, *args, **kwargs):
        if optimizer_generator.target.name is None:
            optimizer_generator.target.name = 'generator'

        if optimizer_critic.target.name is None:
            optimizer_critic.target.name = 'critic'

        iterators = {'main': iterator, 'z': noise_iterator}
        optimizers = {'generator': optimizer_generator,
                      'critic': optimizer_critic}
        self.batchsize = batch_size
        self.lam = lam
        self.iteration = 0
        super(WGANGPUpdater, self).__init__(iterators, optimizers, device=device, *args, **kwargs)

    @property
    def optimizer_generator(self):
        return self._optimizers['generator']

    @property
    def optimizer_critic(self):
        return self._optimizers['critic']

    @property
    def generator(self):
        return self._optimizers['generator'].target

    @property
    def critic(self):
        return self._optimizers['critic'].target

    @property
    def x(self):
        return self._iterators['main']

    @property
    def z(self):
        return self._iterators['z']

    def next_batch(self, iterator):
        batch = self.converter(iterator.next(), self.device)
        return Variable(batch)

    def update_core(self):
        def _update(optimizer, loss):
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

        xp = self.generator.xp
        if self.iteration < 50:
            n_critic = 100
        else:
            n_critic = 5
        # update critic n_critic times
        for _ in range(n_critic):
            # real image
            x_real = self.next_batch(self.x)
            y_real = self.critic(x_real)
            loss1 = - F.sum(y_real)/self.batchsize

            # fake image
            z = self.next_batch(self.z)
            x_fake = self.generator(z)
            y_fake = self.critic(x_fake)
            loss2 = F.sum(y_fake)/self.batchsize

            x_fake.unchain_backward()

            # gp
            eps = xp.random.uniform(0, 1, size=self.batchsize).astype("f")[:, None, None, None]
            x_mid = eps * x_real + (1.0 - eps) * x_fake
            x_mid_v = Variable(x_mid.data)
            y_mid = self.critic(x_mid_v)
            dydx = self.critic.differentiable_backward(xp.ones_like(y_mid.data))
            dydx = F.sqrt(F.sum(dydx ** 2, axis=(1, 2, 3)))
            loss_gp = self.lam * F.mean_squared_error(dydx, xp.ones_like(dydx.data))

            # compute loss
            critic_loss = loss1 + loss2 + loss_gp

            # update critic
            _update(self.optimizer_critic, critic_loss)

            chainer.reporter.report({
                'critic/loss/real': loss1,
                'critic/loss/fake': loss2,
                'critic/loss/gp': loss_gp,
                'critic/loss': critic_loss,
                'wasserstein': -loss1-loss2,
            })

        # update generator 1 time
        z = self.next_batch(self.z)
        x_fake = self.generator(z)
        y_fake = self.critic(x_fake)
        gen_loss = -F.sum(y_fake)/self.batchsize
        _update(self.optimizer_generator, gen_loss)
        chainer.report({'generator/loss': gen_loss})


# Generate sample images
def out_generated_image(gen, rows, cols, seed, dst, n_hidden):
    @chainer.training.make_extension(trigger=(1, 'epoch'))
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable((xp.asarray(np.random.uniform(-1, 1, (n_images, n_hidden, 1, 1)).astype(np.float32))))

        with chainer.using_config('train', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        # when gen_output_activation_func is sigmoid (0 ~ 1)
        x = np.asarray(np.clip(x*255, 0.0, 255.0), dtype=np.uint8)
        # when gen output_activation_func is tanh (-1 ~ 1)
        # x = np.asarray(np.clip((x+1) * 0.5 * 255, 0.0, 255.0), dtype=np.uint8)

        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 1, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image_epoch_{:0>4}.png'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image


def main():
    parser = argparse.ArgumentParser(description='Chainer: WGAN-GP MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden dim of units (z)')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--lam', type=int, default=10,
                        help='lambda of gp in critic')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')

    train, _ = datasets.get_mnist(withlabel=False, ndim=3, scale=1.)  # ndim=3 : (ch,width,height)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    #z_iter = RandomNoiseIterator(GaussianNoiseGenerator(0, 1, args.n_hidden), args.batchsize)
    z_iter = RandomNoiseIterator(UniformNoiseGenerator(-1, 1, args.n_hidden), args.batchsize)

    # make the model
    gen = Generator(n_hidden=args.n_hidden)
    critic = Critic()
    if args.gpu >= 0:
        # make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()  # copy the model to the GPU
        critic.to_gpu()

    # make the optimizer
    optimizer_generator = optimizers.Adam(alpha=0.0001, beta1=0.5, beta2=0.9)
    optimizer_critic = optimizers.Adam(alpha=0.0001, beta1=0.5, beta2=0.9)
    optimizer_generator.setup(gen)
    optimizer_critic.setup(critic)

    updater = WGANGPUpdater(
        iterator=train_iter,
        noise_iterator=z_iter,
        optimizer_generator=optimizer_generator,
        optimizer_critic=optimizer_critic,
        device=args.gpu,
        batch_size=args.batchsize,
        lam=args.lam
    )

    epoch_interval = (1, 'epoch')
    display_interval = (10, 'iteration')

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    trainer.extend(extensions.snapshot_object(gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    trainer.extend(extensions.snapshot_object(critic, 'critic_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'critic/loss', 'critic/loss/real',
                                           'critic/loss/fake', 'critic/loss/gp', 'generator/loss', 'wasserstein']),
                   trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(out_generated_image(gen, 10, 10, args.seed, args.out, args.n_hidden), trigger=epoch_interval)

    trainer.run()

if __name__ == '__main__':
    main()
