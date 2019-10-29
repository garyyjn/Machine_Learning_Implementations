import os
import numpy as np
from tqdm import tqdm

import argparse

import torch

from model import Wide_ResNet

from matplotlib import pyplot as plt
import time

from dataset import get_dataset
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='gpu id', type=int, default=0)
    parser.add_argument('-i', '--cluster_interval', help='clustering frequency', type=int, default=5)
    parser.add_argument('-c', '--coeff_mean', help='lambda', type=float, default=0)
    parser.add_argument('-k', '--k', type=int, help='the number of clusters', default=8)
    parser.add_argument('--model', type=str, help='model path', default=None)
    parser.add_argument('--max_epoch', type=int, help='max epochs', default=150)
    parser.add_argument('--restore', type=int, help='Whether to rescore the optimizer and init_epochs.', default=1)
    parser.add_argument('--dataset', type=str, help='dataset. cifar10 or ImageNet', default='cifar10')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    batch_size = 512
    max_epoch = args.max_epoch
    learning_rate_min = 0.01
    K = args.k
    lambda_means_loss = args.coeff_mean
    cluster_interval = args.cluster_interval

    log_path = 'log'
    checkpoint_name = 'checkpoint_K={}_interval={}_meancoeff={:.3f}.pth.tar'.format(K, cluster_interval,
                                                                                    lambda_means_loss)
    checkpoint_path = os.path.join(log_path, checkpoint_name)

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    real_model = Wide_ResNet(depth=16, widen_factor=4, dropout_rate=0.3, num_classes=10)
    mean_model = Wide_ResNet(depth=16, widen_factor=4, dropout_rate=0.3, num_classes=10)

    train_dataset, test_dataset = get_dataset(args.dataset)

    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    test_queue = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True, num_workers=2)

    real_model = real_model.cuda()
    mean_model = mean_model.cuda()
    # real_model = torch.nn.DataParallel(real_model)
    # mean_model=torch.nn.DataParallel(mean_model)

    params = [param for param in real_model.parameters() if len(param.shape) == 4]
    mean_params = [param for param in mean_model.parameters() if len(param.shape) == 4]

    init_epoch, train_info, means, real_model, optimizer = load_checkpoint(args, real_model, K)
    save_checkpoint(init_epoch, means, train_info, real_model, optimizer, checkpoint_path)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(max_epoch), eta_min=learning_rate_min)

    for epoch in tqdm(range(init_epoch + 1, max_epoch)):
        scheduler.step()
        print('epoch  =  {} learning rate = {}'.format(epoch, scheduler.get_lr()[0]))
        if lambda_means_loss > 0:
            if epoch % cluster_interval == 0:
                means = get_means(params, K)
            get_nearest_weight(params, means, mean_params, mean_model, real_model)
        for name, model, is_train, data_queue in [
            ['train', real_model, True, train_queue],
            ['valid', real_model, False, test_queue],
            ['test', mean_model, False, test_queue]
        ]:
            result = infer(model, data_queue, optimizer, params, means, is_train=is_train,
                           lambda_means_loss=lambda_means_loss)

            if name not in train_info:
                train_info[name] = []
            train_info[name].append(result)

            print('{} loss:{:.6f} top1:{:.2f} top5:{:.2f}'.format(name, *result))
        save_checkpoint(epoch, means, train_info, real_model, optimizer, checkpoint_path)

    means = get_means(params, K)
    get_nearest_weight(params, means, mean_params, mean_model, real_model)

    print('{} loss:{:.6f} top1:{:.2f} top5:{:.2f}'.format('real model final result',
                                                          *infer(real_model, test_queue, is_train=False)))
    print('{} loss:{:.6f} top1:{:.2f} top5:{:.2f}'.format('mean model final result',
                                                          *infer(mean_model, test_queue, is_train=False)))

    # from IPython import embed;embed()

    epochs = list(range(len(train_info['train'])))
    plt.title('loss')
    plt.plot(epochs, [i[0] for i in train_info['train']], label='train')
    plt.plot(epochs, [i[0] for i in train_info['valid']], label='test')
    plt.plot(epochs, [i[0] for i in train_info['test']], label='Compressed test')
    plt.legend()
    plt.savefig('curve_loss.png')
    plt.cla()
    plt.title('accuracy')
    plt.plot(epochs, [i[1] for i in train_info['train']], label='train')
    plt.plot(epochs, [i[1] for i in train_info['valid']], label='test')
    plt.plot(epochs, [i[1] for i in train_info['test']], label='Compressed test')
    plt.legend()
    plt.savefig('curve_accuracy.png')
    plt.cla()


if __name__ == '__main__':
    main()