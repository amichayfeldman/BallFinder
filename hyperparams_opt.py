import torch
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
import configparser
import glob
import os
from Dataset.dataset import get_dataloaders
from training import train
from Model.Model import BallDetector, init_weights
import pandas as pd
import numpy as np
from Utils.Losses import FocalLoss

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
torch.cuda.manual_seed_all(3)
np.random.seed(2)


def train_bayesian(parameterization):
    net = BallDetector(config=config)
    net.apply(init_weights)
    lr, w_d = parameterization.values()
    print("lr: {:.5f},  wd: {:.5f}".format(lr, w_d))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=w_d)

    train_loss_list, val_loss_list = train(model=net, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                                           epochs=epochs, criterion_loss=loss, optimizer=optimizer, scheduler=None,
                                           output_path=output_folder, alpha=1, save_model=False, write_csv=False)
    min_train_loss = min(train_loss_list)
    results.append({'epochs': list(range(len(train_loss_list))), 'train_loss': train_loss_list,
                    'val_loss': val_loss_list, 'lr': [lr] * len(train_loss_list), 'wd': [w_d] * len(train_loss_list)})
    return min_train_loss


def bayesian_opt_main():
    best_parameters = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-5, 1e-2]},
            {"name": "wd", "type": "range", "bounds": [1e-6, 1e-2]},
        ],
        evaluation_function=train_bayesian,
        objective_name='train_loss',
        minimize=True,
        total_trials=40
    )
    # print("Best Params:")
    # print("lr:{}, wd:{}".format(best_parameters['lr'], best_parameters['wd']))


def read_results(csv_path):
    def attach_best_train_loss(row):
        row['best_train_loss'] = min(row['train_loss'])
        # calc slope:
        row['slope'] = (float(row['train_loss'][25]) - float(row['train_loss'][-1])) / float(row['epochs'][-1])
        return row
    results = pd.read_csv(csv_path)
    results = results.apply(attach_best_train_loss, axis=1)
    results.sort_values(by=['best_train_loss', 'slope'], inplace=True)
    print(results[:10])


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    batch_size = config.getint('Params', 'batch_size')
    alpha = config.getfloat('Params', 'alpha')
    epochs = config.getint('Params', 'epochs')
    train_images = glob.glob(os.path.join(config['input_images']['train'], '*/*.jpg'))
    val_images = glob.glob(os.path.join(config['input_images']['val'], '*/*.jpg'))
    train_gt = glob.glob(os.path.join(config['ground_truth']['train'], '*/*/*.png'))
    val_gt = glob.glob(os.path.join(config['ground_truth']['val'], '*/*/*.png'))
    datasets_imgs_folders = {'train': train_images, 'val': val_images}
    gt_imgs_folders = {'train': train_gt, 'val': val_gt}
    # train_dataloader, val_dataloader, _ = get_dataloaders(dataset_dict=datasets_imgs_folders,
    #                                                       gt_dict=gt_imgs_folders,
    #                                                       batch_size=batch_size, num_workers=0, config=config)
    output_folder = config['Paths']['output_folder']
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    loss = FocalLoss(gamma=3)

    # results = pd.DataFrame(columns=['Epoch', 'train_loss', 'val_loss', 'lr', 'wd'])
    results = []
    # bayesian_opt_main()
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, 'bayesian_opt_results.csv'))

    read_results(csv_path='/home/amichay/DL/BallDetector/Runnings/22_Oct_BO/bayesian_opt_results.csv')
