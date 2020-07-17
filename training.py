import torch
import numpy as np
from Utils.help_funcs import write_to_csv
import os
import glob
import configparser
from Dataset.dataset import get_dataloaders
from Model.model import BallDetectorModel
import torch.nn as nn
from Utils.FocalLoss import FocalLoss


def save_model(model, epoch, output_path, best=False):
    if best:
        previous_best_pt = glob.glob(os.path.join(output_path, '*BEST.pt'))
        os.remove(previous_best_pt[0])
        name = os.path.join(output_path, 'Model_stateDict__Epoch={}__BEST.pt'.format(epoch))
    else:
        name = os.path.join(output_path, 'Model_stateDict__Epoch={}.pt'.format(epoch))
    torch.save(model.statae_dict(), os.path.join(output_path, name))


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def train(model, train_dataloader, val_dataloader, epochs, criterion_loss, optimizer, scheduler, output_path):
    train_loss_list, val_loss_list =[], []
    best_train_loss, best_val_loss = np.inf, np.inf
    if not os.path.isdir(os.path.join(output_path, 'saved_checkpoints')):
        os.makedirs(os.path.join(output_path, 'saved_checkpoints'))

    model.train()
    softmax = nn.Softmax(dim=1)

    for epoch in epochs:
        running_loss = 0.0
        # --- TRAIN:  --- #
        for i, data in enumerate(train_dataloader):
            inputs, labels = data['image'], data['gt']
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = softmax(outputs)
            loss = criterion_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / i
        train_loss_list.append(train_loss.data.item())
        ##################

        model.eval()

        # --- VAL:  --- #
        val_running_loss = 0.0
        for val_i, val_data in enumerate(val_dataloader):
            inputs, labels = val_data['image'], val_data['gt']
            outputs = model(inputs)
            outputs = softmax(outputs)
            loss = criterion_loss(outputs, labels)
            val_running_loss += loss.item()
        val_loss = val_running_loss / val_i
        val_loss_list.append(val_loss.data.item())
        ##################

        # --- Save results to csv ---#
        current_lr = optimizer.param_groups[-1]['lr']
        current_wd = optimizer.param_groups[-1]['wd']
        epoch_results = [epoch, train_loss.data.item(), val_loss.data.item(), current_lr, current_wd]
        write_to_csv(os.path.join(output_path, 'results.csv'), epoch_results)
        ##############################

        # --- Save model checkpoint ---#
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            save_model(model=model, epoch=epoch, output_path=os.path.join(output_path, 'saved_checkpoints'), best=True)
        elif epoch % 10:
            save_model(model=model, epoch=epoch, output_path=os.path.join(output_path, 'saved_checkpoints'), best=False)
        ###############################

        scheduler.step(val_loss)
        train_dataloader.dataset.change_img_size()

        #  PRINT:  #
        print("Epoch {}:  train loss: {:.5f}, val loss: {:.5f}".format(epoch, train_loss, val_loss))


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    batch_size = config.getint('Params', 'batch_size')
    lr = config.getfloat('Params','lr')
    wd = config.getfloat('Params', 'wd')
    epochs= config.getint('Params', 'epochs')
    output_folder = config['Paths']['output_folder']

    # --- Dataset --- #
    datasets_imgs_folders = {'train': config['input_images']['train'],
                             'val': config['input_images']['val'],
                             'test': config['input_images']['test']}
    gt_imgs_folders = {'train': config['ground_truth']['train'],
                             'val': config['ground_truth']['val'],
                             'test': config['ground_truth']['test']}

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(dataset_folders=datasets_imgs_folders,
                                                                        gt_folders=gt_imgs_folders,
                                                                        batch_size=batch_size, num_workers=4)
    #####################

    # --- Model --- #
    checkpoint = config['Paths']['model_checkpoint']
    if len(checkpoint) > 0:
        model = BallDetectorModel()
        model.load_state_dict(torch.load(checkpoint))
    else:
        model = BallDetectorModel()
        model.apply(init_weights)
    #####################

    # --- Loss function --- #
    loss_BCE = nn.BCELoss()
    focal_loss = FocalLoss(gamma=2, alpha=1)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    #########################

    # --- send training --- #
    train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=epochs,
          criterion_loss=focal_loss, optimizer=optimizer, scheduler=scheduler, output_path=output_folder)


if __name__ == '__main__':
    main()