import torch
import numpy as np
from Utils.help_funcs import write_to_csv
import os
import glob
import configparser
from Dataset.dataset import get_dataloaders
from Model.Model import BallDetector
import torch.nn as nn
from Utils.Losses import FocalLoss, dice_loss


def save_model(model, epoch, output_path, best=False):
    if best:
        previous_best_pt = glob.glob(os.path.join(output_path, '*BEST.pt'))
        if len(previous_best_pt) > 0:
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loss_list, val_loss_list = [], []
    best_train_loss, best_val_loss = np.inf, np.inf
    if not os.path.isdir(os.path.join(output_path, 'saved_checkpoints')):
        os.makedirs(os.path.join(output_path, 'saved_checkpoints'))

    model.train()
    model.to(device)
    softmax = nn.Softmax(dim=1)

    for epoch in epochs:
        running_loss = 0.0
        # --- TRAIN:  --- #
        for i, data in enumerate(train_dataloader):
            inputs, labels = data['image'].to(device), data['gt'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs.type(torch.FloatTensor).to(device))
            outputs = softmax(outputs)
            loss = criterion_loss(outputs, labels.squeeze().cuda().long())
            dice_score = dice_loss(output=outputs, target=labels)  # TODO: add dice loss to total loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / (i + 1)
        train_loss_list.append(train_loss)
        ##################

        model.eval()

        # --- VAL:  --- #
        val_running_loss = 0.0
        for val_i, val_data in enumerate(val_dataloader):
            inputs, labels = val_data['image'].to(device), val_data['gt'].to(device)
            outputs = model(inputs.type(torch.FloatTensor).to(device))
            outputs = softmax(outputs)
            loss = criterion_loss(outputs, labels.squeeze().cuda().long())
            dice_score = dice_loss(output=outputs, target=labels)  # TODO: add dice loss to total loss
            val_running_loss += loss.item()
        val_loss = val_running_loss / (val_i + 1)
        val_loss_list.append(val_loss)
        ##################

        # --- Save results to csv ---#
        current_lr = optimizer.param_groups[-1]['lr']
        current_wd = optimizer.param_groups[-1]['wd']
        epoch_results = [epoch, train_loss, val_loss, current_lr, current_wd]
        write_to_csv(os.path.join(output_path, 'results.csv'), epoch_results)
        ##############################

        # --- Save model checkpoint ---#
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model=model, epoch=epoch, output_path=os.path.join(output_path, 'saved_checkpoints'), best=True)
        elif epoch % 10:
            save_model(model=model, epoch=epoch, output_path=os.path.join(output_path, 'saved_checkpoints'), best=False)
        ###############################

        scheduler.step(val_loss)
        train_dataloader.dataset.change_img_size()

        #  PRINT:  #
        print("Epoch {}:  train loss: {:.5f}, val loss: {:.5f}".format(epoch, train_loss, val_loss)

    return train_loss, val_loss


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    batch_size = config.getint('Params', 'batch_size')
    lr = config.getfloat('Params', 'lr')
    wd = config.getfloat('Params', 'wd')
    epochs = config.getint('Params', 'epochs')
    output_folder = config['Paths']['output_folder']

    # --- Dataset --- #
    train_images = glob.glob(os.path.join(config['input_images']['train'], '*/*.jpg'))
    val_images = glob.glob(os.path.join(config['input_images']['val'], '*/*.jpg'))
    train_gt = glob.glob(os.path.join(config['ground_truth']['train'], '*/*.jpg'))
    val_gt = glob.glob(os.path.join(config['ground_truth']['val'], '*/*.jpg'))

    datasets_imgs_folders = {'train': train_images, 'val': val_gt}
    gt_imgs_folders = {'train': val_images, 'val': train_gt}

    train_dataloader, val_dataloader, _ = get_dataloaders(dataset_dict=datasets_imgs_folders,
                                                          gt_dict=gt_imgs_folders,
                                                          batch_size=batch_size, num_workers=4)
    #####################

    # --- Model --- #
    checkpoint = config['Paths']['model_checkpoint']
    if len(checkpoint) > 0:
        model = BallDetector(config=config)
        model.load_state_dict(torch.load(checkpoint))
    else:
        model = BallDetector(config=config)
        model.apply(init_weights)
    #####################

    # --- Loss function --- #
    loss_bce = nn.BCELoss()
    focal_loss = FocalLoss(gamma=2, alpha=1)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    #########################

    # --- send training --- #
    train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=epochs,
          criterion_loss=focal_loss, optimizer=optimizer, scheduler=scheduler, output_path=output_folder)


if __name__ == '__main__':
    main()