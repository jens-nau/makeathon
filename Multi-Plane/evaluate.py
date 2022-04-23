import shutil
import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import mp1
import mp2
import mp3
from dataloader import Dataset
import torch.nn.functional as F
import pandas as pd
from sklearn import metrics


def evaluate(args):

<<<<<<< HEAD
    test_dataset = Dataset(args.directory, args.task,
                           test=True, transform=None)
=======
    test_dataset = Dataset(args.directory, args.task, test = True, transform = None)
>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=-True, num_workers=11, drop_last=False)

    model = torch.load(args.model_directory + args.model_name)

    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    losses = []
<<<<<<< HEAD
    log_every = 20
    for i, (image, image1, image2, label, weight) in enumerate(test_loader):

=======
    log_every=20
    for i, (image, image1, image2, label, weight) in enumerate(test_loader):


>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
        if torch.cuda.is_available():
            image = image.cuda()
            image1 = image1.cuda()
            image2 = image2.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

<<<<<<< HEAD
        prediction, _, _, _ = model.forward(
            image.float(), image1.float(), image2.float())
        prediction = prediction.squeeze(0)
=======
        prediction = model.forward(image.float(),image1.float(),image2.float()).squeeze(0)
>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf

        loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0]))
        y_preds.append(probas[0].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

<<<<<<< HEAD
=======


>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
        if (i % log_every == 0) & (i > 0):
            print('''|Single batch number : {0} / {1} ] | avg test loss {2} | test auc : {3} '''.
                  format(
                      i,
                      len(test_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                  )
                  )

<<<<<<< HEAD
=======

>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
    test_loss = np.round(np.mean(losses), 4)
    test_auc = np.round(auc, 4)
    print('Test loss is {}'.format(test_loss))
    print('Test AUC is {}'.format(test_auc))


<<<<<<< HEAD
=======

>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])
<<<<<<< HEAD
    parser.add_argument('--model_name', type=str, required=True, default=None)
    parser.add_argument('-md', '--model_directory',
                        type=str, default='./models/')
    parser.add_argument('-d', '--data_directory', type=str,
                        default='/home/Documents/data/')
=======
    parser.add_argument('--model_name', type=str, required = True, default=None)
    parser.add_argument('-md', '--model_directory', type=str, default='./models/')
    parser.add_argument('-d', '--data_directory', type=str, default='/home/Documents/data/')
>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
    parser.add_argument('--log_every', type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args)
