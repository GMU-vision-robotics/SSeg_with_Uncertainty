import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import DuqHead, calc_gradient_penalty
from dataloaders.cityscapes_proposals import CityscapesProposalsDataset
from metrics import Evaluator
from loss import BinaryCrossEntropyLoss

BATCH_SIZE = 8
rep_style = 'ObjDet' #'both', 'ObjDet', 'SSeg'
saved_folder = 'trained_model/duq'
duq_l_gradient_penalty = 0.0

if rep_style == 'both':
    input_dim = 512
else:
    input_dim = 256

dataset_folder = '/projects/kosecka/yimeng/Datasets/Cityscapes'
ds_train = CityscapesProposalsDataset(dataset_folder, 'train', batch_size=BATCH_SIZE, rep_style=rep_style)
num_classes = ds_train.NUM_CLASSES
ds_val = CityscapesProposalsDataset(dataset_folder, 'val', batch_size=BATCH_SIZE, rep_style=rep_style)

# # Classification
device = torch.device('cuda')

def train_classifier(train_loader, classifier, optimizer):
    loss_ = 0.0
    epoch_loss = []
    for i in range(len(train_loader)):
        classifier.train()
        optimizer.zero_grad()

        images, labels = train_loader[i]
        images, labels = images.to(device), labels.to(device)

        if duq_l_gradient_penalty > 0.0:
            images.requires_grad_(True)
        
        logits = classifier(images)

        loss = BinaryCrossEntropyLoss(logits, labels.long(), num_classes)
        
        if duq_l_gradient_penalty > 0.0:
            logits = logits.permute(0, 2, 3, 1) # B, H, W, C
            gradient_penalty = duq_l_gradient_penalty * calc_gradient_penalty(images, logits)
            print('loss = {:.4f}, gradient_penalty = {:.4f}'.format(loss.item(), gradient_penalty))
            loss += gradient_penalty

        loss.backward()
        optimizer.step()

        #================================== update embedding===============================
        images.requires_grad_(False)
        with torch.no_grad():
            classifier.eval()
            classifier.update_embeddings(images, labels)

        epoch_loss.append(loss.item())
        print('i = {}, loss = {:.3f}'.format(i, loss.item()))
    return np.mean(epoch_loss)

def test_classifier(test_loader, classifier, evaluator):
    with torch.no_grad():
        classifier.eval()
        epoch_loss = []
        for i in range(len(test_loader)):
            images, labels = test_loader[i]
            images, labels = images.to(device), labels.to(device)

            logits = classifier(images)

            loss = BinaryCrossEntropyLoss(logits, labels.long(), num_classes)
            epoch_loss.append(loss.item())
            print('i = {}, loss = {:.3f}'.format(i, loss.item()))

            pred = logits.data.cpu().numpy()
            targets = labels.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            evaluator.add_batch(targets, pred)

        # final test
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    return Acc, Acc_class, mIoU, FWIoU, np.mean(epoch_loss)

classifier = DuqHead(num_classes, input_dim).to(device)
# You can can use this function to reload a network you have already saved previously
#classifier.load_state_dict(torch.load('resNet.pth'))

import torch.optim as optim
optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

evaluator = Evaluator(num_classes)

# Training the Classifier
NUM_EPOCHS = 200

best_pred = 0.0
for epoch in range(1, NUM_EPOCHS+1):
    print("Starting epoch number " + str(epoch))
    train_loss = train_classifier(ds_train, classifier, optimizer)
    print("Loss for Training on Epoch " +str(epoch) + " is "+ str(train_loss))
    
    Acc, Acc_class, mIoU, FWIoU, test_loss = test_classifier(ds_val, classifier, evaluator)
    print('Validation: [Epoch: %d' % (epoch))
    print("Acc:{:.5}, Acc_class:{:.5}, mIoU:{:.5}, fwIoU: {:.5}".format(Acc, Acc_class, mIoU, FWIoU))
    print('Test Loss: {:.3f}'.format(test_loss))

    new_pred = mIoU
    if new_pred > best_pred:
        best_pred = new_pred
        torch.save(classifier.state_dict(), '{}/duq_classifier_{}.pth'.format(saved_folder, duq_l_gradient_penalty))

    scheduler.step(train_loss)

