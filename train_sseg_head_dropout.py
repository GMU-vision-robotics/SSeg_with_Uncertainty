import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import DropoutHead
from dataloaders.cityscapes_proposals import CityscapesProposalsDataset
from metrics import Evaluator

BATCH_SIZE = 32
rep_style = 'ObjDet' #'both', 'ObjDet', 'SSeg'
saved_folder = 'trained_model/dropout'

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

def train_classifier(train_loader, classifier, criterion, optimizer):
    classifier.train()
    loss_ = 0.0
    epoch_loss = []
    for i in range(len(train_loader)):
        images, labels = train_loader[i]
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = classifier(images)

        loss = criterion(logits, labels.long())
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        print('i = {}, loss = {:.4f}'.format(i, loss.item()))
    return np.mean(epoch_loss)

def test_classifier(test_loader, classifier, criterion, evaluator):
    with torch.no_grad():
        classifier.eval()
        epoch_loss = []
        for i in range(len(test_loader)):
            images, labels = test_loader[i]
            images, labels = images.to(device), labels.to(device)

            logits = classifier(images)

            loss = criterion(logits, labels.long())
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

classifier = DropoutHead(num_classes, input_dim).to(device)
# You can can use this function to reload a network you have already saved previously
#classifier.load_state_dict(torch.load('resNet.pth'))

criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)

import torch.optim as optim
optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

evaluator = Evaluator(num_classes)

# Training the Classifier
NUM_EPOCHS = 200

best_pred = 0.0
for epoch in range(1, NUM_EPOCHS+1):
    print("Starting epoch number " + str(epoch))
    train_loss = train_classifier(ds_train, classifier, criterion, optimizer)
    print("Loss for Training on Epoch " +str(epoch) + " is "+ str(train_loss))

    Acc, Acc_class, mIoU, FWIoU, test_loss = test_classifier(ds_val, classifier, criterion, evaluator)
    print('Validation: [Epoch: %d' % (epoch))
    print("Acc:{:.5}, Acc_class:{:.5}, mIoU:{:.5}, fwIoU: {:.5}".format(Acc, Acc_class, mIoU, FWIoU))
    print('Test Loss: {:.3f}'.format(test_loss))

    new_pred = mIoU
    if new_pred > best_pred:
        best_pred = new_pred
        torch.save(classifier.state_dict(), '{}/dropout_classifier.pth'.format(saved_folder))

    scheduler.step(train_loss)

