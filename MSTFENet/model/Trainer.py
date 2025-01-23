import copy
import os
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.cuda import set_device
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self, net, config, opt, loss, scheduler=None, result_log=None):        
        self.batchsize = config['batch_size']
        self.epoch = config['epoch']
        self.device = config['device']

        #  data augmentation
        self.classes = config['classes']
        self.num_segs = config['num_segs']

        self.device = set_device(0)

        self.net = net.to(self.device)

        # training
        self.opt = opt
        self.loss = loss
        self.scheduler = scheduler

        # save result
        self.result_log = result_log
        self.log_writer= None
        if self.result_log is not None:
            self.log_writer= open(os.path.join(self.result_log, 'log_result.txt'), 'w')

    def data_augmentation(self, data, label):
        aug_data = []
        aug_label = []

        N, C, T = data.shape
        seg_size = T // self.num_segs
        aug_data_size = self.batchsize // self.classes

        for cls in range(self.classes):
            cls_idx = np.where(label == cls)
            cls_data = data[cls_idx]
            data_size = cls_data.shape[0]
            if data_size == 0 or data_size == 1:
                continue
            temp_aug_data = np.zeros((aug_data_size, C, T))
            for i in range(aug_data_size):
                rand_idx = np.random.randint(0, data_size, self.num_segs)
                for j in range(self.num_segs):
                    temp_aug_data[i, :, j * seg_size:(j + 1) * seg_size] = cls_data[rand_idx[j], :,
                                                                           j * seg_size:(j + 1) * seg_size]
            aug_data.append(temp_aug_data)
            aug_label.extend([cls] * aug_data_size)

        aug_data = np.concatenate(aug_data, axis=0)
        aug_label = np.array(aug_label)

        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data)
        aug_label = torch.from_numpy(aug_label)

        return aug_data, aug_label




    def trainWithTest(self, train_dataset, test_dataset):


        train_dataloader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=8)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batchsize, num_workers=8)
        best_acc = 0
        best_test_f_score = 0
        best_kappa = 0
        test_f_score = 0
        best_model = None

        for epoch in range(self.epoch):
            start_time = time.time()
            # train
            self.net.train()
            train_loss = 0
            train_pred = []
            train_true = []
            with torch.enable_grad():
                for train_data, train_label in train_dataloader:
                    # data augmentation
                    aug_data, aug_label = self.data_augmentation(train_data, train_label)
                    train_data = torch.cat((train_data, aug_data), axis=0)
                    train_label = torch.cat((train_label, aug_label), axis=0)
                    train_data = train_data.type(torch.FloatTensor).to(self.device)
                    train_label = train_label.type(torch.LongTensor).to(self.device)

                    train_output = self.net(train_data)
                    temp_train_loss = self.loss(train_output, train_label)
                    self.opt.zero_grad()
                    temp_train_loss.backward()
                    self.opt.step()
                    # loss
                    train_loss += temp_train_loss.item()
                    train_pred.extend(torch.max(train_output, 1)[1].cpu().tolist())
                    train_true.extend(train_label.cpu().tolist())
            train_loss /= len(train_dataloader)
            self.scheduler.step(train_loss)
            # test
            self.net.eval()
            test_loss = 0
            test_pred = []
            test_true = []
            with torch.no_grad():
                for test_data, test_label in test_dataloader:
                    test_data = test_data.type(torch.FloatTensor).to(self.device)
                    test_label = test_label.type(torch.LongTensor).to(self.device)
                    test_output = self.net(test_data)

                    temp_test_loss  = self.loss(test_output, test_label)

                    test_pred.extend(torch.max(test_output, 1)[1].cpu().tolist())
                    test_true.extend(test_label.cpu().tolist())
                    test_loss += temp_test_loss
            # test loss
            test_loss /= len(test_dataloader)

            train_acc = accuracy_score(train_true, train_pred)
            test_acc = accuracy_score(test_true, test_pred)
            test_kappa = cohen_kappa_score(test_true, test_pred)
            test_f_score = f1_score(test_true, test_pred, average='weighted')
            # save best data
            if test_acc > best_acc:
                best_acc = test_acc
                best_kappa = test_kappa
                best_test_f_score = test_f_score
                best_model = copy.deepcopy(self.net.state_dict())
            end_time = time.time()
            train_time = end_time - start_time
            self.log_write.write(f'epoch [{epoch+1}]  train loss: {train_loss:.6f}  train acc: {train_acc:.6f}  test loss: {test_loss:.6f} test acc: {test_acc:.6f} test kappa: {test_kappa:.6f}  \n')
            print('epoch [%d] : (train acc: %.6f train loss: %.6f)    (test acc: %.6f test loss: %.6f)  best acc: %.6f lr: %.6f  train_time: %.6f  f_score: %.6f '
                      %(epoch+1, train_acc, train_loss, test_acc,test_loss , best_acc,self.opt.param_groups[0]['lr'],train_time,test_f_score))
        print('best acc is: ', best_acc)
        print('best f_score is: ', test_f_score)
        # save log
        self.log_writer.write(f'best accuracy : {best_acc:.6f}\n')
        self.log_writer.write(f'best kappa : {best_kappa:.6f}\n')
        self.log_writer.write(f'best f_score: {best_test_f_score:.6f}\n')
        self.log_writer.close()
        torch.save(best_model, os.path.join(self.result_log, 'model.pth'))