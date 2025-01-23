import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.MyModel import MyModel
from model.Trainer import  Trainer
import yaml
from data.data_util import *
from data.datasetUtil import eegDataset
import time

torch.set_num_threads(10)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def dictToYaml(filePath, dictToWrite):
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)
    f.close()

def main(config):
    data_path = config['data_path']
    out_log = config['out_log']
    random_folder = str(time.strftime('%Y-%m-%d--%H-%M', time.localtime()))
    for sub in range(1, 10):
        train_data = 'A0' + str(sub) + 'T'
        test_data = 'A0' + str(sub) + 'E'

        out_log_path = os.path.join(out_log, config['network'], 'sub' + str(sub), random_folder)
        if not os.path.exists(out_log_path):
            os.makedirs(out_log_path)
        print("Results  in: " + out_log_path)

        dictToYaml(os.path.join(out_log_path, 'config.yaml'), config)
        setSeed(config['random_seed'])
        train_data, train_label = load_data(data_path, train_data)
        test_data, test_label = load_data(data_path, test_data)
        train_dataset = eegDataset(train_data, train_label)
        test_dataset = eegDataset(test_data, test_label)

        net_vars = config['network_args']
        net = eval(config['network'])(**net_vars)
        print('The Trainable Parameters Is: ' + str(count_parameters(net)))
        print('Current network is ' + config['network'])
        loss = nn.CrossEntropyLoss()
        opt = optim.Adam(net.parameters(), lr=config['lr'])
        trainer = Trainer(net, config, opt, loss, result_log = out_log_path)
        trainer.trainWithTest(train_dataset, test_dataset)


if __name__ == '__main__':
    configFile = 'config/2b_MyModel.yaml'
    file = open(configFile, 'r', encoding='utf-8')
    config = yaml.full_load(file)
    file.close()
    main(config)
