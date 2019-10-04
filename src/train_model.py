import config

config.init()
import argparse
import datetime
import models
import numpy as np
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from data import fetch_dataset, split_dataset
from metrics import Metric
from utils import save, load, to_device, process_control_name, process_evaluation
from logger import Logger

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], help=\'\')'.format(k))
args = vars(parser.parse_args())
for k in config.PARAM:
    if config.PARAM[k] != args[k]:
        exec('config.PARAM[\'{0}\'] = {1}'.format(k, args[k]))


def main():
    process_control_name()
    seeds = list(range(config.PARAM['init_seed'], config.PARAM['init_seed'] + config.PARAM['num_Experiments']))
    for i in range(config.PARAM['num_Experiments']):
        model_tag = '{}_{}_{}_{}'.format(seeds[i], config.PARAM['data_name']['train'], config.PARAM['model_name'],
                                         config.PARAM['control_name'])
        print('Experiment: {}'.format(model_tag))
        runExperiment(model_tag)
    return


def runExperiment(model_tag):
    seed = int(model_tag.split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    config.PARAM['randomGen'] = np.random.RandomState(seed)
    dataset = {'train': fetch_dataset(data_name=config.PARAM['data_name']['train'])['train'],
               'test': fetch_dataset(data_name=config.PARAM['data_name']['test'])['test']}
    data_loader = split_dataset(dataset, data_size=config.PARAM['data_size'], batch_size=config.PARAM['batch_size'],
                                radomGen=config.PARAM['randomGen'])
    model = eval('models.{}().to(config.PARAM["device"])'.format(config.PARAM['model_name']))
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)
    if config.PARAM['resume_mode'] == 1:
        last_epoch, model, optimizer, scheduler, logger_path = resume(model, optimizer, scheduler, model_tag)
    elif config.PARAM['resume_mode'] == 2:
        last_epoch = 1
        _, model, _, _, logger_path = resume(model, optimizer, scheduler, model_tag)
    else:
        last_epoch = 1
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/{}_{}'.format(model_tag,current_time)
    logger = Logger(logger_path)
    config.PARAM['pivot_metric'] = 'train/Loss'
    config.PARAM['pivot'] = 65535
    for epoch in range(last_epoch, config.PARAM['num_epochs'] + 1):
        train(data_loader['train'], model, optimizer, logger, epoch)
        test(data_loader['test'], model, logger)
        if config.PARAM['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.tracker[config.PARAM['pivot_metric']], epoch=epoch)
        else:
            scheduler.step(epoch=epoch + 1)
        if config.PARAM['save_mode'] >= 0:
            model_state_dict = model.module.state_dict() if config.PARAM['world_size'] > 1 else model.state_dict()
            save_result = {
                'config': config.PARAM, 'epoch': epoch + 1, 'model_dict': model_state_dict,
                'optimizer_dict': optimizer.state_dict(), 'scheduler_dict': scheduler.state_dict(), 'logger_path': logger_path}
            save(save_result, './output/model/{}_checkpoint.pkl'.format(model_tag))
            if config.PARAM['pivot'] > logger.tracker[config.PARAM['pivot_metric']]:
                config.PARAM['pivot'] = logger.tracker[config.PARAM['pivot_metric']]
                shutil.copy('./output/model/{}_checkpoint.pkl'.format(model_tag),
                            './output/model/{}_best.pkl'.format(model_tag))
    logger.close()
    return


def train(data_loader, model, optimizer, logger, epoch):
    metric = Metric()
    model.train(True)
    for i, input in enumerate(data_loader):
        start_time = time.time()
        input = collate(input)
        input = to_device(input, config.PARAM['device'])
        model.zero_grad()
        output = model(input)
        output['loss'] = output['loss'].mean() if config.PARAM['world_size'] > 1 else output['loss']
        output['loss'].backward()
        optimizer.step()
        if i % int(len(data_loader) * config.PARAM['log_interval']) == 0:
            batch_time = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=batch_time * (len(data_loader) - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=(config.PARAM['num_epochs'] - epoch) * batch_time * len(data_loader))
            info = {'info': 'Epoch: {}, Learning rate: {}, Epoch Finished Time: {}, ' \
                            'Experiment Finished Time: {}'.format(epoch, lr, epoch_finished_time, exp_finished_time)}
            logger.append_text(info, 'train')
            evaluation = metric.evaluate(config.PARAM['metric_names']['train'], input, output)
            evaluation = process_evaluation(evaluation)
            logger.append_scalar(evaluation, 'train')
    return


def test(data_loader, model, logger):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, config.PARAM['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if config.PARAM['world_size'] > 1 else output['loss']
        evaluation = metric.evaluate(config.PARAM['metric_names']['test'], input, output)
        evaluation = process_evaluation(evaluation)
        logger.append_scalar(evaluation, 'test')
    return


def make_optimizer(model):
    if config.PARAM['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.PARAM['lr'], momentum=config.PARAM['momentum'],
                              weight_decay=config.PARAM['weight_decay'])
    elif config.PARAM['optimizer_name'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config.PARAM['lr'], momentum=config.PARAM['momentum'],
                                  weight_decay=config.PARAM['weight_decay'])
    elif config.PARAM['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.PARAM['lr'], weight_decay=config.PARAM['weight_decay'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer):
    if config.PARAM['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif config.PARAM['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.PARAM['step_size'],
                                              gamma=config.PARAM['factor'])
    elif config.PARAM['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.PARAM['milestones'],
                                                   gamma=config.PARAM['factor'])
    elif config.PARAM['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.PARAM['num_epochs'])
    elif config.PARAM['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.PARAM['factor'],
                                                         patience=config.PARAM['patience'], verbose=True,
                                                         threshold=config.PARAM['threshold'],
                                                         threshold_mode='rel')
    elif config.PARAM['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.PARAM['lr'], max_lr=10 * config.PARAM['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input


def resume(model, optimizer, scheduler, model_tag):
    if os.path.exists('./output/model/{}_checkpoint.pth'.format(model_tag)):
        checkpoint = load('./output/model/{}_checkpoint.pth'.format(model_tag))
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        logger_path = checkpoint['logger_path']
        print('Resume from {}'.format(last_epoch))
        return last_epoch, model, optimizer, scheduler, logger_path
    else:
        print('Not valid model tag')
    return


if __name__ == "__main__":
    main()