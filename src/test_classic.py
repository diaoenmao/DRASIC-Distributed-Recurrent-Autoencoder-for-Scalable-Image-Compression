import config

config.init()
import argparse
import models
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from data import fetch_dataset, split_dataset
from metrics import Metric
from utils import save, load, to_device, process_control_name
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
        model_tag = '{}_{}_{}_{}'.format(seeds[i], config.PARAM['data_name'], config.PARAM['model_name'],
                                         config.PARAM['control_name'])
        print('Experiment: {}'.format(model_tag))
        runExperiment(model_tag)
    return


def runExperiment(model_tag):
    model_tag_list = model_tag.split('_')
    seed = int(model_tag_list[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    randomGen = np.random.RandomState(seed)
    dataset = {'test': fetch_dataset(data_name=config.PARAM['data_name']['test'])['test']}
    data_loader = split_dataset(dataset, data_size=config.PARAM['data_size'], batch_size=config.PARAM['batch_size'],
                                radomGen=randomGen)
    model = eval('models.{}().to(device)'.format(config.PARAM['model_name']))
    logger = Logger('runs/{}'.format(model_tag))
    print(config.PARAM)
    test(data_loader['test'], model, logger)
    return


def test(data_loader, model, logger):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, config.PARAM['device'])
            output = model(input)
            output['Loss'] = output['Loss'].mean() if config.PARAM['world_size'] > 1 else output['Loss']
        evaluation = metric.evaluate(config.PARAM['metric_names']['test'], input, output)
        logger.append_scalar(evaluation, 'test')
    return


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input


if __name__ == "__main__":
    main()