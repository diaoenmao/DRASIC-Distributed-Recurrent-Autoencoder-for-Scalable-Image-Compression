import config
config.init()
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import models
from collections import OrderedDict
from tabulate import tabulate
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from data import *
from metrics import *
from modules.organic import _oConvNd
from utils import *


cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], help=\'\')'.format(k))
args = vars(parser.parse_args())
for k in config.PARAM:
    if(config.PARAM[k]!=args[k]):
        exec('config.PARAM[\'{0}\'] = {1}'.format(k,args[k]))
    
def main():
    model_TAG = '0_{}_{}'.format(config.PARAM['data_name']['train'],config.PARAM['model_name']) \
        if(config.PARAM['special_TAG']=='') else '0_{}_{}_{}'.format(config.PARAM['data_name']['train'],config.PARAM['model_name'],config.PARAM['special_TAG'])
    runExperiment(model_TAG)
    return
    
def runExperiment(model_TAG):
    dataset = {}
    dataset['train'],_ = fetch_dataset(data_name=config.PARAM['data_name']['train'])
    config.PARAM['classes_size'] = dataset['train'].classes_size
    data_loader = split_dataset(dataset,data_size=config.PARAM['data_size'],batch_size=config.PARAM['batch_size'])
    print(config.PARAM)
    
    model = eval('models.{}(\'{}\').to(device)'.format(config.PARAM['model_name'],model_TAG))
    summary = summarize(data_loader['train'],model)
    content = parse_summary(summary)
    print(content)
    return

def summarize(train_loader, model):

    def register_hook(module):

        def hook(module, input, output):
            module_name = str(module.__class__.__name__)
            if(module_name not in summary['count']):
                summary['count'][module_name] = 1
            else:
                summary['count'][module_name] += 1
            key = str(hash(module))
            if(key not in summary['module']):
                summary['module'][key] = OrderedDict()
                summary['module'][key]['module_name'] = '{}_{}'.format(module_name,summary['count'][module_name])
                summary['module'][key]['input_size'] = []
                summary['module'][key]['output_size'] = []
                summary['module'][key]['params'] = {}
            summary['module'][key]['input_size'].append(list(input[0].size()))
            summary['module'][key]['output_size'].append(list(output.size()))                
            for name, param in module.named_parameters():
                if(param.requires_grad):
                    if(name == 'weight'):                        
                        if(name not in summary['module'][key]['params']):
                            summary['module'][key]['params'][name] = {}                       
                            summary['module'][key]['params'][name]['size'] = list(param.size())
                            summary['module'][key]['coordinates'] = []
                            summary['module'][key]['params'][name]['mask'] = torch.zeros(summary['module'][key]['params'][name]['size'],dtype=torch.long,device=config.PARAM['device'])
                    elif(name == 'bias'):
                        if(name not in summary['module'][key]['params']):
                            summary['module'][key]['params'][name] = {}                       
                            summary['module'][key]['params'][name]['size'] = list(param.size())
                            summary['module'][key]['params'][name]['mask'] = torch.zeros(summary['module'][key]['params'][name]['size'],dtype=torch.long,device=config.PARAM['device'])
                    else:
                        continue 
            if(len(summary['module'][key]['params'])==0):
                return
            if('weight' in summary['module'][key]['params']):
                weight_size = summary['module'][key]['params']['weight']['size']
                if(isinstance(module,_ConvNd)):
                    summary['module'][key]['coordinates'].append([torch.arange(weight_size[0],device=config.PARAM['device']),torch.arange(weight_size[1],device=config.PARAM['device'])])
                elif(isinstance(module,_oConvNd)):
                    summary['module'][key]['coordinates'].append(input[1])
                elif(isinstance(module,_BatchNorm)):
                    summary['module'][key]['coordinates'].append([torch.arange(weight_size[0],device=config.PARAM['device'])])
                elif(isinstance(module,nn.Linear)):
                    summary['module'][key]['coordinates'].append([torch.arange(weight_size[0],device=config.PARAM['device']),torch.arange(weight_size[1],device=config.PARAM['device'])])
                else:
                    raise ValueError('parametrized module not supported')
            else:
                raise ValueError('parametrized module no weight')
            for name in summary['module'][key]['params']:
                coordinates = summary['module'][key]['coordinates'][-1]
                if(name == 'weight'):
                    if(len(coordinates)==1):
                        summary['module'][key]['params'][name]['mask'][coordinates[0]] += 1
                    elif(len(coordinates)==2):
                        summary['module'][key]['params'][name]['mask'][coordinates[0].view(-1,1),coordinates[1].view(1,-1),] += 1
                    else:
                        raise ValueError('coordinates dimension not supported')
                elif(name == 'bias'):
                    if(len(coordinates)==1):
                        summary['module'][key]['params'][name]['mask'][coordinates[0]] += 1
                    elif(len(coordinates)==2):
                        summary['module'][key]['params'][name]['mask'][coordinates[0]] += 1
                    else:
                        raise ValueError('coordinates dimension not supported')
                else:
                    raise ValueError('parameters type not supported')
            return
            
        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not isinstance(module, nn.ModuleDict) and not (module == model)):
            hooks.append(module.register_forward_hook(hook))
        return
    
    run_mode = False
    summary = OrderedDict()
    summary['module'] = OrderedDict()
    summary['count'] = OrderedDict()
    hooks = []
    model.train(run_mode)
    model.apply(register_hook)
    for i, input in enumerate(train_loader):
        input = collate(input)
        input = dict_to_device(input,device)
        output = model(input)
        break
    for h in hooks:
        h.remove()
    summary['total_num_params'] = 0
    for key in summary['module']:
        num_params = 0
        for name in summary['module'][key]['params']:
            num_params += (summary['module'][key]['params'][name]['mask']>0).sum().item()
        summary['total_num_params'] += num_params
    summary['total_space_params'] = abs(summary['total_num_params'] * 32. / 8 / (1024 ** 2.))
    return summary
    
def parse_summary(summary):
    content = ''
    headers = ['Module Name','Input Size','Weight Size','Output Size','Number of Parameters']
    records = []
    for key in summary['module']:
        if('weight' not in summary['module'][key]['params']):
            continue
        module_name = summary['module'][key]['module_name']
        input_size = str(summary['module'][key]['input_size'])
        weight_size = str(summary['module'][key]['params']['weight']['size']) if('weight' in summary['module'][key]['params']) else 'N/A'
        output_size = str(summary['module'][key]['output_size'])
        num_params = 0
        for name in summary['module'][key]['params']:
            num_params += (summary['module'][key]['params'][name]['mask']>0).sum().item()
        records.append([module_name,input_size,weight_size,output_size,num_params])
    total_num_params = summary['total_num_params']
    total_space_params = summary['total_space_params']
    
    table = tabulate(records,headers=headers,tablefmt='github')
    content += table+'\n'
    content += '================================================================\n'
    content += 'Total Number of Parameters: {}\n'.format(total_num_params)
    content += 'Total Space of Parameters (MB): {:.2f}\n'.format(total_space_params)
    makedir_exist_ok('./output')
    content_file = open('./output/summary.md', 'w')
    content_file.write(content)
    content_file.close()
    return content
    
def collate(input):
    for k in input:
        input[k] = torch.stack(input[k],0)
    return input
    
if __name__ == "__main__":
    main()