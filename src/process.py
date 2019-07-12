import config
config.init()
import argparse
import itertools
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import models
from collections import OrderedDict
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
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
path = 'milestones_0/02'

def index_TAG(model_TAG,index):
    control = model_TAG.split('_')[index]
    return control

def remove_TAG(model_TAG,index):
    list_model_TAG = model_TAG.split('_')
    list_model_TAG.pop(index)
    model_TAG = ('_').join(list_model_TAG)
    return model_TAG

def Eb_N0(snr,rate):
    Eb_N0 = 10*math.log10((10**(snr/10))/(1/rate))
    return Eb_N0

def main():
    control_names = [['0'],[config.PARAM['data_name']['train']],['psk','smartcode','smartcode2','smartcode3'],['awgn'],['1','2','3','4','5','6','7','8','9','10'],['1','2','4','8']]
    control_names_product = list(itertools.product(*control_names)) 
    model_TAGs = ['_'.join(control_names_product[i]) for i in range(len(control_names_product))]    
    extract_result(model_TAGs)
    gather_result(model_TAGs)
    process_result(control_names)
    show_result()

def extract_result(model_TAGs):
    head = './output/model/{}'.format(path)
    tail = 'best'
    for i in range(len(model_TAGs)):
        model_path = '{}/{}_{}.pkl'.format(head,model_TAGs[i],tail)
        if(os.path.exists(model_path)):
            result = load(model_path)
            save({'train_meter_panel':result['train_meter_panel'],'test_meter_panel':result['test_meter_panel']},'./output/result/{}/{}.pkl'.format(path,model_TAGs[i])) 
        else:
            print('model path {} not exist'.format(model_path))
    return

def gather_result(model_TAGs):
    gathered_result = {}
    dataset = {}
    dataset['train'],_ = fetch_dataset(data_name=config.PARAM['data_name']['train'])
    data_loader = split_dataset(dataset,data_size=config.PARAM['data_size'],batch_size=config.PARAM['batch_size'])   
    head = './output/result/{}'.format(path)
    for i in range(len(model_TAGs)):
        model_TAG = model_TAGs[i]
        result_path = '{}/{}.pkl'.format(head,model_TAG)
        model_name = index_TAG(model_TAG,2)
        if(os.path.exists(result_path)):
            model = eval('models.{}(\'{}\').to(device)'.format(model_name,model_TAG))
            result = load(result_path)
            gathered_result[model_TAG] = {
            'Eb/N0(dB)':Eb_N0(int(index_TAG(model_TAG,4)),int(index_TAG(model_TAG,5))),
            'loss':result['test_meter_panel'].panel['loss'].history_avg[-1],
            'BER':result['test_meter_panel'].panel['ber'].history_avg[-1]}
        else:
            print('result path {} not exist'.format(result_path))
    print(gathered_result)
    save(gathered_result,'./output/result/{}/gathered_result.pkl'.format(path))
    return

def process_result(control_names):
    control_size = [len(control_names[i]) for i in range(len(control_names))]
    result_path = './output/result/{}/gathered_result.pkl'.format(path)
    result = load(result_path)
    evaluation_names = ['Eb/N0(dB)','loss','BER']
    processed_result = {}
    processed_result['indices'] = {}
    processed_result['all'] = {k:torch.zeros(control_size,device=config.PARAM['device']) for k in evaluation_names}
    processed_result['mean'] = {}
    processed_result['stderr'] = {}
    for model_TAG in result:
        processed_result['indices'][model_TAG] = []
        for i in range(len(control_names)):
            processed_result['indices'][model_TAG].append(control_names[i].index(index_TAG(model_TAG,i)))
        print(model_TAG,processed_result['indices'][model_TAG])
        for k in processed_result['all']:
            processed_result['all'][k][tuple(processed_result['indices'][model_TAG])] = result[model_TAG][k]
    for k in evaluation_names:
        processed_result['mean'][k] = processed_result['all'][k].mean(dim=0,keepdim=True)
    for k in evaluation_names:
        processed_result['stderr'][k] = processed_result['all'][k].std(dim=0,keepdim=True)/math.sqrt(processed_result['all'][k].size(0))
    print(processed_result)
    save(processed_result,'./output/result/{}/processed_result.pkl'.format(path))
    return

def show_result():
    fig_format = 'png'
    result_path = './output/result/{}/processed_result.pkl'.format(path)
    result = load(result_path)
    x_name = 'Eb/N0(dB)'
    y_name = 'BER'
    control_index = 4
    num_stderr = 1
    band = False
    save = False
    x,y,y_min,y_max = {},{},{},{}
    for model_TAG in result['indices']:
        label = remove_TAG(model_TAG,control_index)
        if(label not in x):
            x[label],y[label],y_min[label],y_max[label] = [],[],[],[]
        idx = tuple(result['indices'][model_TAG])
        x[label].append(result['mean'][x_name][idx].item())
        y[label].append(result['mean'][y_name][idx].item())
        y_min[label].append(y[label][-1]-num_stderr*result['stderr'][y_name][idx].item())
        y_max[label].append(y[label][-1]+num_stderr*result['stderr'][y_name][idx].item())
    colors = plt.get_cmap('rainbow')
    colors_indices = np.linspace(0.2,1,len(x)).tolist()
    fig = plt.figure()
    fontsize = 14
    k = 0
    for label in x:        
        color = colors(colors_indices[k])
        plt.plot(x[label],y[label],color=color,linestyle='-',label=label,linewidth=3)
        if(band):
           plt.fill_between(x[label],y_max[label],y_min[label],color=color,alpha=0.5,linewidth=1) 
        k = k + 1
    plt.xlabel(x_name,fontsize=fontsize)
    plt.ylabel(y_name,fontsize=fontsize)
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()
    makedir_exist_ok('./output/fig/{}'.format(path))
    fig.savefig('./output/fig/{}/result.{}'.format(path,fig_format),bbox_inches='tight',pad_inches=0)       
    plt.close()
    return
   
if __name__ == '__main__':
    main()