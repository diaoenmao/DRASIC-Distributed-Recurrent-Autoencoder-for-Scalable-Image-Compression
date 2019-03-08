import torch
import config
import time
import torch.backends.cudnn as cudnn
import models
import torch.optim as optim
import os
import datetime
import argparse
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from data import *
from utils import *
from metrics import *

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
config.init()
for k in config.PARAM:
    exec('{0} = config.PARAM[\'{0}\']'.format(k))
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], help=\'\')'.format(k))
args = vars(parser.parse_args())
for k in config.PARAM:
    if(config.PARAM[k]!=args[k]):
        exec('config.PARAM[\'{0}\'] = {1}'.format(k,args[k]))

def main():
    seeds = list(range(init_seed,init_seed+num_Experiments))
    for i in range(num_Experiments):
        print('Experiment: {}'.format(seeds[i]))
        runExperiment(seeds[i])
    return

def runExperiment(seed):
    print(config.PARAM)
    resume_model_TAG = '{}_{}_{}'.format(seed,model_data_name,model_name) if(resume_TAG=='') else '{}_{}_{}_{}'.format(seed,model_data_name,model_name,resume_TAG)
    model_TAG = resume_model_TAG if(special_TAG=='') else '{}_{}'.format(resume_model_TAG,special_TAG)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    randomGen = np.random.RandomState(seed)
    
    train_dataset,test_dataset = fetch_dataset(data_name=test_data_name)
    valid_data_size = len(train_dataset) if(data_size==0) else data_size
    _,test_loader = split_dataset(train_dataset,test_dataset,valid_data_size,batch_size=batch_size,radomGen=randomGen)
    best = load('./output/model/{}_best.pkl'.format(resume_model_TAG))
    last_epoch = best['epoch']
    model = eval('models.{}.{}(classes_size=test_dataset.classes_size).to(device)'.format(model_dir,model_name))
    model.load_state_dict(best['model_dict'])
    test_protocol = init_test_protocol(test_dataset)
    result = []
    for i in range(1,num_iter+1):
        test_meter_panel = test(test_loader,model,last_epoch,test_protocol,i,model_TAG)
        print_result(last_epoch,i,test_meter_panel)
        result.append(test_meter_panel)
    save({'config':config.PARAM,'epoch':last_epoch,'result':result},'./output/result/{}.pkl'.format(model_TAG))  
    return
    
def test(validation_loader,model,epoch,protocol,iter,model_TAG):
    entropy_codec = models.classic.Entropy()
    meter_panel = Meter_Panel(protocol['metric_names'])
    with torch.no_grad():
        model.train(False)
        end = time.time()
        for i, input in enumerate(validation_loader):
            input = collate(input)
            input['img'] = input['img'][input['label']<config.PARAM['num_node']['E']] if(protocol['byclass']) else input['img']
            input['label'] = input['label'][input['label']<config.PARAM['num_node']['E']] if(protocol['byclass']) else input['img']
            input = dict_to_device(input,device)
            protocol = update_test_protocol(input,iter,i,len(validation_loader),protocol)
            output = model(input,protocol)
            output['loss'] = torch.mean(output['loss']) if(world_size > 1) else output['loss']
            output['compression']['code'] = entropy_codec.encode(output['compression']['code'],protocol)
            evaluation = meter_panel.eval(input,output,protocol)
            batch_time = time.time() - end
            meter_panel.update(evaluation,len(input['img']))
            meter_panel.update({'batch_time':batch_time})
            end = time.time()
        if(tuning_param['compression'] > 0):                                            
            save_img(input['img'],'./output/img/image.png')
            save_img(output['compression']['img'],'./output/img/image_{}_{}_{}.png'.format(model_TAG,epoch,iter))
    return meter_panel

def init_test_protocol(dataset):
    protocol = {}
    protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
    protocol['metric_names'] = config.PARAM['test_metric_names'].copy()
    protocol['loss_mode'] = config.PARAM['loss_mode']
    protocol['node_name'] = {'E':[str(i) for i in range(config.PARAM['num_node']['E'])],'D':[str(i) for i in range(config.PARAM['num_node']['D'])]}
    protocol['byclass'] = config.PARAM['byclass']
    return protocol
    
def collate(input):
    for k in input:
        input[k] = torch.stack(input[k],0)
    return input

def update_test_protocol(input,iter,i,num_batch,protocol):
    protocol['num_iter'] = iter
    if(i == num_batch-1):
        protocol['activate_full'] = True
    else:
        protocol['activate_full'] = False
    if(input['img'].size(1)==1):
        protocol['img_mode'] = 'L'
    elif(input['img'].size(1)==3):
        protocol['img_mode'] = 'RGB'
    else:
        raise ValueError('Wrong number of channel')
    return protocol

def print_result(epoch,iter,test_meter_panel):
    print('Test Epoch: {}({}){}'.format(epoch,iter,test_meter_panel.summary(['loss']+config.PARAM['test_metric_names'])))
    return
    
if __name__ == "__main__":
    main()