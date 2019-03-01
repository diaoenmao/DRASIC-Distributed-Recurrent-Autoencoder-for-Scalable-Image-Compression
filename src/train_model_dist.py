import torch
import config
import time
import torch.backends.cudnn as cudnn
import models
import torch.optim as optim
import os
import datetime
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from data import *
from utils import *
from metrics import *

cudnn.benchmark = True
config.init()
for k in config.PARAM:
    exec('{0} = config.PARAM[\'{0}\']'.format(k))
seeds = list(range(init_seed,init_seed+num_Experiments))

def main():
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
    
    train_dataset,_ = fetch_dataset(data_name=train_data_name)
    _,test_dataset = fetch_dataset(data_name=test_data_name)
    validated_num_epochs = max_num_epochs
    valid_data_size = len(train_dataset) if(data_size==0) else data_size
    train_loader,test_loader = split_dataset(train_dataset,test_dataset,valid_data_size,batch_size=batch_size,radomGen=randomGen,shuffle=False)
    print('Training data size {}, Number of Batches {}, Test data size {}'.format(valid_data_size,len(train_loader),len(test_dataset)))
    last_epoch = 0
    model = eval('models.{}.{}(classes_size=train_dataset.classes_size).to(device)'.format(model_dir,model_name))
    optimizer = make_optimizer(optimizer_name,model)
    scheduler = make_scheduler(scheduler_name,optimizer)
    if(resume_mode == 2):
        _,model,_,_ = resume(model,optimizer,scheduler,resume_model_TAG)       
    elif(resume_mode == 1):
        last_epoch,model,optimizer,scheduler = resume(model,optimizer,scheduler,resume_model_TAG)
    if(world_size > 1):
        model = torch.nn.DataParallel(model,device_ids=list(range(world_size)))
    best_pivot = 255
    best_pivot_name = 'loss'
    train_meter_panel = Meter_Panel(config.PARAM['train_metric_names'])
    test_meter_panel = Meter_Panel(config.PARAM['test_metric_names'])
    for epoch in range(last_epoch, validated_num_epochs+1):
        train_protocol = init_train_protocol(train_dataset)
        test_protocol = init_test_protocol(test_dataset)
        cur_train_meter_panel = train(train_loader,model,optimizer,epoch,train_protocol)
        cur_test_meter_panel = test(test_loader,model,epoch,test_protocol,model_TAG)
        print_result(epoch,cur_train_meter_panel,cur_test_meter_panel)
        scheduler.step(cur_test_meter_panel.panel['loss'].avg)
        train_meter_panel.update(cur_train_meter_panel)
        test_meter_panel.update(cur_test_meter_panel)
        if(save_mode>=0):
            model_state_dict = model.module.state_dict() if(world_size > 1) else model.state_dict()
            save_result = {'config':config.PARAM,'epoch':epoch+1,'model_dict':model_state_dict,'optimizer_dict':optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict(),'train_meter_panel':train_meter_panel,'test_meter_panel':test_meter_panel}
            save(save_result,'./output/model/{}_checkpoint.pkl'.format(model_TAG))
            if(best_pivot > test_meter_panel.panel[best_pivot_name].avg):
                best_pivot = test_meter_panel.panel[best_pivot_name].avg
                save(save_result,'./output/model/{}_best.pkl'.format(model_TAG))
    return
   
def train(train_loader,model,optimizer,epoch,protocol):
    meter_panel = Meter_Panel(protocol['metric_names'])
    model.train(True)
    end = time.time()
    for i, input in enumerate(train_loader):
        input = collate(input)
        input = dict_to_device(input,device)
        protocol = update_protocol(input,protocol)           
        output = model(input,protocol)
        output['loss'] = torch.mean(output['loss']) if(world_size > 1) else output['loss']                                                                                          
        optimizer.zero_grad()
        output['loss'].backward()
        optimizer.step()
        evaluation = meter_panel.eval(input,output,protocol)
        batch_time = time.time() - end
        meter_panel.update(evaluation,len(input['img']))
        meter_panel.update({'batch_time':batch_time})
        end = time.time()
        if(i % (len(train_loader)//5) == 0):
            estimated_finish_time = str(datetime.timedelta(seconds=(len(train_loader)-i-1)*batch_time))
            print('Train Epoch: {}[({:.0f}%)]{}, Estimated Finish Time: {}'.format(
                epoch, 100. * i / len(train_loader), meter_panel.summary(['loss','psnr','acc','batch_time']), estimated_finish_time))
    return meter_panel

def test(validation_loader,model,epoch,protocol,model_TAG):
    meter_panel = Meter_Panel(protocol['metric_names'])
    with torch.no_grad():
        model.train(False)
        end = time.time()
        for i, input in enumerate(validation_loader):
            input = collate(input)
            input = dict_to_device(input,device)
            protocol = update_protocol(input,protocol)  
            output = model(input,protocol)
            output['loss'] = torch.mean(output['loss']) if(world_size > 1) else output['loss']
            evaluation = meter_panel.eval(input,output,protocol)
            batch_time = time.time() - end
            meter_panel.update(evaluation,len(input['img']))
            meter_panel.update({'batch_time':batch_time})
            end = time.time()
        save_img(input['img'],'./output/img/image.png')
        save_img(output['compression']['img'],'./output/img/image_{}_{}.png'.format(model_TAG,epoch))
    return meter_panel

def make_optimizer(optimizer_name,model):
    if(optimizer_name=='Adam'):
        optimizer = optim.Adam(model.parameters(),lr=1e-3)
    elif(optimizer_name=='SGD'):
        optimizer = optim.SGD(model.parameters(),lr=1e-1, momentum=0.9)
    else:
        raise ValueError('Optimizer name not supported')
    return optimizer
    
def make_scheduler(scheduler_name,optimizer):
    if(scheduler_name=='MultiStepLR'):
        scheduler = MultiStepLR(optimizer,milestones=[10,20,50,100],gamma=factor)
    elif(scheduler_name=='ReduceLROnPlateau'):
        scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=factor,verbose=True,threshold=threshold,threshold_mode='abs')
    else:
        raise ValueError('Scheduler_name name not supported')
    return scheduler
    
def init_train_protocol(dataset):
    protocol = {}
    protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
    protocol['metric_names'] = config.PARAM['train_metric_names'].copy()
    protocol['topk'] = config.PARAM['topk']
    protocol['loss_mode'] = config.PARAM['loss_mode']
    protocol['num_node'] = config.PARAM['train_num_node']
    if(config.PARAM['balance']):
        protocol['classes_counts'] = dataset.classes_counts.expand(world_size,-1).to(device)
    return protocol 

def init_test_protocol(dataset):
    protocol = {}
    protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
    protocol['metric_names'] = config.PARAM['test_metric_names'].copy()
    protocol['topk'] = config.PARAM['topk']
    protocol['loss_mode'] = config.PARAM['loss_mode']
    protocol['num_node'] = config.PARAM['test_num_node']
    if(config.PARAM['balance']):
        protocol['classes_counts'] = dataset.classes_counts.expand(world_size,-1).to(device)
    return protocol
    
def collate(input):
    for k in input:
        input[k] = torch.stack(input[k],0)
    return input
   
def update_protocol(input,protocol):
    protocol['num_iter'] = config.PARAM['num_iter']
    protocol['depth'] = config.PARAM['max_depth']
    protocol['jump_rate'] = config.PARAM['jump_rate']
    protocol['patch_shape'] = config.PARAM['patch_shape']
    protocol['img_shape'] = (input['img'].size(2),input['img'].size(3))
    protocol['step'] = config.PARAM['step']
    if(input['img'].size(1)==1):
        protocol['mode'] = 'L'
    elif(input['img'].size(1)==3):
        protocol['mode'] = 'RGB'
    else:
        raise ValueError('Wrong number of channel')
    return protocol 

def print_result(epoch,train_meter_panel,test_meter_panel):
    estimated_finish_time = str(datetime.timedelta(seconds=(max_num_epochs - epoch - 1)*train_meter_panel.panel['batch_time'].sum))
    print('Test Epoch: {}{}{}, Estimated Finish Time: {}'.format(epoch,test_meter_panel.summary(['loss']+config.PARAM['test_metric_names']),train_meter_panel.summary(['batch_time']),estimated_finish_time))
    return

def resume(model,optimizer,scheduler,resume_model_TAG):
    if(os.path.exists('./output/model/{}_checkpoint.pkl'.format(resume_model_TAG))):
        checkpoint = load('./output/model/{}_checkpoint.pkl'.format(resume_model_TAG))
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        print('Resume from {}'.format(last_epoch))
    else:
        last_epoch = 0
        print('Not found existing model, and start from epoch {}'.format(last_epoch))
    return last_epoch,model,optimizer,scheduler
    
if __name__ == "__main__":
    main()   