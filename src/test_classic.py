import torch
import config
import time
import torch.backends.cudnn as cudnn
import models
from data import *
from utils import *
from metrics import *

cudnn.benchmark = True
config.init()
for k in config.PARAM:
    exec('{} = config.PARAM[\'{}\']'.format(k,k))
init_seed = 0
seeds = list(range(init_seed,init_seed+num_Experiments))
format = special_TAG

def main():
    for i in range(num_Experiments):
        print('Experiment: {}'.format(seeds[i]))
        runExperiment(seeds[i])
    return
       
def runExperiment(seed):
    print(config.PARAM)
    model_TAG = '{}_{}_{}'.format(seed,model_data_name,model_name) if(special_TAG=='') else '{}_{}_{}_{}'.format(seed,model_data_name,model_name,special_TAG)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    randomGen = np.random.RandomState(seed)
    
    train_dataset,test_dataset = fetch_dataset(data_name=test_data_name)
    valid_data_size = len(train_dataset) if(data_size==0) else data_size
    _,test_loader = split_dataset(train_dataset,test_dataset,valid_data_size,batch_size=batch_size,radomGen=randomGen)
    model = eval('models.{}.{}()'.format(model_dir,model_name))
    test_protocol = init_test_protocol(test_dataset)
    if(special_TAG=='jpg'):
        qualities = list(range(8,97,8)) #MNIST
    elif(special_TAG=='jp2'):
        qualities = list(range(16,53,4)) #SVHN
        #qualities = list(range(16,25,1)) #MNIST
    else:
        qualities = list(range(8,97,8))
    bpp = np.zeros(len(qualities))
    psnr = np.zeros(len(qualities))
    result = []
    for i in range(len(qualities)):
        test_meter_panel = test(test_loader,model,test_protocol,qualities[i],model_TAG)
        if(test_meter_panel.panel['psnr'].avg == float('Inf')):
            break
        print_result(i,test_meter_panel)
        result.append(test_meter_panel)
    save({'config':config.PARAM,'result':result},'./output/result/{}.pkl'.format(model_TAG))  
    return
    
def test(validation_loader,model,protocol,quality,model_TAG):
    meter_panel = Meter_Panel(protocol['metric_names'])
    with torch.no_grad():
        model.train(False)
        end = time.time()
        for i, input in enumerate(validation_loader):
            input = dict_to_device(input,device)
            protocol = update_protocol(input,quality,protocol)
            output = model(input,protocol)
            evaluation = meter_panel.eval(input,output,protocol)
            batch_time = time.time() - end
            meter_panel.update(evaluation,len(input['img']))
            meter_panel.update({'batch_time':batch_time})
            end = time.time()
        save_img(input['img'][-1].unsqueeze(0),'./output/img/image.png')
        save_img(output['compression']['img'][-1].unsqueeze(0),'./output/img/image_{}_{}.png'.format(model_TAG,quality))
    return meter_panel

def init_test_protocol(dataset):
    protocol = {}
    protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
    protocol['metric_names'] = config.PARAM['test_metric_names']
    protocol['format'] = format
    protocol['filename'] = 'tmp'
    protocol['sampling_factor'] = None
    protocol['loss_mode'] = {'compression':'mae'}
    return protocol

def update_protocol(input,quality,protocol):
    protocol['quality'] = quality
    return protocol

def print_result(quality,test_meter_panel):
    print('Test ({}){}'.format(quality,test_meter_panel.summary(['loss']+config.PARAM['test_metric_names'])))
    return
    
if __name__ == "__main__":
    main()