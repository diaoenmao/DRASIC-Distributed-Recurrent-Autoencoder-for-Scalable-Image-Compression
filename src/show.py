import numpy as np
import cv2
import config
import copy
import os
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from utils import *

config.init()
for k in config.PARAM:
    exec('{0} = config.PARAM[\'{0}\']'.format(k))
seeds = list(range(init_seed,init_seed+num_Experiments))
metric_names = config.PARAM['test_metric_names']
fig_format = 'eps'
plt.rc('font', family='sans-serif')
plt.rcParams.update({'font.size': 12})
             
def main():
    codec_result = process_codec_model()
    single_result = process_single_model()
    full_result = process_full_model()
    half_result = process_half_model()
    
    plt_codec(codec_result,single_result)

    plt_full_subset_mean(full_result,single_result)
    plt_full_class_mean(full_result,single_result)
    plt_full_subset_band(full_result,single_result)
    plt_full_class_band(full_result,single_result)

    plt_half_subset_mean(half_result,single_result)
    plt_half_class_mean(half_result,single_result)
    plt_half_subset_band(half_result,single_result)
    plt_half_class_band(half_result,single_result)
    return
    
def process_codec_model():
    result_names = os.listdir('./output/result/')
    codec_result = {}
    for i in range(len(result_names)):
        result = load('./output/result/'+result_names[i])
        if(result_names[i].split('_')[2]!='Magick' and result_names[i].split('_')[2]!='tod'):
            continue
        model_TAG = '_'.join(result_names[i].split('_')[1:])[:-4]
        result_array = {'bpp':[],'psnr':[]}
        for j in range(len(result)):
            tmp_result = {'bpp':[],'psnr':[]}
            for k in range(len(result[j])):
                tmp_result['bpp'].append(result[j][k].panel['bpp'].avg)
                tmp_result['psnr'].append(result[j][k].panel['psnr'].avg)
            tmp_result['bpp'] = np.array(tmp_result['bpp'])
            tmp_result['psnr'] = np.array(tmp_result['psnr'])
            result_array['bpp'].append(tmp_result['bpp'])
            result_array['psnr'].append(tmp_result['psnr'])
            codec_result[model_TAG+'_{}'.format(j)] = tmp_result
        mean_result_array = copy.deepcopy(result_array)
        mean_result_array['bpp'] = np.mean(np.vstack(result_array['bpp']),axis=0)
        mean_result_array['psnr'] = np.mean(np.vstack(result_array['psnr']),axis=0)
        codec_result[model_TAG+'_mean'] = mean_result_array
    return codec_result
    
def process_single_model():
    result_names = os.listdir('./output/result/')
    single_result = {}
    for i in range(len(result_names)):
        result = load('./output/result/'+result_names[i])
        if(result_names[i].split('_')[-3]!='sin'):
            continue
        model_TAG = '_'.join(result_names[i].split('_')[1:])[:-4]
        result_array = {'bpp':[],'psnr':[]}
        for j in range(len(result)):
            tmp_result = {'bpp':[],'psnr':[]}
            for k in range(len(result[j])):
                tmp_result['bpp'].append(result[j][k].panel['bpp'].avg)
                tmp_result['psnr'].append(result[j][k].panel['psnr'].avg)
            tmp_result['bpp'] = np.array(tmp_result['bpp'])
            tmp_result['psnr'] = np.array(tmp_result['psnr'])
            result_array['bpp'].append(tmp_result['bpp'])
            result_array['psnr'].append(tmp_result['psnr'])
            single_result[model_TAG+'_{}'.format(j)] = tmp_result
        mean_result_array = copy.deepcopy(result_array)
        mean_result_array['bpp'] = np.mean(np.vstack(result_array['bpp']),axis=0)
        mean_result_array['psnr'] = np.mean(np.vstack(result_array['psnr']),axis=0)
        single_result[model_TAG+'_mean'] = mean_result_array
    return single_result
    
def process_full_model():
    result_names = os.listdir('./output/result/')
    full_result = {}
    for i in range(len(result_names)):
        result = load('./output/result/'+result_names[i])
        if(result_names[i].split('_')[-4]!='full' or result_names[i].split('_')[-3]=='sin'):
            continue
        model_TAG = '_'.join(result_names[i].split('_')[1:])[:-4]
        result_array = {'bpp':[],'psnr':[]}
        for j in range(len(result)):
            tmp_result = {'bpp':[],'psnr':[]}
            for k in range(len(result[j])):
                tmp_result['bpp'].append(result[j][k].panel['bpp'].avg)
                tmp_result['psnr'].append(result[j][k].panel['psnr'].avg)
            tmp_result['bpp'] = np.array(tmp_result['bpp'])
            tmp_result['psnr'] = np.array(tmp_result['psnr'])
            result_array['bpp'].append(tmp_result['bpp'])
            result_array['psnr'].append(tmp_result['psnr'])
            full_result[model_TAG+'_{}'.format(j)] = tmp_result
        mean_result_array = copy.deepcopy(result_array)
        mean_result_array['bpp'] = np.mean(np.vstack(result_array['bpp']),axis=0)
        mean_result_array['psnr'] = np.mean(np.vstack(result_array['psnr']),axis=0)
        max_result_array = copy.deepcopy(result_array)
        max_result_array['bpp'] = np.max(np.vstack(result_array['bpp']),axis=0)
        max_result_array['psnr'] = np.max(np.vstack(result_array['psnr']),axis=0)
        min_result_array = copy.deepcopy(result_array)
        min_result_array['bpp'] = np.min(np.vstack(result_array['bpp']),axis=0)
        min_result_array['psnr'] = np.min(np.vstack(result_array['psnr']),axis=0)
        full_result[model_TAG+'_mean'] = mean_result_array
        full_result[model_TAG+'_max'] = max_result_array
        full_result[model_TAG+'_min'] = min_result_array
    return full_result

def process_half_model():
    result_names = os.listdir('./output/result/')
    half_result = {}
    for i in range(len(result_names)):
        result = load('./output/result/'+result_names[i])
        if(result_names[i].split('_')[-4]!='half' or result_names[i].split('_')[-3]=='sin'):
            continue
        model_TAG = '_'.join(result_names[i].split('_')[1:])[:-4]
        result_array = {'bpp':[],'psnr':[]}
        for j in range(len(result)):
            tmp_result = {'bpp':[],'psnr':[]}
            for k in range(len(result[j])):
                tmp_result['bpp'].append(result[j][k].panel['bpp'].avg)
                tmp_result['psnr'].append(result[j][k].panel['psnr'].avg)
            tmp_result['bpp'] = np.array(tmp_result['bpp'])
            tmp_result['psnr'] = np.array(tmp_result['psnr'])
            result_array['bpp'].append(tmp_result['bpp'])
            result_array['psnr'].append(tmp_result['psnr'])
            half_result[model_TAG+'_{}'.format(j)] = tmp_result
        stack_bpp = np.vstack(result_array['bpp'])
        stack_psnr = np.vstack(result_array['psnr'])
        full_mean_result_array = copy.deepcopy(result_array)
        full_mean_result_array['bpp'] = np.mean(stack_bpp[:stack_bpp.shape[0]//2,:],axis=0)
        full_mean_result_array['psnr'] = np.mean(stack_psnr[:stack_psnr.shape[0]//2,:],axis=0)
        full_max_result_array = copy.deepcopy(result_array)
        full_max_result_array['bpp'] = np.max(stack_bpp[:stack_bpp.shape[0]//2,:],axis=0)
        full_max_result_array['psnr'] = np.max(stack_psnr[:stack_psnr.shape[0]//2,:],axis=0)
        full_min_result_array = copy.deepcopy(result_array)
        full_min_result_array['bpp'] = np.min(stack_bpp[:stack_bpp.shape[0]//2,:],axis=0)
        full_min_result_array['psnr'] = np.min(stack_psnr[:stack_psnr.shape[0]//2,:],axis=0)
        
        half_mean_result_array = copy.deepcopy(result_array)
        half_mean_result_array['bpp'] = np.mean(stack_bpp[stack_bpp.shape[0]//2:,:],axis=0)
        half_mean_result_array['psnr'] = np.mean(stack_psnr[stack_psnr.shape[0]//2:,:],axis=0)
        half_max_result_array = copy.deepcopy(result_array)
        half_max_result_array['bpp'] = np.max(stack_bpp[stack_bpp.shape[0]//2:,:],axis=0)
        half_max_result_array['psnr'] = np.max(stack_psnr[stack_psnr.shape[0]//2:,:],axis=0)
        half_min_result_array = copy.deepcopy(result_array)
        half_min_result_array['bpp'] = np.min(stack_bpp[stack_bpp.shape[0]//2:,:],axis=0)
        half_min_result_array['psnr'] = np.min(stack_psnr[stack_psnr.shape[0]//2:,:],axis=0)
        
        half_result[model_TAG+'_mean_full'] = full_mean_result_array
        half_result[model_TAG+'_max_full'] = full_max_result_array
        half_result[model_TAG+'_min_full'] = full_min_result_array
        
        half_result[model_TAG+'_mean_half'] = half_mean_result_array
        half_result[model_TAG+'_max_half'] = half_max_result_array
        half_result[model_TAG+'_min_half'] = half_min_result_array

    return half_result

def plt_codec(codec_result,single_result):
    head = 'MNIST_iter_shuffle_codec_'
    colors = ['darkorange','blue','royalblue','lightskyblue']
    model_names = ['MNIST_tod_full_sin_class_10','MNIST_Magick_bpg','MNIST_Magick_jp2','MNIST_Magick_jpg']
    labels = ['Toderici (Baseline)','BPG','JPEG2000','JPEG']
    plt.figure('codec')
    x_sin = single_result[head+'full_sin_class_10_mean']['bpp']
    y_sin = single_result[head+'full_sin_class_10_mean']['psnr']
    plt.plot(x_sin,y_sin,color='red',linestyle='-',label='Ours',linewidth=3)
    for j in range(len(model_names)):
        x = codec_result[model_names[j]+'_mean']['bpp']
        y = codec_result[model_names[j]+'_mean']['psnr']      
        label = labels[j]
        plt.plot(x,y,color=colors[j],linestyle='-',label=label,linewidth=3)            
    plt.xlim(0,6)
    plt.ylim(10,62.5)
    plt.xlabel('BPP')
    plt.ylabel('PSNR')
    plt.grid()
    plt.legend(loc='lower right')
    makedir_exist_ok('./output/fig')
    plt.savefig('./output/fig/codec.{}'.format(fig_format),dpi=300,bbox_inches='tight',pad_inches=0)
    plt.close('codec')
        
def plt_full_subset_mean(full_result,single_result):
    head = 'MNIST_iter_shuffle_codec_'
    num_node = [2,4,8,10]
    colors = ['red','gold']
    for i in range(len(num_node)):
        model_names = ['full_dis_subset_'+str(num_node[i]),'full_sep_subset_'+str(num_node[i])]
        plt.figure('full_subset_mean_{}'.format(str(num_node[i])))
        x_sin = single_result[head+'full_sin_class_10_mean']['bpp']
        y_sin = single_result[head+'full_sin_class_10_mean']['psnr']
        plt.plot(x_sin,y_sin,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3) 
        for j in range(len(model_names)):
            x = full_result[head+model_names[j]+'_mean']['bpp']
            y = full_result[head+model_names[j]+'_mean']['psnr']
            label = ''
            list_model_names = model_names[j].split('_')            
            label += 'Distributed' if(list_model_names[1] == 'dis') else 'Separate'
            label += '(Full)'
            label += ' m={}'.format(str(num_node[i]))
            plt.plot(x,y,color=colors[j],linestyle='-',label=label,linewidth=3)            
        plt.xlim(0,2.125)
        plt.ylim(20,62.5)
        plt.xlabel('BPP',fontsize=12)
        plt.ylabel('PSNR',fontsize=12)
        plt.grid()
        plt.legend(loc='lower right')
        makedir_exist_ok('./output/fig')
        plt.savefig('./output/fig/full_subset_mean_{}.{}'.format(str(num_node[i]),fig_format),dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close('full_subset_mean_{}'.format(str(num_node[i])))
    
def plt_full_class_mean(full_result,single_result):
    head = 'MNIST_iter_shuffle_codec_'
    num_node = [2,4,8,10]
    colors = ['blue','lightskyblue']
    for i in range(len(num_node)):
        model_names = ['full_dis_class_'+str(num_node[i]),'full_sep_class_'+str(num_node[i])]
        plt.figure('full_class_mean_{}'.format(str(num_node[i])))
        x_sin = single_result[head+'full_sin_class_{}_mean'.format(str(num_node[i]))]['bpp']
        y_sin = single_result[head+'full_sin_class_{}_mean'.format(str(num_node[i]))]['psnr']
        plt.plot(x_sin,y_sin,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3) 
        for j in range(len(model_names)):
            x = full_result[head+model_names[j]+'_mean']['bpp']
            y = full_result[head+model_names[j]+'_mean']['psnr']
            label = ''
            list_model_names = model_names[j].split('_')            
            label += 'Distributed' if(list_model_names[1] == 'dis') else 'Separate'
            label += '(Full)'
            label += ' m={}'.format(str(num_node[i]))
            plt.plot(x,y,color=colors[j],linestyle='-',label=label,linewidth=3)
        plt.xlim(0,2.125)
        plt.ylim(20,62.5)
        plt.xlabel('BPP',fontsize=12)
        plt.ylabel('PSNR',fontsize=12)
        plt.grid()
        plt.legend(loc='lower right')
        makedir_exist_ok('./output/fig')
        plt.savefig('./output/fig/full_class_mean_{}.{}'.format(str(num_node[i]),fig_format),dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close('full_class_mean_{}'.format(str(num_node[i])))

def plt_half_subset_mean(half_result,single_result):
    head = 'MNIST_iter_shuffle_codec_'
    num_node = [2,4,8,10]
    colors = ['red','gold']
    for i in range(len(num_node)):
        model_names = ['half_dis_subset_'+str(num_node[i]),'half_sep_subset_'+str(num_node[i])]
        plt.figure('half_subset_mean_{}'.format(str(num_node[i])))
        x_sin = single_result[head+'full_sin_class_10_mean']['bpp']
        y_sin = single_result[head+'full_sin_class_10_mean']['psnr']
        plt.plot(x_sin,y_sin,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3) 
        for j in range(len(model_names)):
            x = half_result[head+model_names[j]+'_mean_full']['bpp']
            y = half_result[head+model_names[j]+'_mean_full']['psnr']
            label = ''
            list_model_names = model_names[j].split('_')            
            label += 'Distributed' if(list_model_names[1] == 'dis') else 'Separate'
            label += '(Full)'
            label += ' m={}'.format(str(num_node[i]))
            plt.plot(x,y,color=colors[j],linestyle='-',label=label,linewidth=3)
        x_sin = single_result[head+'half_sin_class_10_mean']['bpp']
        y_sin = single_result[head+'half_sin_class_10_mean']['psnr']
        plt.plot(x_sin,y_sin,color='black',linestyle='--',label='Joint(Half) m=1',linewidth=3)
        for j in range(len(model_names)):
            x = half_result[head+model_names[j]+'_mean_half']['bpp']
            y = half_result[head+model_names[j]+'_mean_half']['psnr']
            label = ''
            list_model_names = model_names[j].split('_')            
            label += 'Distributed' if(list_model_names[1] == 'dis') else 'Separate'
            label += '(Half)'
            label += ' m={}'.format(str(num_node[i]))
            plt.plot(x,y,color=colors[j],linestyle='--',label=label,linewidth=3)
        plt.xlim(0,2.125)
        plt.ylim(20,62.5)
        plt.xlabel('BPP',fontsize=12)
        plt.ylabel('PSNR',fontsize=12)
        plt.grid()
        plt.legend(loc='lower right')
        makedir_exist_ok('./output/fig')
        plt.savefig('./output/fig/half_subset_mean_{}.{}'.format(str(num_node[i]),fig_format),dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close('half_subset_mean_{}'.format(str(num_node[i])))
        
def plt_half_class_mean(half_result,single_result):
    head = 'MNIST_iter_shuffle_codec_'
    num_node = [2,4,8,10]
    colors = ['blue','lightskyblue']
    for i in range(len(num_node)):
        model_names = ['half_dis_class_'+str(num_node[i]),'half_sep_class_'+str(num_node[i])]
        plt.figure('half_class_mean_{}'.format(str(num_node[i])))
        x_sin = single_result[head+'full_sin_class_{}_mean'.format(str(num_node[i]))]['bpp']
        y_sin = single_result[head+'full_sin_class_{}_mean'.format(str(num_node[i]))]['psnr']
        plt.plot(x_sin,y_sin,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3)
        for j in range(len(model_names)):
            x = half_result[head+model_names[j]+'_mean_full']['bpp']
            y = half_result[head+model_names[j]+'_mean_full']['psnr']
            label = ''
            list_model_names = model_names[j].split('_')            
            label += 'Distributed' if(list_model_names[1] == 'dis') else 'Separate'
            label += '(Full)'
            label += ' m={}'.format(str(num_node[i]))
            plt.plot(x,y,color=colors[j],linestyle='-',label=label,linewidth=3)
        x_sin = single_result[head+'half_sin_class_{}_mean'.format(str(num_node[i]))]['bpp']
        y_sin = single_result[head+'half_sin_class_{}_mean'.format(str(num_node[i]))]['psnr']
        plt.plot(x_sin,y_sin,color='black',linestyle='--',label='Joint(Half) m=1',linewidth=3)
        for j in range(len(model_names)):
            x = half_result[head+model_names[j]+'_mean_half']['bpp']
            y = half_result[head+model_names[j]+'_mean_half']['psnr']
            label = ''
            list_model_names = model_names[j].split('_')            
            label += 'Distributed' if(list_model_names[1] == 'dis') else 'Separate'
            label += '(Half)'
            label += ' m={}'.format(str(num_node[i]))
            plt.plot(x,y,color=colors[j],linestyle='--',label=label,linewidth=3)            
        plt.xlim(0,2.125)
        plt.ylim(20,62.5)
        plt.xlabel('BPP',fontsize=12)
        plt.ylabel('PSNR',fontsize=12)
        plt.grid()
        plt.legend(loc='lower right')
        makedir_exist_ok('./output/fig')
        plt.savefig('./output/fig/half_class_mean_{}.{}'.format(str(num_node[i]),fig_format),dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close('half_class_mean_{}'.format(str(num_node[i])))
        
def plt_full_subset_band(full_result,single_result):
    head = 'MNIST_iter_shuffle_codec_'
    num_node = [2,4,8,10]
    colors = ['red','gold']
    for i in range(len(num_node)):
        model_names = ['full_dis_subset_'+str(num_node[i]),'full_sep_subset_'+str(num_node[i])]
        plt.figure('full_subset_band_{}'.format(str(num_node[i])))
        x_sin = single_result[head+'full_sin_class_10_mean']['bpp']
        y_sin = single_result[head+'full_sin_class_10_mean']['psnr']
        plt.plot(x_sin,y_sin,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3) 
        for j in range(len(model_names)):
            x = full_result[head+model_names[j]+'_mean']['bpp']
            y = full_result[head+model_names[j]+'_mean']['psnr']
            y_max = full_result[head+model_names[j]+'_max']['psnr']
            y_min = full_result[head+model_names[j]+'_min']['psnr']
            label = ''
            list_model_names = model_names[j].split('_')            
            label += 'Distributed' if(list_model_names[1] == 'dis') else 'Separate'
            label += '(Full)'
            label += ' m={}'.format(str(num_node[i]))
            plt.plot(x,y,color=colors[j],linestyle='-',label=label,linewidth=3)
            plt.fill_between(x,y_max,y_min,color=colors[j],alpha=0.5,linewidth=1)
        plt.xlim(0,2.125)
        plt.ylim(20,62.5)
        plt.xlabel('BPP',fontsize=12)
        plt.ylabel('PSNR',fontsize=12)
        plt.grid()
        plt.legend(loc='lower right')
        makedir_exist_ok('./output/fig')
        plt.savefig('./output/fig/full_subset_band_{}.{}'.format(str(num_node[i]),fig_format),dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close('full_subset_band_{}'.format(str(num_node[i])))
        
def plt_full_class_band(full_result,single_result):
    head = 'MNIST_iter_shuffle_codec_'
    num_node = [2,4,8,10]
    colors = ['blue','lightskyblue']
    for i in range(len(num_node)):
        model_names = ['full_dis_class_'+str(num_node[i]),'full_sep_class_'+str(num_node[i])]
        plt.figure('full_class_band_{}'.format(str(num_node[i])))
        x_sin = single_result[head+'full_sin_class_{}_mean'.format(str(num_node[i]))]['bpp']
        y_sin = single_result[head+'full_sin_class_{}_mean'.format(str(num_node[i]))]['psnr']
        plt.plot(x_sin,y_sin,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3)
        for j in range(len(model_names)):
            x = full_result[head+model_names[j]+'_mean']['bpp']
            y = full_result[head+model_names[j]+'_mean']['psnr']
            y_max = full_result[head+model_names[j]+'_max']['psnr']
            y_min = full_result[head+model_names[j]+'_min']['psnr']
            label = ''
            list_model_names = model_names[j].split('_')            
            label += 'Distributed' if(list_model_names[1] == 'dis') else 'Separate'
            label += '(Full)'
            label += ' m={}'.format(str(num_node[i]))
            plt.plot(x,y,color=colors[j],linestyle='-',label=label,linewidth=3)
            plt.fill_between(x,y_max,y_min,color=colors[j],alpha=0.5,linewidth=1)
        plt.xlim(0,2.125)
        plt.ylim(20,62.5)
        plt.xlabel('BPP',fontsize=12)
        plt.ylabel('PSNR',fontsize=12)
        plt.grid()
        plt.legend(loc='lower right')
        makedir_exist_ok('./output/fig')
        plt.savefig('./output/fig/full_class_band_{}.{}'.format(str(num_node[i]),fig_format),dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close('full_class_band_{}'.format(str(num_node[i])))
        
def plt_half_subset_band(half_result,single_result):
    head = 'MNIST_iter_shuffle_codec_'
    num_node = [2,4,8,10]
    colors = ['red','gold']
    for i in range(len(num_node)):
        model_names = ['half_dis_subset_'+str(num_node[i]),'half_sep_subset_'+str(num_node[i])]
        plt.figure('half_subset_band_{}'.format(str(num_node[i])))
        x_sin = single_result[head+'full_sin_class_10_mean']['bpp']
        y_sin = single_result[head+'full_sin_class_10_mean']['psnr']
        plt.plot(x_sin,y_sin,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3) 
        for j in range(len(model_names)):
            x = half_result[head+model_names[j]+'_mean_full']['bpp']
            y = half_result[head+model_names[j]+'_mean_full']['psnr']
            y_max = half_result[head+model_names[j]+'_max_full']['psnr']
            y_min = half_result[head+model_names[j]+'_min_full']['psnr']
            label = ''
            list_model_names = model_names[j].split('_')            
            label += 'Distributed' if(list_model_names[1] == 'dis') else 'Separate'
            label += '(Full)'
            label += ' m={}'.format(str(num_node[i]))
            plt.plot(x,y,color=colors[j],linestyle='-',label=label,linewidth=3)
            plt.fill_between(x,y_max,y_min,color=colors[j],alpha=0.5,linewidth=1)
        x_sin = single_result[head+'half_sin_class_10_mean']['bpp']
        y_sin = single_result[head+'half_sin_class_10_mean']['psnr']
        plt.plot(x_sin,y_sin,color='black',linestyle='--',label='Joint(Half) m=1',linewidth=3) 
        for j in range(len(model_names)):
            x = half_result[head+model_names[j]+'_mean_half']['bpp']
            y = half_result[head+model_names[j]+'_mean_half']['psnr']
            y_max = half_result[head+model_names[j]+'_max_half']['psnr']
            y_min = half_result[head+model_names[j]+'_min_half']['psnr']
            label = ''
            list_model_names = model_names[j].split('_')            
            label += 'Distributed' if(list_model_names[1] == 'dis') else 'Separate'
            label += '(Half)'
            label += ' m={}'.format(str(num_node[i]))
            plt.plot(x,y,color=colors[j],linestyle='--',label=label,linewidth=3)
            plt.fill_between(x,y_max,y_min,color=colors[j],alpha=0.5,linewidth=1)
        plt.xlim(0,2.125)
        plt.ylim(20,62.5)
        plt.xlabel('BPP',fontsize=12)
        plt.ylabel('PSNR',fontsize=12)
        plt.grid()
        plt.legend(loc='lower right')
        makedir_exist_ok('./output/fig')
        plt.savefig('./output/fig/half_subset_band_{}.{}'.format(str(num_node[i]),fig_format),dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close('half_subset_band_{}'.format(str(num_node[i])))

def plt_half_class_band(half_result,single_result):
    head = 'MNIST_iter_shuffle_codec_'
    num_node = [2,4,8,10]
    colors = ['blue','lightskyblue']
    for i in range(len(num_node)):
        model_names = ['half_dis_class_'+str(num_node[i]),'half_sep_class_'+str(num_node[i])]
        plt.figure('half_class_band_{}'.format(str(num_node[i])))
        x_sin = single_result[head+'full_sin_class_{}_mean'.format(str(num_node[i]))]['bpp']
        y_sin = single_result[head+'full_sin_class_{}_mean'.format(str(num_node[i]))]['psnr']
        plt.plot(x_sin,y_sin,color='black',linestyle='-',label='Joint(Half) m=1',linewidth=3)
        for j in range(len(model_names)):
            x = half_result[head+model_names[j]+'_mean_full']['bpp']
            y = half_result[head+model_names[j]+'_mean_full']['psnr']
            y_max = half_result[head+model_names[j]+'_max_full']['psnr']
            y_min = half_result[head+model_names[j]+'_min_full']['psnr']
            label = ''
            list_model_names = model_names[j].split('_')            
            label += 'Distributed' if(list_model_names[1] == 'dis') else 'Separate'
            label += '(Full)'
            label += ' m={}'.format(str(num_node[i]))
            plt.plot(x,y,color=colors[j],linestyle='-',label=label,linewidth=3)
            plt.fill_between(x,y_max,y_min,color=colors[j],alpha=0.5,linewidth=1)
        x_sin = single_result[head+'half_sin_class_{}_mean'.format(str(num_node[i]))]['bpp']
        y_sin = single_result[head+'half_sin_class_{}_mean'.format(str(num_node[i]))]['psnr']
        plt.plot(x_sin,y_sin,color='black',linestyle='--',label='Joint(Half) m=1',linewidth=3)
        for j in range(len(model_names)):
            x = half_result[head+model_names[j]+'_mean_half']['bpp']
            y = half_result[head+model_names[j]+'_mean_half']['psnr']
            y_max = half_result[head+model_names[j]+'_max_half']['psnr']
            y_min = half_result[head+model_names[j]+'_min_half']['psnr']
            label = ''
            list_model_names = model_names[j].split('_')            
            label += 'Distributed' if(list_model_names[1] == 'dis') else 'Separate'
            label += '(Half)'
            label += ' m={}'.format(str(num_node[i]))
            plt.plot(x,y,color=colors[j],linestyle='--',label=label,linewidth=3)
            plt.fill_between(x,y_max,y_min,color=colors[j],alpha=0.5,linewidth=1)
        plt.xlim(0,2.125)
        plt.ylim(20,62.5)
        plt.xlabel('BPP',fontsize=12)
        plt.ylabel('PSNR',fontsize=12)
        plt.grid()
        plt.legend(loc='lower right')
        makedir_exist_ok('./output/fig')
        plt.savefig('./output/fig/half_class_band_{}.{}'.format(str(num_node[i]),fig_format),dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close('half_class_band_{}'.format(str(num_node[i])))
        
if __name__ == "__main__":
   main()
