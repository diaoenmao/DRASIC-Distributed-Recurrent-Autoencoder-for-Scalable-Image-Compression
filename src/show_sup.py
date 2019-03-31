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
fig_format = 'png'
plt.rc('font', family='sans-serif')
plt.rcParams.update({'font.size': 12})
figsize = (16,9)

model_head = '0_MNIST_shuffle_codec'            
def main():
    processed_result = process_result()
    print(processed_result)
    plt_result(processed_result)
    return
    
def process_result():
    result_names = os.listdir('./output/result/')
    resume_TAG_mode = [['{}_full_sin_class'.format(model_head),'{}_full_dis_subset'.format(model_head),'{}_full_dis_class'.format(model_head),'{}_full_sep_subset'.format(model_head),'{}_full_sep_class'.format(model_head)],['2','4','8','10']]
    code_size = ['8','32','64','96','128']
    resume_TAGs = list(itertools.product(*resume_TAG_mode))
    processed_result = {}
    for i in range(len(resume_TAGs)):
        list_resume_TAG = list(resume_TAGs[i])
        resume_TAG = "_".join(list_resume_TAG)
        if(list_resume_TAG[0].split('_')[-2]=='sin'):
            processed_result[resume_TAG] = {'bpp':np.zeros((1,len(code_size))),'psnr':np.zeros((1,len(code_size)))}
        else:
            processed_result[resume_TAG] = {'bpp':np.zeros((int(list_resume_TAG[1]),len(code_size))),'psnr':np.zeros((int(list_resume_TAG[1]),len(code_size)))}
        for j in range(len(code_size)):
            result = load('./output/result/{}_{}.pkl'.format(resume_TAG,code_size[j]))
            for k in range(len(result)):
                processed_result[resume_TAG]['bpp'][k,j] = result[k].panel['bpp'].avg
                processed_result[resume_TAG]['psnr'][k,j] = result[k].panel['psnr'].avg
    return processed_result
        
def plt_result(processed_result):
    num_node = ['2','4','8','10']
    plt.figure('full_subset_band',figsize=figsize,dpi=300)
    for i in range(len(num_node)):
        plt.subplot(2,2,i+1)
        
        model_name = '{}_full_sin_class_10'.format(model_head)
        x = processed_result[model_name]['bpp'].mean(axis=0)
        y = processed_result[model_name]['psnr'].mean(axis=0)
        plt.plot(x,y,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3)
        
        model_name = '{}_full_dis_subset_{}'.format(model_head,num_node[i])        
        x = processed_result[model_name]['bpp'].mean(axis=0)
        y = processed_result[model_name]['psnr'].mean(axis=0)
        y_max = processed_result[model_name]['psnr'].max(axis=0)
        y_min = processed_result[model_name]['psnr'].min(axis=0)
        plt.plot(x,y,color='red',linestyle='-',label='Distributed(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='red',alpha=0.5,linewidth=1)
        
        model_name = '{}_full_sep_subset_{}'.format(model_head,num_node[i])
        x = processed_result[model_name]['bpp'].mean(axis=0)
        y = processed_result[model_name]['psnr'].mean(axis=0)
        y_max = processed_result[model_name]['psnr'].max(axis=0)
        y_min = processed_result[model_name]['psnr'].min(axis=0)
        plt.plot(x,y,color='gold',linestyle='-',label='Separate(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='gold',alpha=0.5,linewidth=1)
        
        plt.xlim(0,2.125)
        plt.ylim(16,32)
        plt.xlabel('BPP')
        plt.ylabel('PSNR')
        plt.grid()
        plt.legend(loc='upper left',fontsize=9)
    makedir_exist_ok('./output/fig')
    plt.savefig('./output/fig/full_subset_band.{}'.format(fig_format),bbox_inches='tight',pad_inches=0)
    plt.close('full_subset_band')
        
    plt.figure('full_class_band',figsize=figsize,dpi=300)
    for i in range(len(num_node)):
        plt.subplot(2,2,i+1)
        
        model_name = '{}_full_sin_class_{}'.format(model_head,num_node[i])
        x = processed_result[model_name]['bpp'].mean(axis=0)
        y = processed_result[model_name]['psnr'].mean(axis=0)
        plt.plot(x,y,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3)
        
        model_name = '{}_full_dis_class_{}'.format(model_head,num_node[i])        
        x = processed_result[model_name]['bpp'].mean(axis=0)
        y = processed_result[model_name]['psnr'].mean(axis=0)
        y_max = processed_result[model_name]['psnr'].max(axis=0)
        y_min = processed_result[model_name]['psnr'].min(axis=0)
        plt.plot(x,y,color='blue',linestyle='-',label='Distributed(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='blue',alpha=0.5,linewidth=1)
        
        model_name = '{}_full_sep_class_{}'.format(model_head,num_node[i])
        x = processed_result[model_name]['bpp'].mean(axis=0)
        y = processed_result[model_name]['psnr'].mean(axis=0)
        y_max = processed_result[model_name]['psnr'].max(axis=0)
        y_min = processed_result[model_name]['psnr'].min(axis=0)
        plt.plot(x,y,color='lightskyblue',linestyle='-',label='Separate(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='lightskyblue',alpha=0.5,linewidth=1)
        
        plt.xlim(0,2.125)
        plt.ylim(16,32)
        plt.xlabel('BPP')
        plt.ylabel('PSNR')
        plt.grid()
        plt.legend(loc='upper left',fontsize=9)
    makedir_exist_ok('./output/fig')
    plt.savefig('./output/fig/full_class_band.{}'.format(fig_format),bbox_inches='tight',pad_inches=0)
    plt.close('full_class_band')        
if __name__ == "__main__":
   main()