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
            
def main():
    plt_result_classic()
    plt_result()
    plt_result_sup()
    return

def plt_result_classic():
    model_head = '0_MNIST_iter_shuffle_codec'
    result_names = os.listdir('./output/result/')
    resume_TAG_mode = [['0_MNIST_iter_shuffle_codec_full_sin_class_10','0_MNIST_tod_full_sin_class_10','0_MNIST_Magick_bpg','0_MNIST_Magick_jp2','0_MNIST_Magick_jpg']]
    resume_TAGs = list(itertools.product(*resume_TAG_mode))
    processed_result = {}
    for i in range(len(resume_TAGs)):
        resume_TAG = "_".join(list(resume_TAGs[i]))
        result = load('./output/result/{}.pkl'.format(resume_TAG))
        processed_result[resume_TAG] = {'bpp':np.zeros((1,len(result[0]))),'psnr':np.zeros((1,len(result[0])))}
        for k in range(len(result)):
            for j in range(len(result[k])):
                processed_result[resume_TAG]['bpp'][k,j] = result[k][j].panel['bpp'].avg
                processed_result[resume_TAG]['psnr'][k,j] = result[k][j].panel['psnr'].avg
                
    colors = ['red','darkorange','blue','royalblue','lightskyblue']            
    labels = ['Ours','Toderici (Baseline)','BPG','JPEG2000','JPEG']
    plt.figure('classic',dpi=300)
    for i in range(len(resume_TAGs)):
        resume_TAG = "_".join(list(resume_TAGs[i]))
        bpp = processed_result[resume_TAG]['bpp']
        psnr = processed_result[resume_TAG]['psnr']
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)        
        plt.plot(x,y,color=colors[i],linestyle='-',label=labels[i],linewidth=3)    
    plt.xlim(0,6)
    plt.ylim(10,62.5)
    plt.xlabel('BPP')
    plt.ylabel('PSNR')
    plt.grid()
    plt.legend(loc='lower right',fontsize=9)
    makedir_exist_ok('./output/fig')
    plt.savefig('./output/fig/classic.{}'.format(fig_format),bbox_inches='tight',pad_inches=0)
    plt.close('classic')
    return
    
def plt_result():
    model_head = '0_MNIST_iter_shuffle_codec'
    result_names = os.listdir('./output/result/')
    resume_TAG_mode = [[model_head],['full','half'],['sin_class','dis_subset','dis_class','sep_subset','sep_class'],['2','4','8','10']]
    resume_TAGs = list(itertools.product(*resume_TAG_mode))
    processed_result = {}
    for i in range(len(resume_TAGs)):
        resume_TAG = "_".join(list(resume_TAGs[i]))
        result = load('./output/result/{}.pkl'.format(resume_TAG))
        if(resume_TAG.split('_')[-3]=='sin'):
            processed_result[resume_TAG] = {'bpp':np.zeros((1,num_iter)),'psnr':np.zeros((1,num_iter))}
        else:
            processed_result[resume_TAG] = {'bpp':np.zeros((int(resume_TAG.split('_')[-1]),num_iter)),'psnr':np.zeros((int(resume_TAG.split('_')[-1]),num_iter))}
        for k in range(len(result)):
            for j in range(len(result[k])):
                processed_result[resume_TAG]['bpp'][k,j] = result[k][j].panel['bpp'].avg
                processed_result[resume_TAG]['psnr'][k,j] = result[k][j].panel['psnr'].avg

    model_head = '0_MNIST_iter_shuffle_codec'
    num_node = ['2','4','8','10']
    
    plt.figure('full_subset_band',figsize=figsize,dpi=300)
    for i in range(len(num_node)):
        plt.subplot(2,2,i+1)
        
        model_name = '{}_full_sin_class_10'.format(model_head)
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr']
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        plt.plot(x,y,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3)
        
        model_name = '{}_full_dis_subset_{}'.format(model_head,num_node[i]) 
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr']        
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        y_max = psnr.max(axis=0)
        y_min = psnr.min(axis=0)
        plt.plot(x,y,color='red',linestyle='-',label='Distributed(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='red',alpha=0.5,linewidth=1)
        
        model_name = '{}_full_sep_subset_{}'.format(model_head,num_node[i])
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        y_max = psnr.max(axis=0)
        y_min = psnr.min(axis=0)
        plt.plot(x,y,color='gold',linestyle='-',label='Separate(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='gold',alpha=0.5,linewidth=1)
        
        plt.xlim(0,2.125)
        plt.ylim(20,62.5)
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
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        plt.plot(x,y,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3)
        
        model_name = '{}_full_dis_class_{}'.format(model_head,num_node[i])
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr']         
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        y_max = psnr.max(axis=0)
        y_min = psnr.min(axis=0)
        plt.plot(x,y,color='blue',linestyle='-',label='Distributed(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='blue',alpha=0.5,linewidth=1)
        
        model_name = '{}_full_sep_class_{}'.format(model_head,num_node[i])
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        y_max = psnr.max(axis=0)
        y_min = psnr.min(axis=0)
        plt.plot(x,y,color='lightskyblue',linestyle='-',label='Separate(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='lightskyblue',alpha=0.5,linewidth=1)
        
        plt.xlim(0,2.125)
        plt.ylim(20,62.5)
        plt.xlabel('BPP')
        plt.ylabel('PSNR')
        plt.grid()
        plt.legend(loc='upper left',fontsize=9)
    makedir_exist_ok('./output/fig')
    plt.savefig('./output/fig/full_class_band.{}'.format(fig_format),bbox_inches='tight',pad_inches=0)
    plt.close('full_class_band')

    plt.figure('half_subset_band',figsize=figsize,dpi=300)
    for i in range(len(num_node)):
        plt.subplot(2,2,i+1)

        model_name = '{}_full_sin_class_10'.format(model_head)
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        plt.plot(x,y,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3)
        
        model_name = '{}_half_sin_class_10'.format(model_head)
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        plt.plot(x,y,color='black',linestyle='--',label='Joint(Half) m=1',linewidth=3)
        
        model_name = '{}_half_dis_subset_{}'.format(model_head,num_node[i])        
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp[:bpp.shape[0]//2].mean(axis=0)
        y = psnr[:psnr.shape[0]//2].mean(axis=0)
        y_max = psnr[:psnr.shape[0]//2].max(axis=0)
        y_min = psnr[:psnr.shape[0]//2].min(axis=0)
        plt.plot(x,y,color='red',linestyle='-',label='Distributed(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='red',alpha=0.5,linewidth=1)

        x = bpp[bpp.shape[0]//2:].mean(axis=0)
        y = psnr[psnr.shape[0]//2:].mean(axis=0)
        y_max = psnr[psnr.shape[0]//2:].max(axis=0)
        y_min = psnr[psnr.shape[0]//2:].min(axis=0)
        plt.plot(x,y,color='red',linestyle='--',label='Distributed(Half) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='red',alpha=0.5,linewidth=1)
        
        model_name = '{}_half_sep_subset_{}'.format(model_head,num_node[i])
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp[:bpp.shape[0]//2].mean(axis=0)
        y = psnr[:psnr.shape[0]//2].mean(axis=0)
        y_max = psnr[:psnr.shape[0]//2].max(axis=0)
        y_min = psnr[:psnr.shape[0]//2].min(axis=0)
        plt.plot(x,y,color='gold',linestyle='-',label='Separate(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='gold',alpha=0.5,linewidth=1)
 
        x = bpp[bpp.shape[0]//2:].mean(axis=0)
        y = psnr[psnr.shape[0]//2:].mean(axis=0)
        y_max = psnr[psnr.shape[0]//2:].max(axis=0)
        y_min = psnr[psnr.shape[0]//2:].min(axis=0)
        plt.plot(x,y,color='gold',linestyle='--',label='Separate(Half) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='gold',alpha=0.5,linewidth=1)
        
        plt.xlim(0,2.125)
        plt.ylim(20,62.5)
        plt.xlabel('BPP')
        plt.ylabel('PSNR')
        plt.grid()
        plt.legend(loc='upper left',fontsize=9)
    makedir_exist_ok('./output/fig')
    plt.savefig('./output/fig/half_subset_band.{}'.format(fig_format),bbox_inches='tight',pad_inches=0)
    plt.close('half_subset_band')
        
    plt.figure('half_class_band',figsize=figsize,dpi=300)
    for i in range(len(num_node)):
        plt.subplot(2,2,i+1)

        model_name = '{}_full_sin_class_{}'.format(model_head,num_node[i])
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        plt.plot(x,y,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3)
        
        model_name = '{}_half_sin_class_{}'.format(model_head,num_node[i])
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        plt.plot(x,y,color='black',linestyle='--',label='Joint(Half) m=1',linewidth=3)
        
        model_name = '{}_half_dis_class_{}'.format(model_head,num_node[i])        
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp[:bpp.shape[0]//2].mean(axis=0)
        y = psnr[:psnr.shape[0]//2].mean(axis=0)
        y_max = psnr[:psnr.shape[0]//2].max(axis=0)
        y_min = psnr[:psnr.shape[0]//2].min(axis=0)
        plt.plot(x,y,color='blue',linestyle='-',label='Distributed(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='blue',alpha=0.5,linewidth=1)
 
        x = bpp[bpp.shape[0]//2:].mean(axis=0)
        y = psnr[psnr.shape[0]//2:].mean(axis=0)
        y_max = psnr[psnr.shape[0]//2:].max(axis=0)
        y_min = psnr[psnr.shape[0]//2:].min(axis=0)
        plt.plot(x,y,color='blue',linestyle='--',label='Distributed(Half) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='blue',alpha=0.5,linewidth=1)
        
        model_name = '{}_half_sep_class_{}'.format(model_head,num_node[i])
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp[:bpp.shape[0]//2].mean(axis=0)
        y = psnr[:psnr.shape[0]//2].mean(axis=0)
        y_max = psnr[:psnr.shape[0]//2].max(axis=0)
        y_min = psnr[:psnr.shape[0]//2].min(axis=0)
        plt.plot(x,y,color='lightskyblue',linestyle='-',label='Separate(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='lightskyblue',alpha=0.5,linewidth=1)

        x = bpp[bpp.shape[0]//2:].mean(axis=0)
        y = psnr[psnr.shape[0]//2:].mean(axis=0)
        y_max = psnr[psnr.shape[0]//2:].max(axis=0)
        y_min = psnr[psnr.shape[0]//2:].min(axis=0)
        plt.plot(x,y,color='lightskyblue',linestyle='--',label='Separate(Half) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='lightskyblue',alpha=0.5,linewidth=1)
        
        plt.xlim(0,2.125)
        plt.ylim(20,62.5)
        plt.xlabel('BPP')
        plt.ylabel('PSNR')
        plt.grid()
        plt.legend(loc='upper left',fontsize=9)
    makedir_exist_ok('./output/fig')
    plt.savefig('./output/fig/half_class_band.{}'.format(fig_format),bbox_inches='tight',pad_inches=0)
    plt.close('half_class_band')
    
def plt_result_sup():
    model_head = '0_MNIST_shuffle_codec'
    result_names = os.listdir('./output/result/')
    resume_TAG_mode = [[model_head],['full'],['sin_class','dis_subset','dis_class','sep_subset','sep_class'],['2','4','8','10']]
    code_size = ['8','32','64','96','128']
    resume_TAGs = list(itertools.product(*resume_TAG_mode))
    processed_result = {}
    for i in range(len(resume_TAGs)):
        resume_TAG = "_".join(list(resume_TAGs[i]))
        if(resume_TAG.split('_')[-3]=='sin'):
            processed_result[resume_TAG] = {'bpp':np.zeros((1,len(code_size))),'psnr':np.zeros((1,len(code_size)))}
        else:
            processed_result[resume_TAG] = {'bpp':np.zeros((int(resume_TAG.split('_')[-1]),len(code_size))),'psnr':np.zeros((int(resume_TAG.split('_')[-1]),len(code_size)))}
        for j in range(len(code_size)):
            result = load('./output/result/{}_{}.pkl'.format(resume_TAG,code_size[j]))
            for k in range(len(result)):
                processed_result[resume_TAG]['bpp'][k,j] = result[k].panel['bpp'].avg
                processed_result[resume_TAG]['psnr'][k,j] = result[k].panel['psnr'].avg
                
    model_head = '0_MNIST_shuffle_codec'
    num_node = ['2','4','8','10']
    
    plt.figure('sup_full_subset_band',figsize=figsize,dpi=300)
    for i in range(len(num_node)):
        plt.subplot(2,2,i+1)

        model_name = '{}_full_sin_class_10'.format(model_head)
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr']
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        plt.plot(x,y,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3)
        
        model_name = '{}_full_dis_subset_{}'.format(model_head,num_node[i]) 
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr']        
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        y_max = psnr.max(axis=0)
        y_min = psnr.min(axis=0)
        plt.plot(x,y,color='red',linestyle='-',label='Distributed(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='red',alpha=0.5,linewidth=1)
        
        model_name = '{}_full_sep_subset_{}'.format(model_head,num_node[i])
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        y_max = psnr.max(axis=0)
        y_min = psnr.min(axis=0)
        plt.plot(x,y,color='gold',linestyle='-',label='Separate(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='gold',alpha=0.5,linewidth=1)
        
        plt.xlim(0,2.125)
        plt.ylim(16,32)
        plt.xlabel('BPP')
        plt.ylabel('PSNR')
        plt.grid()
        plt.legend(loc='upper left',fontsize=9)
    makedir_exist_ok('./output/fig')
    plt.savefig('./output/fig/sup_full_subset_band.{}'.format(fig_format),bbox_inches='tight',pad_inches=0)
    plt.close('sup_full_subset_band')
        
    plt.figure('sup_full_class_band',figsize=figsize,dpi=300)
    for i in range(len(num_node)):
        plt.subplot(2,2,i+1)
        
        model_name = '{}_full_sin_class_{}'.format(model_head,num_node[i])
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        plt.plot(x,y,color='black',linestyle='-',label='Joint(Full) m=1',linewidth=3)
        
        model_name = '{}_full_dis_class_{}'.format(model_head,num_node[i])
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr']         
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        y_max = psnr.max(axis=0)
        y_min = psnr.min(axis=0)
        plt.plot(x,y,color='blue',linestyle='-',label='Distributed(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='blue',alpha=0.5,linewidth=1)
        
        model_name = '{}_full_sep_class_{}'.format(model_head,num_node[i])
        bpp = processed_result[model_name]['bpp']
        psnr = processed_result[model_name]['psnr'] 
        x = bpp.mean(axis=0)
        y = psnr.mean(axis=0)
        y_max = psnr.max(axis=0)
        y_min = psnr.min(axis=0)
        plt.plot(x,y,color='lightskyblue',linestyle='-',label='Separate(Full) m={}'.format(str(num_node[i])),linewidth=3)
        plt.fill_between(x,y_max,y_min,color='lightskyblue',alpha=0.5,linewidth=1)
        
        plt.xlim(0,2.125)
        plt.ylim(16,32)
        plt.xlabel('BPP')
        plt.ylabel('PSNR')
        plt.grid()
        plt.legend(loc='upper left',fontsize=9)
    makedir_exist_ok('./output/fig')
    plt.savefig('./output/fig/sup_full_class_band.{}'.format(fig_format),bbox_inches='tight',pad_inches=0)
    plt.close('sup_full_class_band')

if __name__ == "__main__":
   main()