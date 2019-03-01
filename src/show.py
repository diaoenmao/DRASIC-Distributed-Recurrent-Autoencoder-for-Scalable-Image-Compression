import numpy as np
import cv2
import config
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from utils import *

config.init()
for k in config.PARAM:
    exec('{0} = config.PARAM[\'{0}\']'.format(k))
seeds = list(range(init_seed,init_seed+num_Experiments))
metric_names = config.PARAM['test_metric_names']

#model_names = ['testnet_2','Joint','Joint_tod_lstm','Magick_bpg','Magick_jp2','Magick_jpg']
#model_names = ['Joint_cor','Joint_dist_cor','Joint_dist_cor_base']
model_names = ['Joint']
# model_names = ['Joint','Joint_dist_2_0','Joint_dist_2_1',\
                # 'Joint_dist_base_2_0','Joint_dist_base_2_1']
# model_names = ['Joint','Joint_dist_4_0','Joint_dist_4_1','Joint_dist_4_2','Joint_dist_4_3',\
                # 'Joint_dist_base_4_0','Joint_dist_base_4_1','Joint_dist_base_4_2','Joint_dist_base_4_3']
# model_names = ['Joint','Joint_dist_8_0','Joint_dist_8_1','Joint_dist_8_2','Joint_dist_8_3','Joint_dist_8_4','Joint_dist_8_5','Joint_dist_8_6','Joint_dist_8_7',\
                # 'Joint_dist_base_8_0','Joint_dist_base_8_1','Joint_dist_base_8_1','Joint_dist_base_8_2','Joint_dist_base_8_3','Joint_dist_base_8_4','Joint_dist_base_8_5','Joint_dist_base_8_6','Joint_dist_base_8_7']
# model_names = ['Joint','Joint_dist_2_0','Joint_dist_2_1',\
                # 'Joint_dist_4_0','Joint_dist_4_1','Joint_dist_4_2','Joint_dist_4_3',\
                # 'Joint_dist_8_0','Joint_dist_8_1','Joint_dist_8_2','Joint_dist_8_3','Joint_dist_8_4','Joint_dist_8_5','Joint_dist_8_6','Joint_dist_8_7',\
                # 'Joint_dist_base_2_0','Joint_dist_base_2_1',\
                # 'Joint_dist_base_4_0','Joint_dist_base_4_1','Joint_dist_base_4_2','Joint_dist_base_4_3',\
                # 'Joint_dist_base_8_0','Joint_dist_base_8_1','Joint_dist_base_8_1','Joint_dist_base_8_2','Joint_dist_base_8_3','Joint_dist_base_8_4','Joint_dist_base_8_5','Joint_dist_base_8_6','Joint_dist_base_8_7']
# supmodel_names = ['Joint','Joint_dist_2',\
                    # 'Joint_dist_base_2']
# supmodel_names = ['Joint','Joint_dist_4',\
                    # 'Joint_dist_base_4']
supmodel_names = ['Joint','Joint_dist_8',\
                    'Joint_dist_base_8']                    
# supmodel_names = ['Joint','Joint_dist_2','Joint_dist_4','Joint_dist_8','Joint_dist_base_2','Joint_dist_base_4','Joint_dist_base_8']

colors = sns.color_palette('Set1', n_colors=16)
# colors = ['black','red','blue']
#colors = ['black','darkorange','royalblue']
# colors = ['black','gold','lightskyblue']
#colors = ['red','darkorange','gold','blue','royalblue','lightskyblue']
default_colors = {model_names[i]:colors[i] for i in range(len(model_names))}
# default_colors = {supmodel_names[i]:colors[i] for i in range(len(supmodel_names))}

default_linestyles = {'testnet_2':'-','Joint':'-','Joint_tod_lstm':'-',\
                        'Magick_jpg':'-','Magick_jp2':'-','Magick_bpg':'-',\
                        'Joint_dist_2':'-','Joint_dist_4':'-','Joint_dist_8':'-',\
                        'Joint_dist_base_2':'--','Joint_dist_base_4':'--','Joint_dist_base_8':'--',\
                        'Joint_cor':'-','Joint_dist_cor':'-','Joint_dist_cor_base':'-'}
default_labels = {'testnet_2':'ResConvLSTM','Joint':'ConvLSTM','Joint_tod_lstm':'Toderici (Baseline)',\
                    'Magick_jpg':'JPEG','Magick_jp2':'JPEG2000','Magick_bpg':'BPG',\
                    'Joint_dist_2':'Distributed m=2','Joint_dist_4':'Distributed m=4','Joint_dist_8':'Distributed m=8',\
                    'Joint_dist_base_2':'Separate m=2','Joint_dist_base_4':'Separate m=4','Joint_dist_base_8':'Separate m=8',\
                    'Joint_cor':'Joint','Joint_dist_cor':'Distributed m=2','Joint_dist_cor_base':'Separate m=2'}
                    
                    
                    
def main():
    results = merge()
    results = process(results)
    show(results)
    return

# def merge():
    # results = {}
    # for i in range(len(model_names)):
        # results[model_names[i]] = {k:[[] for jj in range(num_Experiments)] for k in metric_names}    
        # for j in range(num_Experiments):
            # resume_model_TAG = '{}_{}_{}'.format(seeds[j],model_data_name,model_names[i]) if(resume_TAG=='') else '{}_{}_{}_{}'.format(seeds[j],model_data_name,model_names[i],resume_TAG)
            # model_TAG = resume_model_TAG if(special_TAG=='') else '{}_{}'.format(resume_model_TAG,special_TAG)
            # result = load('./output/result/{}.pkl'.format(model_TAG))
            # for n in range(len(result['result'])):
                # for k in metric_names:
                    # if(k in result['result'][n].panel):
                        # results[model_names[i]][k][j].append(result['result'][n].panel[k].avg)
        # for k in metric_names:
            # results[model_names[i]][k] = np.array(results[model_names[i]][k]).mean(axis=0,keepdims=False)
    # return results

def merge():
    results = {}
    for i in range(len(model_names)):
        results[model_names[i]] = {k:[[] for jj in range(num_Experiments)] for k in metric_names}    
        for j in range(num_Experiments):
            resume_model_TAG = '{}_{}_{}'.format(seeds[j],model_data_name,model_names[i]) if(resume_TAG=='') else '{}_{}_{}_{}'.format(seeds[j],model_data_name,model_names[i],resume_TAG)
            model_TAG = resume_model_TAG if(special_TAG=='') else '{}_{}'.format(resume_model_TAG,special_TAG)
            result = load('./output/result/{}.pkl'.format(model_TAG))
            for n in range(len(result['result'])):
                for k in metric_names:
                    if(k in result['result'][n].panel):
                        results[model_names[i]][k][j].append(result['result'][n].panel[k])
        for k in metric_names:
            results[model_names[i]][k] = results[model_names[i]][k][0]
    return results

def process(results):
    for i in range(len(model_names)):
        for k in results[model_names[i]]:
            if(k!='bpp'):
                yp = results[model_names[i]][k]
                results[model_names[i]][k] = yp
        results[model_names[i]]['bpp'] = results[model_names[i]]['bpp']
    return results

def show(results):
    for i in range(len(metric_names)):
        if(metric_names[i] =='bpp'):
            continue
        for j in range(len(model_names)):
            plt.figure(j)
            bpp = results[model_names[j]]['bpp']
            for k in range(len(results[model_names[j]]['bpp'])):
                # tpr = np.zeros(results[model_names[j]][metric_names[i]][k].val['tpr']['0'].shape)
                # for kk in (results[model_names[j]][metric_names[i]][k].val['tpr']):
                    # print(tpr.shape)
                    # print(results[model_names[j]][metric_names[i]][k].val['tpr'][kk].shape)
                    # tpr = tpr + results[model_names[j]][metric_names[i]][k].val['tpr'][kk]
                # exit()
                # tpr = tpr/10
                # fpr = np.zeros(results[model_names[j]][metric_names[i]][k].val['fpr']['0'].shape)
                # for kk in (results[model_names[j]][metric_names[i]][k].val['fpr']):
                    # fpr = fpr + results[model_names[j]][metric_names[i]][k].val['fpr'][kk]
                # fpr = fpr/10                
                xp = results[model_names[j]][metric_names[i]][k].val['fpr']['0']
                yp = results[model_names[j]][metric_names[i]][k].val['tpr']['0']
                # x = np.linspace(0,0.1,100)
                # y = np.interp(x,xp,yp)
                x = xp
                y = yp
                color = 'black' if model_names[j] not in default_colors else colors[k]
                linestyle = '-' if model_names[j] not in default_linestyles else default_linestyles[model_names[j]]
                label = 'bpp={}'.format(bpp[k].val)
                plt.plot(x,y,color=color,linestyle=linestyle,label=label,linewidth=2)
            plt.xlabel('False Positive Rate',fontsize=12)
            if(metric_names[i]=='psnr'):
                plt.ylabel('PSNR',fontsize=12)
            else:
                plt.ylabel('True Positive Rate',fontsize=12)
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()
    return
    
# def process(results):
    # output = {}
    # all_results = {k:{s:[] for s in supmodel_names} for k in metric_names}   
    # for i in range(len(model_names)):
        # for k in metric_names:
            # if(model_names[i][:15]=='Joint_dist_base'):
                # all_results[k][model_names[i][:17]].append(results[model_names[i]][k])
            # elif(model_names[i][:10]=='Joint_dist'):
                # all_results[k][model_names[i][:12]].append(results[model_names[i]][k])
            # elif(model_names[i][:5]=='Joint'):
                # all_results[k][model_names[i]].append(results[model_names[i]][k])
    # for k in metric_names:
        # output[k] = {}
        # for m in all_results[k]:
            # if(m!='Joint'):
                # all_results[k][m] = np.vstack(all_results[k][m])
                # output[k][m+'_min'] = np.amin(all_results[k][m],axis=0)
                # output[k][m+'_mean'] = np.mean(all_results[k][m],axis=0)
                # output[k][m+'_max'] = np.amax(all_results[k][m],axis=0)
            # else:
                # output[k][m+'_mean'] = np.array(all_results[k][m]).reshape(-1)
    # output['bpp'] = np.linspace(0.125,2,16)
    # return output

# def show(results):
    # for i in range(len(metric_names)):
        # if(metric_names[i] =='bpp'):
            # continue
        # plt.figure(i)
        # x = results['bpp']
        # for k in supmodel_names:
            # y_mean = results[metric_names[i]][k+'_mean']
            # color = 'black' if k not in default_colors else default_colors[k]
            # linestyle = '-' if k not in default_linestyles else default_linestyles[k]
            # label = k if k not in default_labels else default_labels[k]
            # # if(k!='Joint'):
                # # y_min = results[metric_names[i]][k+'_min']            
                # # y_max = results[metric_names[i]][k+'_max']    
                # # plt.fill_between(x,y_max,y_min,color=color,alpha=0.5,linewidth=1)
            # plt.plot(x,y_mean,color=color,linestyle=linestyle,label=label,linewidth=3)
        # plt.xlabel('BPP',fontsize=12)
        # plt.ylabel('PSNR',fontsize=12)
        # plt.grid()
        # plt.legend(loc='lower right')
        # plt.show()
    # return

# def process(results):
    # for i in range(len(model_names)):
        # xp = results[model_names[i]]['bpp']
        # x = np.linspace(xp.min(),xp.max(),100)
        # for k in results[model_names[i]]:
            # if(k!='bpp'):
                # yp = results[model_names[i]][k]
                # print(model_names[i],k)
                # print(xp)
                # print(yp)
                # results[model_names[i]][k] = np.interp(x,xp,yp)
        # results[model_names[i]]['bpp'] = x
    # return results

# def show(results):
    # for i in range(len(metric_names)):
        # if(metric_names[i] =='bpp'):
            # continue
        # plt.figure(i)
        # for j in range(len(model_names)):
            # x = results[model_names[j]]['bpp']
            # y = results[model_names[j]][metric_names[i]]
            # color = 'black' if model_names[j] not in default_colors else default_colors[model_names[j]]
            # linestyle = '-' if model_names[j] not in default_linestyles else default_linestyles[model_names[j]]
            # label = model_names[j] if model_names[j] not in default_labels else default_labels[model_names[j]]
            # plt.plot(x,y,color=color,linestyle=linestyle,label=label,linewidth=3)
        # plt.xlabel('BPP',fontsize=12)
        # if(metric_names[i]=='psnr'):
            # plt.ylabel('PSNR',fontsize=12)
        # else:
            # plt.ylabel('Accuracy',fontsize=12)
        # plt.grid()
        # plt.legend(loc='lower right')
        # plt.show()
    # return
    
if __name__ == "__main__":
   main()