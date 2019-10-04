import config
config.init()
import argparse
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
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
    mine = process(load('./output/result/0_CIFAR10_codec_16_8.pkl'))
    jpg = process(load('./output/result/0_Kodak_Magick_jpg.pkl')['result'])
    jp2 = process(load('./output/result/0_Kodak_Magick_jp2.pkl')['result'])
    bpg = process(load('./output/result/0_Kodak_Magick_bpg.pkl')['result'])
    toderici = np.genfromtxt('./output/result/toderici.csv', delimiter=',').T
    result = [mine,toderici,bpg,jp2,jpg]
    labels = ['Ours','Toderici (Baseline)','BPG','JPG2000','JPG']
    fig_format = 'eps'
    x_name = 'BPP'
    y_name = 'PSNR'
    colors = ['red','darkorange','blue','royalblue','lightskyblue']
    fig = plt.figure()
    fontsize = 14
    for i in range(len(result)):        
        mask = (result[i][0]>0.1) & (result[i][0]<2.3)
        plt.plot(result[i][0][mask],result[i][1][mask],color=colors[i],linestyle='-',label=labels[i],linewidth=3)
    plt.xlabel(x_name,fontsize=fontsize)
    plt.ylabel(y_name,fontsize=fontsize)
    plt.grid()
    plt.legend()
    plt.show()
    makedir_exist_ok('./output/fig/')
    fig.savefig('./output/fig/result.{}'.format(fig_format),bbox_inches='tight',pad_inches=0)       
    plt.close()

def process(input):
    bpp = []
    psnr = []
    for i in range(len(input)):
        bpp += [input[i].panel['bpp'].avg]
        psnr += [input[i].panel['psnr'].avg]
    result = np.array([bpp,psnr])
    return result
        
if __name__ == '__main__':
    main()