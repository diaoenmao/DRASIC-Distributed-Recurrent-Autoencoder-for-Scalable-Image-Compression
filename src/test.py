import config

config.init()
import argparse
import datetime
import models
import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from data import fetch_dataset, split_dataset
from metrics import Metric
from utils import save, load, to_device, process_control_name
from logger import Logger

device = config.PARAM['device']

def collate(input):
    for k in input:
        input[k] = torch.stack(input[k],0)
    return input

# if __name__ == '__main__':
    # batch_size = 2
    # train_dataset, test_dataset = fetch_dataset('MOSI')
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=0, collate_fn=input_collate)
    # print(len(train_dataset))    
    # for i, input in enumerate(train_loader):
        # input = collate(input)
        # input = dict_to_device(input,device)
        # print(input)
        # exit()   
        
        
# if __name__ == '__main__':
    # weight = torch.randn(5, 5, 3, 3)
    # # in_coordinates = torch.arange(1,4).view(1,-1)
    # # out_coordinates = torch.arange(1,4).view(-1,1)
    # in_coordinates = torch.LongTensor([0,2,4]).view(1,-1)
    # out_coordinates = torch.LongTensor([0,2,4]).view(-1,1)
    # weight_used = weight[out_coordinates,in_coordinates,]
    # print(weight_used.size())
    # print(weight[:,:,0,0])
    # print(weight_used[:,:,0,0])
    # exit()

# def make_cardinal_coordinates(channel_size, cardinality, sharing_rate):
    # indices_set = []
    # total_cardinals = 1
    # for i in range(len(channel_size)):
        # assert channel_size[i] % total_cardinals == 0
        # assert channel_size[i] % (total_cardinals*cardinality[i]) == 0
        # indices = torch.arange(channel_size[i]//total_cardinals)
        # chunked_indices = indices.chunk(cardinality[i])
        # shared_indices = torch.arange(round(len(chunked_indices[0])*sharing_rate[i]))
        # for j in range(1,len(chunked_indices)):
            # chunked_indices[j][shared_indices] = shared_indices
        # indices_set.append(chunked_indices)
        # total_cardinals = total_cardinals*cardinality[i]
    # cardinal_coordinates = list(itertools.product(*indices_set))
    # return cardinal_coordinates
            
# def make_cardinal_masks(channel_size, cardinal_coordinates):
    # cardinal_masks = [torch.zeros(channel_size,dtype=torch.long, device=device) for _ in range(len(cardinal_coordinates))]
    # for i in range(len(cardinal_coordinates)):
        # #cardinal_masks[i][torch.meshgrid(cardinal_coordinates[i])] = 1
        # cardinal_coordinate = (cardinal_coordinates[i][0].view(-1,1),cardinal_coordinates[i][1].view(1,-1))
        # cardinal_masks[i][cardinal_coordinate] = 1
    # return cardinal_masks
    
# if __name__ == '__main__':
    # channel_size = [128,128]
    # cardinality = [2,2]
    # sharing_rate = [0,0.5]
    # cardinal_coordinates = make_cardinal_coordinates(channel_size,cardinality,sharing_rate)
    # cardinal_masks = make_cardinal_masks(channel_size,cardinal_coordinates)
    # cardinal_mask = sum(cardinal_masks)
    # print(cardinal_coordinates)
    # print(cardinal_mask)
    # plt.figure()
    # plt.imshow(cardinal_mask.cpu().numpy(),cmap=plt.cm.Greys_r,)
    # plt.show()
    
    
# def fn(a):
    # return a + 2
    
# if __name__ == '__main__':
    # test = fn(a=2,b=3)
    # print(test)

# if __name__ == '__main__':
    # input_size = 100
    # output_size = 200
    # cardinality = 10
    # sharing_rate = 1
    # feature_size = (2,input_size,8,8)
    # print('feature_size',feature_size)
    # encoder_info_0 = {'input_size':input_size,'output_size':output_size,'cell':'CartesianCell','mode':'pass','cardinality':[cardinality],'sharing_rate':[sharing_rate],
        # 'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}
    # cell_0 = Cell(encoder_info_0)
    # encoder_info_1 = {'input_size':output_size,'output_size':output_size,'cell':'CartesianCell','mode':'pass','cardinality':[cardinality],'sharing_rate':[sharing_rate],
        # 'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}
    # cell_1 = Cell(encoder_info_1)
    # x = torch.rand(feature_size)
    # x = cell_0(x)
    # x = cell_1(x)
    
# if __name__ == '__main__':
    # input = torch.rand(50,10)
    # weight = torch.rand(20,10,requires_grad=True)
    # logits = torch.rand(10,10,requires_grad=True)
    # mask = torch.nn.functional.gumbel_softmax(logits,tau=1,hard=True)
    # masked_weight = mask.matmul(weight.t()).t()
    # output = torch.nn.functional.linear(input,masked_weight)
    # output.mean().backward()
    # print(weight.grad)
    # print(logits.grad)
    
# if __name__ == '__main__':
    # weight = torch.arange(12,dtype=torch.float32).view(3,4)
    # logits_out = torch.rand(3,3,requires_grad=True)
    # permutation_out = torch.nn.functional.gumbel_softmax(logits_out,tau=1,hard=True,dim=1)
    # logits_in = torch.rand(4,4,requires_grad=True)
    # permutation_in = torch.nn.functional.gumbel_softmax(logits_in,tau=1,hard=True,dim=1)
    # result1 = torch.einsum('ao,oi,bi->ab',permutation_out,weight,permutation_in)
    # tmp_result = permutation_out.matmul(weight)
    # result2 = permutation_in.matmul(tmp_result.t()).t()
    # print(weight)
    # print(permutation_out)
    # print(permutation_in)
    # print(result1)
    # print(result2)

# if __name__ == '__main__':
    # input = torch.rand(2,3,1,1)
    # conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
    # oconv = oConv2d(in_channels=3, out_channels=4, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
    # oconv.weight.data.copy_(conv.weight)
    # index = torch.randperm(3).view(-1,1)
    # print(index)
    # permutation = torch.zeros(3,3).scatter_(1, index, 1.0)
    # print(permutation)
    # print(conv.weight[:,:,0,0])
    # x = input
    # y_1 = conv(x)
    # x = input
    # x = torch.einsum('ic,nchw->nihw',permutation,x)
    # y_2 = oconv(x,permutation)
    # print(y_1[:,:,0,0])
    # print(y_2[:,:,0,0])
    # r = torch.eq(y_1,y_2)
    # print(r[:,:,0,0])
    
# if __name__ == '__main__':
    # input = torch.rand(2,3,1,1)
    # conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
    # oconv = oConv2d(in_channels=3, out_channels=4, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
    # oconv.weight.data.copy_(conv.weight)
    # logits = torch.Tensor(1,3)
    # nn.init.kaiming_uniform_(logits,a=math.sqrt(5))
    # permutation = gumbel_softrank(logits,tau=1,hard=True,sample=True,dim=1)
    # print(permutation)
    # print(conv.weight[:,:,0,0])
    # x = input
    # y_1 = conv(x)
    # x = input
    # x = torch.einsum('ic,nchw->nihw',permutation,x)
    # y_2 = oconv(x,permutation)
    # print(y_1[:,:,0,0])
    # print(y_2[:,:,0,0])
    # r = torch.eq(y_1,y_2)
    # print(r[:,:,0,0])
    
# if __name__ == '__main__':
    # batch_size = 2
    # dataset = fetch_dataset('Flickr30k')
    # train_loader = torch.utils.data.DataLoader(dataset=dataset['train'], batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=0, collate_fn=input_collate)
    # print(len(dataset['train']))
    # print(len(dataset['test']))   
    # for i, input in enumerate(train_loader):
        # input = collate(input)
        # input = dict_to_device(input,device)
        # print(input['img'].size())
        # exit()

# if __name__ == '__main__':
    # m = nn.Sequential(
			# nn.ZeroPad2d((0, 0, 0, 0)),
			# nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
		# )
    # #m = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
    # #m = nn.ConvTranspose2d(3, 64, kernel_size=5, stride=2, padding=2)
    # #m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # x = torch.randn(1, 3, 128, 128)
    # x = m(x)
    # print(x.size())
    # x = m(x)
    # print(x.size())
    #m = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
    #x = m(x)
    # print(x.size())
    # x = m(x)
    # print(x.size())

# if __name__ == '__main__':
    # m_1 = Cell({'cell':'PixelShuffleCell','mode':'down','scale_factor':2,'groups':1})
    # m_3 = Cell({'cell':'PixelShuffleCell','mode':'down','scale_factor':2,'groups':3})
    # input = torch.randn(1,32*3,128,128)
    # output_1 = m_1(input)
    # output_3 = m_3(input)
    # print(torch.eq(output_1,output_3).all())
    
    
# if __name__ == '__main__':
    # batch_size = 2
    # dataset = fetch_dataset('Kodak_patch')
    # train_loader = torch.utils.data.DataLoader(dataset=dataset['train'], batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=0, collate_fn=input_collate)
    # print(len(dataset['train']))
    # print(len(dataset['test']))
    # m = Cell({'cell':'PixelShuffleCell','mode':'down','scale_factor':2})
    # for i, input in enumerate(train_loader):
        # input = collate(input)
        # input = dict_to_device(input,device)
        # output = input['img']
        # save_img(input['img'],'./output/img/image.png')
        # for j in range(3):
            # output = m(output)
            # save_img(output,'./output/img/image_shuffled_{}.png'.format(j))
        # exit()
        
        
# if __name__ == '__main__':
#     from torch.utils.tensorboard import SummaryWriter
#     import numpy as np
#
#     writer = SummaryWriter()
#
#     for n_iter in range(100):
#         writer.add_scalar('Loss/train', np.random.random(), n_iter)
#         writer.add_scalar('Loss/test', np.random.random(), n_iter)
#         writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#         writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

# if __name__ == '__main__':
#     from torch.utils.tensorboard import SummaryWriter
#
#     writer = SummaryWriter('runs/test')
#     for n_iter in range(100):
#         t = datetime.timedelta(seconds = n_iter*10)
#         writer.add_text('time', str(t), n_iter)
#     for n_iter in range(100):
#         t = datetime.timedelta(seconds = n_iter*10)
#         writer.add_text('time', str(t), n_iter)
#     writer.close()
