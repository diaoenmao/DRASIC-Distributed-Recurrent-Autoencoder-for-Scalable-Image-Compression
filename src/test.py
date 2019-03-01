import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import models
import os
from utils import *
from metrics import *
from torchvision.utils import make_grid
from data import *
from PIL import Image


# imagefilename = './data/sample/lena.tif'
# img=cv2.imread(imagefilename, 1)
# ret,img_encode = cv2.imencode('.jpg', img)
# binary_img_encode = img_encode.tostring()
# with open('./data/sample/lena.jpg', 'wb') as f:
    # f.write(binary_img_encode)
# with open('./sample/lena.jpg', 'rb') as f:
    # binary_img_encode_r = f.read()
# img_encode_r = np.frombuffer(binary_img_encode_r, dtype=np.uint8);    
# img_decode = cv2.imdecode(img_encode_r, cv2.IMREAD_COLOR)
# print(img_encode_r)
# assert np.array_equal(img_encode.reshape(-1),img_encode_r)

# path = './data/sample/input_video.mp4'
# cap = cv2.VideoCapture(path)
# print(cap.isOpened())   # True = read video successfully. False - fail to read video.
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("./sample/output_video.avi", fourcc, 20.0, (640, 360))
# print(out.isOpened())  # True = write out video successfully. False - fail to write out video.
# while(cap.isOpened()):
    # ret,frame = cap.read()
    # if ret==True:
        # out.write(frame)
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
    # else:
        # break
# cap.release()
# out.release()
# cv2.destroyAllWindows()




# N = 10
# C = 3
# H = 1023
# W = 800
# x = torch.randn(N,C,H,W)
# print(x.shape)
# size = (128,190)
# patches_fold_H = x.unfold(2, size[0], size[0])
# print(patches_fold_H.shape)
# if(H % size[0] != 0):
    # patches_fold_H = torch.cat((patches_fold_H,x[:,:,-size[0]:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
# print(patches_fold_H.shape)
# patches_fold_HW = patches_fold_H.unfold(3, size[1], size[1])
# if(W % size[1] != 0):
    # patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-size[1]:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
# print(patches_fold_HW.shape)
# patches = patches_fold_HW.permute(0,2,3,1,4,5).reshape(-1,C,size[0],size[1])
# print(patches.shape)


# size = (128,128)
# # size = (500,800)
# imagefilename = './data/sample/mountain.png'
# img_np = cv2.cvtColor(cv2.imread(imagefilename, 1), cv2.COLOR_BGR2RGB)
# img = torch.from_numpy(img_np).float().permute(2,0,1).unsqueeze(0)
# img = torch.cat((img,img),0)
# nrow = int(np.ceil(float(img.size(3))/size[1]))
# print(img.size())
# patches = extract_patches_2D(img,size)
#save_img(patches/255,'./test.jpg',nrow)
# batch1 = make_grid(patches[:,0,],nrow=nrow)
# batch2 = make_grid(patches[:,1,],nrow=nrow)
# output_img = torch.stack((batch1,batch2),0)
# save_seq_img(patches,'./output/test.jpg',nrow)
# plt.imshow(batch1.permute(1,2,0).numpy())
# plt.show()

# imagefilename = 'data/sample/mountain.png'
# BGR_img = cv2.imread(imagefilename, 1)
# YCC_img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2YCR_CB)
# my_YCC_img = RGB_to_YCbCr(torch.from_numpy(np.float32(BGR_img)).permute(2,0,1).unsqueeze(0)).squeeze().permute(1,2,0).numpy()
# assert np.array_equal(YCC_img,my_YCC_img)
# my_RGB_img = YCbCr_to_RGB(RGB_to_YCbCr(torch.from_numpy(np.float32(BGR_img)).permute(2,0,1).unsqueeze(0))).squeeze().permute(1,2,0).numpy()
# my_BGR_img = cv2.cvtColor(BGR_img, cv2.COLOR_RGB2BGR)
# assert np.array_equal(BGR_img,my_BGR_img)


# path = './data/ImageNet/train'
# unzip(path,'tar')

# path = './data/sample/lena.jpg'
# img = Image.open(path)
# transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.Lambda(lambda x: RGB_to_YCbCr(x))])
# transformed_img = transform(img)
# copy_image = np.array(transformed_img.copy()) # Make a copy
# copy_image[:,:,0] = 0
# copy_image[:,:,1] = 0
# plt.imshow(copy_image)
# plt.show()

# path = './data/sample/lena.jpg'
# img = Image.open(path)
# transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.Lambda(lambda x: RGB_to_YCbCr(x)),
                # transforms.ToTensor()])
# transformed_img = transform(img)
# print(transformed_img.size())

# def main():    
    # randomGen = np.random.RandomState(0)
    # data_name = 'VOCDetection'
    # data_size = 0
    # train_dataset,test_dataset = fetch_dataset(data_name)
    # whole_dataset = torch.utils.data.ConcatDataset([train_dataset,test_dataset])
    # data_loader = torch.utils.data.DataLoader(dataset=whole_dataset, batch_size=1, pin_memory=True, num_workers=num_workers*world_size)
    # #train_loader,test_loader = split_dataset(train_dataset,test_dataset,data_size=0,batch_size=1,radomGen=randomGen)
    # for i, (input,target) in enumerate(data_loader):
        # print(i)
        # print(target)

    
# if __name__=='__main__':
    # main()

    
# size = (128,128)
# step = [1.0,1.0]
# imagefilename = './data/sample/mountain.png'
# img_np = cv2.cvtColor(cv2.imread(imagefilename, 1), cv2.COLOR_BGR2RGB)
# img = torch.from_numpy(img_np).float().permute(2,0,1).unsqueeze(0)
# img = torch.cat((img,img),0)
# patches = extract_patches_2d(img,size,step=step,batch_first=True)
# step = int(size[0]*step) if(isinstance(step, float)) else step
# nrow = 1 + (img.size(-1) - size[1])//step[1] + 1 if(img.size(-1) % size[1] != 0) else 1 + (img.size(-1) - size[1])//step[1]
# print(img.size())
# print(patches.size())
# save_img(patches/255,'./test.jpg',nrow,batch_first=True)
# reconstruct_img = reconstruct_from_patches_2d(patches,(img.size(2),img.size(3)),step=step,batch_first=True)
# print(reconstruct_img.size())
# save_img(reconstruct_img/255,'./test_recon.jpg')

# input = np.random.randint(2, size=(100,32,2,2), dtype=np.bool)
# print(input.nbytes)
# input = np.packbits(input)
# print(input.nbytes)
# input = input.tobytes()
# entropy_codec = models.classic.Entropy()
# entropy_codec.encode(input)


# with Image(filename='data/sample/mountain.png') as img:
    # print('width =', img.width)
    # print('height =', img.height)
    # img.format = 'jp2'
    # print(img.format)
    # img.compression_quality = 95
    # print(img.compression_quality)
    # img.save(filename='test.jp2')
    
# f = open('trainprep.sh', 'w')    
# filenames = filenames_in('.','tar')
# for fn in filenames:
    # f.write('tar -xvf {fn}.tar -C ./{fn}\n'.format(fn=fn))
# f.close()

# if __name__ == '__main__':
    # batch_size = 1000
    # train_dataset, test_dataset = fetch_dataset('CIFAR100')
    # train_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, pin_memory=True, num_workers=0)
    # print(len(test_dataset))    
    # for i, input in enumerate(train_loader):
        # img = input['img']
        # img_show = []
        # for i in range(100):
            # img_show.append(img[input['label'] == i,][0])
        # img_show = torch.stack(img_show,0)
        # save_img(img_show,'./output/img/test.png',nrow = 1)
        # exit()
    
# if __name__ == '__main__':
    # batch_size = 2
    # train_dataset, test_dataset = fetch_dataset('VOCSegmentation')
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, num_workers=0,collate_fn=input_collate)
    # print(len(train_dataset))    
    # for i, input in enumerate(train_loader):
        # print(input)
        # exit()

# def collate(input):
    # for k in input:
        # input[k] = torch.stack(input[k],0)
    # return input
    
# if __name__ == '__main__':
    # batch_size = 2
    # train_dataset, test_dataset = fetch_dataset('MNIST')
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, num_workers=0,collate_fn=input_collate)
    # print(len(train_dataset))    
    # for i, input in enumerate(train_loader):
        # input = collate(input)
        # mssim = MSSIM(input['img'],input['img'])
        # print(mssim)
        # exit()        