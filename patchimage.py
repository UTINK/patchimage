# from patchify import patchify
from skimage import io
from skimage.color import rgb2gray
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


# patches_1 = np.lib.stride_tricks.sliding_window_view(image, (64, 64, 64))
# patches_2 = np.lib.stride_tricks.as_strided(image, (64, 64, 64), (34, 24, 24))
# print(patches_1.shape)
# print(patches_2.shape)


# 定义辅助函数   体素转化为体素块
def volume2patches(voxel, patch_size, stride):
    """
    
    :param voxel: 需要切分为图像块的图像, 如(321, 481, 523) 或 (321, 481, 523, ch)
    :param patch_size: 图像块的尺寸，如:(10,10,10)
    :param stride: 切分图像块时移动过得步长，如:(5, 4, 2)
    :return: np.array类型的patches
    """

    if len(voxel.shape) == 3:
        # 灰度体素
        voxdeth, voxhigh, voxwidth = voxel.shape
    if len(voxel.shape) == 4:
        # RGB图像
        voxdeth, voxhigh, voxwidth, voxch = voxel.shape

    # 构建图像块的索引
    range_z = np.arange(0, voxdeth - patch_size[0], stride[0])
    range_y = np.arange(0, voxhigh - patch_size[1], stride[1])
    range_x = np.arange(0, voxwidth - patch_size[2], stride[2])

    if range_z[-1] != voxdeth - patch_size[0]:
        range_z = np.append(range_z, voxdeth - patch_size[0])
    if range_y[-1] != voxhigh - patch_size[1]:
        range_y = np.append(range_y, voxhigh - patch_size[1])
    if range_x[-1] != voxwidth - patch_size[2]:
        range_x = np.append(range_x, voxwidth - patch_size[2])

    sz = len(range_z) * len(range_y) * len(range_x)  ## 图像块的数量

    if len(voxel.shape) == 3:
        # # 初始化灰度图像
        res = np.zeros((sz, patch_size[0], patch_size[1], patch_size[2]))
    if len(voxel.shape) == 4:
        # # 初始化RGB图像
        res = np.zeros((sz, patch_size[0], patch_size[1], patch_size[2], voxch))

    index = 0
    for z in range_z:
        for y in range_y:
            for x in range_x:
                patch = voxel[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]]
                res[index] = patch
                index = index + 1

    dtype = voxel.dtype
    res = np.array(res, dtype=dtype)
    return res


# 定义辅助函数    图像转化为图像块
def image2patches(image, patch_size, stride):
    """
    
    :param image: 需要切分为图像块的图像, 如 (321, 481) 或 (321, 481, ch)
    :param patch_size: 图像块的尺寸，如:(10,5)
    :param stride: 切分图像块时移动过得步长，如:5
    :return: np.array类型的patches
    """
    """
    image:需要切分为图像块的图像, 如 (321, 481) 或 (321, 481, 3)
    patch_size:图像块的尺寸，如:(10,10)
    stride:切分图像块时移动过得步长，如:5
    """
    if len(image.shape) == 2:
        # 灰度图像
        imhigh, imwidth = image.shape
    if len(image.shape) == 3:
        # RGB图像
        imhigh, imwidth, imch = image.shape
    # 构建图像块的索引
    range_y = np.arange(0, imhigh - patch_size[0], stride)
    range_x = np.arange(0, imwidth - patch_size[1], stride)

    if range_y[-1] != imhigh - patch_size[0]:
        range_y = np.append(range_y, imhigh - patch_size[0])
    if range_x[-1] != imwidth - patch_size[1]:
        range_x = np.append(range_x, imwidth - patch_size[1])
    sz = len(range_y) * len(range_x)  ## 图像块的数量

    if len(image.shape) == 2:
        ## 初始化灰度图像
        res = np.zeros((sz, patch_size[0], patch_size[1]))
    if len(image.shape) == 3:
        ## 初始化RGB图像
        res = np.zeros((sz, patch_size[0], patch_size[1], imch))

    index = 0
    for y in range_y:
        for x in range_x:
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            res[index] = patch
            index = index + 1

    dtype = image.dtype
    res = np.array(res, dtype=dtype)

    return res


# 定义函数  体素块还原成体素
def patches2volume(coldata, imsize, stride):
    """
    
    :param coldata: 使用voxel2cols得到的数据  如 (z*y*x, 10,10,10) 或者 (z*y*x, 10,10,10, ch)
    :param imsize: 原始图像的深,高,宽，注意位置zyx 如 (321, 481, 523)
    :param stride: 图像切分时的步长，如(10, 4, 2)
    :return: 原始大小三维图像
    """
    
    patch_size = coldata.shape[1:4]
    if len(coldata.shape) == 4:
        # 初始化灰度图像
        res = np.zeros((imsize[0], imsize[1], imsize[2]))
        w = np.zeros((imsize[0], imsize[1], imsize[2]))
    if len(coldata.shape) == 5:
        # 初始化RGB图像
        res = np.zeros((imsize[0], imsize[1], imsize[2], 3))
        w = np.zeros((imsize[0], imsize[1], imsize[2], 3))

    range_z = np.arange(0, imsize[0] - patch_size[0], stride[0])
    range_y = np.arange(0, imsize[1] - patch_size[1], stride[1])
    range_x = np.arange(0, imsize[2] - patch_size[2], stride[2])

    if range_z[-1] != imsize[0] - patch_size[0]:
        range_z = np.append(range_z, imsize[0] - patch_size[0])
    if range_y[-1] != imsize[1] - patch_size[1]:
        range_y = np.append(range_y, imsize[1] - patch_size[1])
    if range_x[-1] != imsize[2] - patch_size[2]:
        range_x = np.append(range_x, imsize[2] - patch_size[2])

    index = 0
    for z in range_z:
        for y in range_y:
            for x in range_x:
                res[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]] = \
                    res[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]] + coldata[index]
                w[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]] = \
                    w[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]] + 1
                index = index + 1

    return np.array(res / w, dtype=coldata.dtype)


# 定义函数  图像转化为图像块的逆变换
def patches2image(coldata, imsize, stride):
    """
    
    :param coldata: 使用image2cols得到的数据 (z*y*x, 10,10) 或者 (z*y*x, 10,10, ch)
    :param imsize: 原始图像的宽和高，如(321, 481)
    :param stride: 图像切分时的步长，如10
    :return: 原始大小图像
    """
    patch_size = coldata.shape[1:3]
    if len(coldata.shape) == 3:
        # 初始化灰度图像
        res = np.zeros((imsize[0], imsize[1]))
        w = np.zeros((imsize[0], imsize[1]))
    if len(coldata.shape) == 4:
        # 初始化RGB图像
        res = np.zeros((imsize[0], imsize[1], 3))
        w = np.zeros((imsize[0], imsize[1], 3))
    range_y = np.arange(0, imsize[0] - patch_size[0], stride)
    range_x = np.arange(0, imsize[1] - patch_size[1], stride)

    if range_y[-1] != imsize[0] - patch_size[0]:
        range_y = np.append(range_y, imsize[0] - patch_size[0])
    if range_x[-1] != imsize[1] - patch_size[1]:
        range_x = np.append(range_x, imsize[1] - patch_size[1])
    index = 0
    for y in range_y:
        for x in range_x:
            res[y:y + patch_size[0], x:x + patch_size[1]] = \
                res[y:y + patch_size[0], x:x + patch_size[1]] + coldata[index]
            w[y:y + patch_size[0], x:x + patch_size[1]] = \
                w[y:y + patch_size[0], x:x + patch_size[1]] + 1
            index = index + 1

    return np.array(res / w, dtype=coldata.dtype)


#
# #
# def main():
#     voxel_path = os.path.join(os.getcwd(), 'dataset' + os.sep, 'GTImage.tif')
#     gt_path = os.path.join(os.getcwd(), 'dataset' + os.sep, 'GTLabel.tif')
#     voxl = tiff.imread(voxel_path)
#     mask = tiff.imread(gt_path)
#     print(voxl.shape, mask.shape)
#     # print("voxl.shape: ", voxl.shape, ", voxl.type: ", type(voxl))
#     # print("mask.shape: ", mask.shape, ", mask.type: ", type(mask))
# 
#     # model_name = 'UNet3D'  # 'UNet3DS'
#     # prediction_dir = os.path.join(os.getcwd(), 'dataset', 'test', 'predict_f0', model_name + '_results' + os.sep)
#     #
#     # img_name_list = glob.glob(prediction_dir + os.sep + '*')
#     # print("...... %d images......" % (len(img_name_list)))
#     # img_name_list.sort(key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))
#     # print(img_name_list)
#     #
#     # colvoxls = []
#     # for i in img_name_list:
#     #     v = tiff.imread(i)
#     #     colvoxls.append(v)
# 
#     patch_size = (64, 64, 64)
#     voxelsize = voxl.shape
#     stride_size = (34, 32, 32)
# 
#     # 分块
#     patches = volume2patches(voxl, patch_size, stride_size)
#     patches_mask = volume2patches(mask, patch_size, stride_size)
#     print(patches.shape)
#     print(patches_mask.shape)
#     for i in range(len(patches)):
#         tiff.imwrite(r'F:\unet3d\dataset\train\Voxels\%d.tif' % i, patches[i])
#         tiff.imwrite(r'F:\unet3d\dataset\train\GT\%d.tif' % i, patches_mask[i])
# 
#     # # 复原
#     # colvoxls = np.array(colvoxls)
#     # print(colvoxls.shape)
#     # revoxl = patches2volume(coldata=colvoxls, imsize=voxelsize, stride=stride_size)
#     # print("revoxl.shape: ", revoxl.shape, ", revoxl.type: ", type(revoxl), ", revoxl.dtype: ", revoxl.dtype)
# 
#     # plt.figure()
#     # plt.imshow(patches[1,0])
#     # plt.axis("off")
#     # plt.show()
# 
#     # io.imsave(r'C:\Users\UT\Desktop\1-1.tif', revoxl)
# 
# 
# #
# if __name__ == "__main__":
#     main()
