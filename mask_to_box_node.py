# mask_to_box_node.py
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# from comfy.utils import PIL_to_numpy, numpy_to_PIL

def apply_affine_transform(src_points, dst_points, src_image, dst_image):
    """
    计算仿射变换矩阵并应用变换，将原始图片中的部分区域映射到目标图片的指定区域。
    :param src_points: 原始图片上待变换区域的三个坐标点 [(x0, y0), (x0, y1), (x1, y1)]
    :param dst_points: 目标图片上对应区域的四个坐标点 [(x0, y0), (x0, y1), (x1, y1),(x1,y0)]
    :param src_image: 原始图片
    :param dst_image: 目标图片
    :return: 合成后的目标图片
    """

    # 转换坐标为 NumPy 数组
    src_pts = np.float32(src_points)
    dst_pts1 = np.float32(dst_points)
    dst_pts = dst_pts1[0:3]

    # 计算仿射变换矩阵
    affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)

    # 计算变换区域的边界框
    x, y, w, h = cv2.boundingRect(np.array(dst_pts, dtype=np.int32))

    # 进行仿射变换
    transformed_region = cv2.warpAffine(src_image, affine_matrix, (dst_image.shape[1], dst_image.shape[0]))

    # 创建掩码，仅保留变换后的区域
    mask = np.zeros_like(dst_image, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_pts1), (255, 255, 255))
    transformed_region = cv2.bitwise_and(transformed_region,mask)
    # 将变换区域覆盖到目标图片上
    dst_image = cv2.bitwise_and(dst_image, cv2.bitwise_not(mask))
    dst_image = cv2.bitwise_or(dst_image, transformed_region)

    return dst_image

def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)  # connectivity参数的默认值为8
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1]  # 排除最外层的连通图

def mask_to_box_transform(mask, source_pth, target_pth):
    ret, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    bboxs = mask_find_bboxs(mask_bin)
    b = bboxs[0]
    x0, y0 = b[0], b[1]
    x1 = b[0] + b[2]
    y1 = b[1] + b[3]
    print(f'x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}')
    # 构造仿射变换（简单拉伸 source 到 bbox）
    src_image = cv2.imread(source_pth, cv2.IMREAD_COLOR)
    dst_image = cv2.imread(target_pth, cv2.IMREAD_COLOR)
    # print(src_image.shape)
    # print(dst_image.shape)
    src_pts = [[0, 0], [0, src_image.shape[0] - 1], [src_image.shape[1] - 1, src_image.shape[0] - 1]]
    dst_pts = [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
    result_image = apply_affine_transform(src_pts, dst_pts, src_image, dst_image)
    return result_image

if __name__ == "__main__":
    mask = cv2.imread("D:\\study\\2025Autumn\code\ComfyUI_00065_.png", cv2.IMREAD_GRAYSCALE)
    syn = mask_to_box_transform(mask, "D:\\study\\2025Autumn\\code\\source_test.png", "D:\\study\\2025Autumn\\code\\target_test.jpg")
    # plt.imshow(syn)
    # plt.axis('off')  # 去坐标轴
    # plt.xticks([])  # 去 x 轴刻度
    # plt.yticks([])  # 去 y 轴刻度
    # plt.savefig(os.path.join("tmp.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    cv2.imwrite("tmp.png", syn)
