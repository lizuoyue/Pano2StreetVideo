import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, cv2, tqdm

def point_cloud_to_panorama(point_cloud,
							panorama_size=[512, 256],
							min_max_lat=[-np.pi/2, np.pi/2]):
	"""
	Input:
		point_cloud:		(n, 3 + c), each row float coordinates + c-channel information (c can be 0)
		panorama_size:		[width, height]
		min_max_lat:		[min_lat, max_lat]

	# X  Right
	# Y  Down
	# Z  Inside
	# Coordinates in camera system (local)
	# Points in the ray [0, 0, z] (z > 0) have zero lon and lat

	Output:
		panorama:			(panorama_size[1], panorama_size[0], c)

	"""

	c = point_cloud.shape[1] - 3
	if c == 0:
		pc = point_cloud.copy()
		pc_info = None
	else:
		pc, pc_info = point_cloud[:,:3], point_cloud[:,3:]

	ncol, nrow = panorama_size
	min_lat, max_lat = min_max_lat
	delta_lat = (max_lat - min_lat) / nrow
	dist = np.sqrt(np.sum(pc ** 2, axis=-1))
	if pc_info is None:
		c = 1
		pc_info = dist[..., np.newaxis]

	valid = (dist > 0)
	pc = pc[valid]
	pc_info = pc_info[valid]
	dist = dist[valid]

	order = np.argsort(dist)[::-1]
	pc = pc[order]
	pc_info = pc_info[order]
	dist = dist[order]

	x, y, z = pc[:,0], pc[:,1], pc[:,2]

	lon = np.arctan2(x, z)
	lat = -np.arcsin(y / dist)

	u = np.round((lon / np.pi + 1.0) / 2.0 * ncol).astype(np.int32)
	v = np.round((max_lat - lat) / delta_lat).astype(np.int32)
	img_1d_idx = v * ncol + u

	valid = (-1 < u) & (u < ncol) & (-1 < v) & (v < nrow)
	res = np.zeros((nrow * ncol, c)) * np.nan
	res[img_1d_idx[valid]] = pc_info[valid]
	return res.reshape((nrow, ncol, c))

def pano_img_dis_to_point_cloud(img,
								dis,
								min_max_lat=[-np.pi/2, np.pi/2]):
	"""
	Input:
		img:					(height, width, c)
		dis:					(height, width)
		min_max_lat:			[min_lat, max_lat]

	# Local point cloud is generated
	# X  Right
	# Y  Down
	# Z  Inside

	Output:
		point_cloud:			(n, 3 + c), each row float coordinates + c-channel information (c can be 0)

	"""

	img = np.array(img)
	dis = np.array(dis).astype(np.float)

	if False:
		dis = dis * 0 + 50

	assert(img.shape[:2] == dis.shape[:2])
	nrow, ncol, c = img.shape
	min_lat, max_lat = min_max_lat

	x, y = np.meshgrid(np.arange(0, ncol), np.arange(0, nrow))
	lon = x / ncol * 2.0 * np.pi - np.pi
	lat = (1.0 - y / nrow) * (max_lat - min_lat) + min_lat
	
	vx = np.cos(lat) * np.sin(lon)
	vy = -np.sin(lat)
	vz = np.cos(lat) * np.cos(lon)

	v = np.dstack([vx, vy, vz]).reshape((-1, 3))

	color = img.reshape((-1, 3))
	pc = (v.T * dis.reshape((-1))).T
	valid1 = dis.reshape((-1)) > 1e-3
	valid2 = dis.reshape((-1)) < 200

	return np.hstack([pc, color])[valid1 & valid2]


if __name__ == '__main__':

	if False:
		n = 101
		x = np.linspace(-1, 1, n)
		y = np.linspace(-1, 1, n)
		z = np.linspace(-1, 1, n)
		pc = np.stack(np.meshgrid(x, y, z) * 2).reshape((6, -1)).T
		abs_pc = np.abs(pc)
		pc = pc[(abs_pc == 1).any(axis=-1)]
		hehe = ((np.abs(pc) == 1).sum(axis=-1) == 4)
		pc[hehe,3:]=-1

		img = ((point_cloud_to_panorama(pc, panorama_size=[1024,512]) + 1) / 2 * 255.0).astype(np.uint8)
		
		plt.imshow(img)
		plt.show()

	STEP = 10
	# img_files = sorted(glob.glob('/home/zoli/xiaohu_new_data/train2_new/*_street_rgb.png'))
	# img_files = sorted(glob.glob('../Pano2StreetVideoOld/data/*_street_rgb.png'))
	# for img_file in tqdm.tqdm(img_files):
	# 	dis_file = img_file.replace('_street_rgb', '_proj_dis')
	# 	pc = pano_img_dis_to_point_cloud(Image.open(img_file), Image.open(dis_file).convert('L'))
	# 	res = []
	# 	for i in np.linspace(-1.5, 1.5, 21):
	# 		pc[:, 2] -= float(i * STEP)
	# 		fake_img = point_cloud_to_panorama(pc)
	# 		mask = np.isnan(fake_img.sum(axis=-1))
	# 		fake_img[mask] = [0, 0, 0]
	# 		fake_img_filled = cv2.inpaint(fake_img.astype(np.uint8), mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
	# 		to_save = np.hstack([fake_img_filled, fake_img]).astype(np.uint8)
	# 		res.append(Image.fromarray(to_save))
	# 		pc[:, 2] += float(i * STEP)

	# 	res[0].save(os.path.basename(img_file).replace('.png', '.gif'), save_all=True, append_images=res[1:])

	# kernel = np.ones((3, 3),np.uint8)
	kernel = np.array([[0,0,0],[1,1,1],[0,0,0]]).astype(np.uint8)

	for j in range(10):
		depth = np.array(Image.open('/Users/lizuoyue/Desktop/CVPR_2020_Rebuttal/code/val_ori_sync/images/input_%03d_input.png' % j))[..., 2]
		depth = Image.fromarray(depth)
		img = Image.open('/Users/lizuoyue/Desktop/CVPR_2020_Rebuttal/code/val_ori_sync/images/input_%03d_encoded.png' % j)
		pc = pano_img_dis_to_point_cloud(img, depth)
		res = []
		for i in np.linspace(-1.5, 1.5, 21):
			pc[:, 2] -= float(i * STEP)
			fake_img = point_cloud_to_panorama(pc)
			mask = np.isnan(fake_img.sum(axis=-1))

			ground_mask = mask.astype(np.int32).argmin(axis=0)
			ground_mask = (((np.ones((512,256)) * np.arange(256)).T) > ground_mask).astype(np.uint8) * 255
			ground_mask = cv2.erode(ground_mask, kernel, iterations = 1)
			ground_mask = cv2.dilate(ground_mask, kernel, iterations = 5)
			ground_mask = cv2.erode(ground_mask, kernel, iterations = 4)

			mask[ground_mask == 0] = False
			fake_img[mask] = np.nan#[0, 0, 0]
			fake_img_filled = cv2.inpaint(fake_img.astype(np.uint8), mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
			to_save = np.hstack([fake_img_filled, fake_img]).astype(np.uint8)
			res.append(Image.fromarray(to_save))
			pc[:, 2] += float(i * STEP)
		res[0].save('%03d.gif' % j, save_all=True, append_images=res[1:])





