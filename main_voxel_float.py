import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

"""
3D Voxel Grid
		 z
		/|
		 |
		 |
		 |	  \
		 *------ x
		/
	  / 
	|/  
	y
"""

def __consistent_args(tensor, condition, indices, dim):
	d = [1,] * len(tensor.size())
	d[dim] = tensor.size(dim)
	mask = condition(tensor).float() * indices.view(*d).expand_as(tensor)
	return torch.argmax(mask, dim=dim)

def consistent_find_leftmost(tensor, condition, dim):
	indices = torch.arange(tensor.size(dim), 0, -1, dtype=torch.float, device=tensor.device)
	return __consistent_args(tensor, condition, indices, dim)

def consistent_find_rightmost(tensor, condition, dim):
	indices = torch.arange(0, tensor.size(dim), 1, dtype=torch.float, device=tensor.device)
	return __consistent_args(tensor, condition, indices, dim)

def visualize_voxel_occupy(voxel_occupy):
	# voxel_occupy: np.array np.bool
	colors = np.empty(voxel_occupy.shape, dtype=object)
	colors[voxel_occupy] = 'red'
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.voxels(voxel_occupy, facecolors=colors, edgecolor='k')
	plt.show()
	return

def get_torch_normalized_meshgrid(r, c):
	x = np.linspace(-1.0, 1.0, c)
	y = np.linspace(-1.0, 1.0, r)
	grid = torch.from_numpy(np.array(np.meshgrid(x, y))).float()
	return grid.permute(1, 2, 0)

def satellite_elevation_to_voxel_occupy(satellite_elevation,
										grid_shape=[256, 256, 256]):
	"""
	Input:
		satellite_elevation:	torch tensor (batch_size, nrow, ncol)
		grid_shape:				python tuple [nz, ny, nx]
	"""

	satellite_elevation = satellite_elevation.float()
	n, nrow, ncol = satellite_elevation.size()
	nz, ny, nx = grid_shape

	if len(satellite_elevation.shape) == 3:
		satellite_elevation = satellite_elevation.unsqueeze(dim=1) # (batch_size, 1, nrow, ncol)

	grid = get_torch_normalized_meshgrid(ny, nx) # shape (ny, nx, 2)
	grid = grid.unsqueeze(dim=0).expand(n, ny, nx, 2) # shape (n, ny, nx, 2)

	elevation = F.grid_sample(satellite_elevation, grid, padding_mode='border', align_corners=True) # shape (n, 1, ny, nx)
	voxel_dz = torch.arange(0, nz, 1).float().view(1, nz, 1, 1).expand(n, nz, ny, nx)
	voxel_occupy = torch.gt(elevation.expand(n, nz, ny, nx), voxel_dz)

	return voxel_occupy

def voxel_occupy_to_panorama(voxel_occupy,
							 center_loc,
							 direction,
							 panorama_size=[512, 256],
							 min_max_lat=[-np.pi/2, np.pi/2],
							 ray_num=16,
							 ray_time=2,
							 th=0.5,
							 sky_height=1.0
							 ):
	"""
	Input:
		voxel_occupy:		(batch_size, nz, ny, nx)
		center_loc:			(batch_size * n_sample, 3), last dim [-1 ~ 1, -1 ~ 1, -1 ~ 1]
		direction:			(batch_size * n_sample) # rad
		panorama_size:		[width, height]
		min_max_lat:		[min_lat, max_lat]

	Output:
		panorama:			(batch_size, n_sample, height, width)

	"""

	assert(len(voxel_occupy.size()) == 4)
	assert(len(center_loc.size()) == 2)
	assert(len(direction.size()) == 1)

	n, nz, ny, nx = voxel_occupy.size()
	ns = center_loc.size(0)
	cx, cy, cz = center_loc[:, 0], center_loc[:, 1], center_loc[:, 2]
	ncol, nrow = panorama_size
	min_lat, max_lat = min_max_lat

	# X
	delta_angle = direction.float().view(ns, 1).expand(ns, ncol)
	x = (torch.arange(0, ncol, 1, dtype=torch.float32) + 0.5) / ncol
	y = (torch.arange(0, nrow, 1, dtype=torch.float32) + 0.5) / nrow
	lon = delta_angle + (x * 2.0 * np.pi - np.pi).view(1, ncol).expand(ns, ncol)
	lat = (1.0 - y) * (max_lat - min_lat) + min_lat

	sin_lon = torch.sin(lon).view(ns, 1, 1, ncol).expand(ns, 1, nrow, ncol)
	cos_lon = torch.cos(lon).view(ns, 1, 1, ncol).expand(ns, 1, nrow, ncol)
	sin_lat = torch.sin(lat).view(1, 1, nrow, 1).expand(ns, 1, nrow, ncol)
	cos_lat = torch.cos(lat).view(1, 1, nrow, 1).expand(ns, 1, nrow, ncol)

	# Compute the unit vector of each pixel
	vx = cos_lat.mul(sin_lon) # shape (ns, 1, nrow, ncol)
	vy = cos_lat.mul(cos_lon) * -1.0 # shape (ns, 1, nrow, ncol)
	vz = sin_lat # shape (ns, 1, nrow, ncol)

	sample_beg = torch.zeros(ns, ray_num+1, nrow, ncol).float()
	sample_gap = 1.0
	sample_inc = torch.arange(0, ray_num+1, 1, dtype=torch.float32).view(1, ray_num+1, 1, 1).expand(ns, ray_num+1, nrow, ncol)

	for _ in range(ray_time):
		sample_gap /= ray_num
		sample = sample_beg + sample_inc * sample_gap

		idx_x = torch.clamp(sample * vx.expand(ns, ray_num+1, nrow, ncol) + cx.view(ns, 1, 1, 1).expand(ns, ray_num+1, nrow, ncol), -1, 1)
		idx_y = torch.clamp(sample * vy.expand(ns, ray_num+1, nrow, ncol) + cy.view(ns, 1, 1, 1).expand(ns, ray_num+1, nrow, ncol), -1, 1)
		idx_z = torch.clamp(sample * vz.expand(ns, ray_num+1, nrow, ncol) + cz.view(ns, 1, 1, 1).expand(ns, ray_num+1, nrow, ncol), -1, 1)
		idx = torch.stack([idx_x, idx_y, idx_z], dim=-1)

		ray_occupy = F.grid_sample(
			voxel_occupy.float().view(n, 1, nz, ny, nx).expand(ns, 1, nz, ny, nx),
			idx, padding_mode='border', align_corners=True
		) # shape (ns, 1, ray_num+1, nrow, ncol)

		inside_idx = consistent_find_leftmost(ray_occupy, lambda x: x>th, dim=2)
		outside_idx = inside_idx - 1 # torch.clamp(inside_idx - 1, 0, ray_num)

		sample_beg = sample_beg + outside_idx.float().expand(ns, ray_num+1, nrow, ncol) * sample_gap

		# for i in range(ns):
		# 	plt.imshow(sample_beg.numpy()[i, 0, 0])
		# 	plt.title('Sample Begin')
		# 	plt.show()

	inside_val = torch.gather(ray_occupy, 2, inside_idx.unsqueeze(dim=2)).squeeze(dim=2)
	outside_val = torch.gather(ray_occupy, 2, outside_idx.unsqueeze(dim=2)).squeeze(dim=2)
	valid = outside_val < inside_val
	sky_x = sky_height / vz * vx + cx.view(ns, 1, 1, 1).expand(ns, 1, nrow, ncol)
	sky_y = sky_height / vz * vy + cy.view(ns, 1, 1, 1).expand(ns, 1, nrow, ncol)

	dist = (0.5 - outside_val) / (inside_val - outside_val + 1e-6)
	dist = sample_beg[:, 0:1] + dist * sample_gap
	dist = ~valid * 1 + torch.clamp(valid * dist, 0, 1)

	idx_x = torch.clamp(dist * vx.expand(ns, 1, nrow, ncol) + cx.view(ns, 1, 1, 1).expand(ns, 1, nrow, ncol), -1, 1)
	idx_y = torch.clamp(dist * vy.expand(ns, 1, nrow, ncol) + cy.view(ns, 1, 1, 1).expand(ns, 1, nrow, ncol), -1, 1)
	idx_z = torch.clamp(dist * vz.expand(ns, 1, nrow, ncol) + cz.view(ns, 1, 1, 1).expand(ns, 1, nrow, ncol), -1, 1)

	idx_x = ~valid * sky_x + valid * idx_x
	idx_y = ~valid * sky_y + valid * idx_y
	idx_z = ~valid * sky_height + valid * idx_z
	idx = torch.stack([idx_x, idx_y, idx_z], dim=-1)

	# Check whether is around 0.5
	if True:
		ray_occupy = F.grid_sample(
			voxel_occupy.float().view(n, 1, nz, ny, nx).expand(ns, 1, nz, ny, nx),
			idx, padding_mode='border', align_corners=True
		).squeeze(dim=1) # shape (ns, 1, nrow, ncol)
		for i in range(ns):
			plt.imshow(((idx.numpy()[i, 0]+1)*128).astype(np.uint8))
			# plt.imshow(ray_occupy.numpy()[i,0])
			plt.show()
		quit()

	# for i in range(n * ns):
	# 	plt.imshow(dist.numpy()[i, 0, 0], vmin=0, vmax=1)
	# 	plt.title(f'{ray_num} {ray_time} {th}')
	# 	plt.show()

	return dist




	idx_n = torch.arange(0, n, 1, dtype=torch.long).view(n, 1, 1, 1).expand(n, num_sample, nrow, ncol)
	sample_idx = idx_n * nz * ny * nx + idx_z * ny * nx + idx_y * nx + idx_x
	sample_idx_voxel = idx_z * ny * nx + idx_y * nx + idx_x

	voxel_distance_flatten = voxel_distance.contiguous().view(n * nz * ny * nx)
	sample_distance = torch.index_select(voxel_distance_flatten, 0, sample_idx.view(n * num_sample * nrow * ncol))

	sample_distance, which_sample = sample_distance.view(n, num_sample, nrow, ncol).min(dim=1)
	sample_valid = sample_distance < 1e6

	panorama_voxel_idx = torch.gather(sample_idx_voxel, 1, which_sample.unsqueeze(dim=1)).squeeze()
	panorama_voxel_idx = sample_valid * panorama_voxel_idx + ~sample_valid * -1

	# voxel_index_flatten = torch.arange(0, nz * ny * nx).view(1, nz, ny, nx).expand(n, nz, ny, nx)
	# sample_index = torch.index_select(voxel_index_flatten, 0, sample_idx.view(n * num_sample * nrow * ncol))
	
	return sample_distance, panorama_voxel_idx


def warp(src_img, panorama_voxel_idx_batch, ref):

	n, nrow, ncol = panorama_voxel_idx_batch.size()

	# for i in range(-5, 6):
	# 	for j in range(-5, 6):
	# 		print(100+i, 300+j)
	# 		print((panorama_voxel_idx_batch == panorama_voxel_idx_batch[1, 100+i, 300+j]).nonzero())
	# 		print()

	# quit()


	src_img_flatten = src_img.view(nrow * ncol, 3)
	panorama_voxel_idx_batch_flatten = panorama_voxel_idx_batch.view(n, nrow * ncol).long()

	voxel_color = torch.zeros(panorama_voxel_idx_batch_flatten.max().int().item() + 2, 3).byte()
	voxel_color[panorama_voxel_idx_batch_flatten[ref] + 1] = src_img_flatten
	voxel_color[0] = torch.zeros(3).byte()

	# voxel_color_r = torch.arange(0, 256, 2).view(128, 1, 1).byte().expand(128, 128, 128).reshape(-1)
	# voxel_color_g = torch.arange(0, 256, 2).view(1, 128, 1).byte().expand(128, 128, 128).reshape(-1)
	# voxel_color_b = torch.arange(0, 256, 2).view(1, 1, 128).byte().expand(128, 128, 128).reshape(-1)
	# voxel_color = torch.stack([voxel_color_r, voxel_color_g, voxel_color_b], dim=-1)
	# voxel_color = torch.cat([torch.Tensor([[0,0,0]]).byte(), voxel_color], dim=0)

	warpped_img_batch = torch.index_select(voxel_color, 0, panorama_voxel_idx_batch_flatten.view(n * nrow * ncol) + 1).view(n, nrow, ncol, 3)
	
	return warpped_img_batch




if __name__ == '__main__':

	ns           =  11 # odd
	xy_scale     =   0.5 # meter
	z_scale      =   1 # meter
	sample_range =  24 # pixel
	size, hsize  = 256, 128
	nz, hnz      = 256, 128
	view_bias    = 6.0 # 3.0m

	print(f'Sample range: {sample_range * xy_scale}m.')
	step = (ns != 1) * np.linspace(-1, 1, ns) * sample_range / 2

	# files = sorted(glob.glob('/home/zoli/xiaohu_new_data/train2_new/*_sate_depth.png'))
	files = sorted(glob.glob('../Sate2StreetPanoVideoOld/data/*_sate_depth.png'))
	for file in files:

		# img = np.array(Image.open(file.replace('_sate_depth.png', '_street_rgb.png')))
		# img_tensor = torch.from_numpy(img)
		# plt.imshow(img)
		# plt.show()

		elevation = np.array(Image.open(file).convert('L')).astype(np.float32)[np.newaxis, ...] * (z_scale / xy_scale)
		assert(elevation.shape[1:] == (size, size))
		view_ele = elevation[0, hsize-1:hsize+1, hsize-1:hsize+1].mean() + view_bias
		elevation = elevation + (hnz - view_ele) # raise view ele to 0

		direction = float(file.replace('_sate_depth.png', '').split(',')[2]) / 360.0 * np.pi * 2.0
		center_loc = (np.stack([[np.sin(direction), -np.cos(direction)]] * ns).T * step).T
		center_loc = np.concatenate([center_loc / hsize, np.stack([[0]] * ns)], axis = -1)
		center_loc = torch.from_numpy(center_loc).float()

		direction = torch.from_numpy(np.array([direction] * ns)).float()
		elevation = torch.from_numpy(elevation).float()
		voxel_occupy = satellite_elevation_to_voxel_occupy(elevation, grid_shape=[nz, size, size])

		# res = []
		# for rn, rt, th in [(16,2,0.5), (128,1,0.5)]:
		# 	tic = time.time()
		# 	res.append(voxel_occupy_to_panorama(voxel_occupy, center_loc, direction, ray_num=rn, ray_time=rt, th=th).numpy()[0,0,0])
		# 	toc = time.time()
		# 	print(toc - tic)
		# plt.imshow(res[1]-res[0])
		# plt.show()
		# quit()

		# dist = voxel_occupy_to_panorama(voxel_occupy, center_loc, direction, ray_num=128, ray_time=1)
		dist = voxel_occupy_to_panorama(voxel_occupy, center_loc, direction, ray_num=32)
		for i in range(ns):
			plt.imshow(dist.numpy()[i])
			plt.show()

		quit()


		for i in range(int(num/bs)):

			voxel_unsigned_distance, center_grid = satellite_elevation_to_voxel_grid(elevation_batch[0+i*bs:bs+i*bs], center_loc=centor_loc_batch[0+i*bs:bs+i*bs], grid_shape=[256]*3, xy_scale=0.25, z_scale=0.5)
			panorama_batch, panorama_voxel_idx_batch = voxel_grid_to_panorama(voxel_unsigned_distance, center_grid, direction_batch[0+i*bs:bs+i*bs])
			warpped_img_batch = warp(img_tensor, panorama_voxel_idx_batch, int(num / 2))

			for j in range(bs):
				if j != 4:
					continue
				img = (panorama_batch[j].numpy())
				img[img>116]=116
				img = img/img.max() * 255

			# 	img = panorama_voxel_idx_batch[j].numpy()
			# 	a = img%256
			# 	a_res = ((img-a)/256).astype(np.int32)
			# 	b = a_res%256
			# 	c = ((a_res-b)/256).astype(np.int32)%256
			# 	img = np.stack([c,b,a], axis=-1)

				plt.imshow(img.astype(np.uint8))
				plt.show()
				# plt.imshow(warpped_img_batch[j].numpy())
				# plt.savefig('%d.png' % (j+i*bs))
				# plt.clf()
				# plt.show()

		# quit()

		plt.imshow(np.array(Image.open(file.replace('_sate_depth.png', '_proj_dis.png')).convert('L')))
		plt.show()

		quit()


