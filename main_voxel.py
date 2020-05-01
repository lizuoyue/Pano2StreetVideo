import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
3D Voxel Grid
	          \
	    *------ x
	   /|
	  / |
	|/  |/
	y   z
"""

  # noqa: F401 unused import

def visualize_voxel_grid(voxel_occupy):
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

def satellite_elevation_to_voxel_grid(satellite_elevation,
									  xy_scale=0.5,
									  z_scale=1.0,
									  grid_shape=[128, 128, 128],
									  center_loc=None,
									  center_bias=None,
									  same_elevation=True):
	"""
	Input:
		satellite_elevation:	torch tensor (batch_size, nrow, ncol)
		xy_scale:				scalar, meter per pixel
		z_scale:				scalar, meter per unit
		center_loc:				torch tensor (batch_size, 2), each row [0 ~ ncol, 0 ~ nrow]
		center_bias:			torch tensor (batch_size) # pixel value
		grid_shape:				python tuple [nz, ny, nx]

	"""

	satellite_elevation /= z_scale

	n, nrow, ncol = satellite_elevation.size()
	nz, ny, nx = grid_shape
	if center_loc is None:
		center_loc = torch.zeros(n, 1, 1, 2).float() # shape (n, 1, 1, 2)
	else:
		center_loc = torch.cat([center_loc[:, 0:1] / nrow, center_loc[:, 1:2] / ncol], dim=1).view(n, 1, 1, 2).float() # shape (n, 1, 1, 2)
		center_loc = center_loc * 2 - 1
	if center_bias is None:
		center_bias = torch.ones(n).float() * 3.001

	grid = get_torch_normalized_meshgrid(ny, nx) # shape (ny, nx, 2)
	grid = grid.unsqueeze(dim=0).expand(n, ny, nx, 2) # shape (n, ny, nx, 2)

	elevation = F.grid_sample(satellite_elevation.unsqueeze(dim=1), grid, padding_mode='border', align_corners=True) # shape (n, 1, ny, nx)
	center_elevation = F.grid_sample(satellite_elevation.unsqueeze(dim=1), center_loc, padding_mode='border', align_corners=True) # shape (n, 1, 1, 1)
	if same_elevation:
		center_elevation = (center_elevation.max().expand(n) + center_bias).view(n, 1, 1, 1).expand(n, nz, ny, nx)
	else:
		center_elevation = (center_elevation.squeeze() + center_bias).view(n, 1, 1, 1).expand(n, nz, ny, nx)
	voxel_elevation = elevation.expand(n, nz, ny, nx) - center_elevation # shape (n, nz, ny, nx)

	voxel_dz = torch.arange(-int(nz/2), nz-int(nz/2), 1).float().view(1, nz, 1, 1).expand(n, nz, ny, nx)
	voxel_occupy = torch.ge(voxel_elevation, voxel_dz)

	voxel_dx = (grid[..., 0] - center_loc[..., 0]).unsqueeze(dim=1).expand(n, nz, ny, nx) * xy_scale * ncol / 2.0
	voxel_dy = (grid[..., 1] - center_loc[..., 1]).unsqueeze(dim=1).expand(n, nz, ny, nx) * xy_scale * nrow / 2.0
	voxel_dz = (voxel_dz + torch.fmod(center_bias, 1).view(n, 1, 1, 1).expand(n, nz, ny, nx)) * z_scale

	voxel_dis = torch.sqrt(voxel_dx ** 2 + voxel_dy ** 2 + voxel_dz ** 2)
	voxel_dis = voxel_dis * voxel_occupy + ~voxel_occupy * 1e9

	# visualize_voxel_grid(voxel_occupy[0].numpy().astype(np.bool))
	# visualize_voxel_grid(voxel_occupy[1].numpy().astype(np.bool))
	# visualize_voxel_grid(voxel_occupy[2].numpy().astype(np.bool))

	# print(depth.shape)
	# print(depth_c.shape)
	# print(center_bias.shape)
	# print(satellite_elevation.min(dim=-1)[0].min(dim=-1)[0])
	# print(satellite_elevation.max(dim=-1)[0].max(dim=-1)[0])

	# print(voxel_dis)
	# print(voxel_dis.shape)

	for i in range(3):
	# 	plt.imshow(satellite_elevation[i].numpy())
	# 	plt.show()
		continue
		# for j in range(128):
		# 	plt.imshow(voxel_dis[i, j].numpy(), vmin=0, vmax=256)
		# 	# plt.show(block=False)
		# 	plt.title(str(j))
		# 	plt.draw()
		# 	plt.pause(0.1)
		# plt.imshow(voxel_dx[i, 0].numpy())
		# plt.show()
		# plt.imshow(voxel_dy[i, 0].numpy())
		# plt.show()

	# print(depth_c)

	center_grid = ((center_loc + 1) / 2.0)[:, 0, 0, :]
	center_grid[:, 0] *= nx
	center_grid[:, 1] *= ny
	center_grid = torch.cat([center_grid, int(nz/2) * torch.ones(n, 1)], dim=-1)

	# print(center_grid)

	for i in range(3):
		continue
		xx, yy, zz = np.round(center_grid[i].numpy()).astype(np.int32)
		# print(xx, yy, zz)
		print(voxel_dx[i, zz, yy, xx-1], voxel_dx[i, zz, yy, xx], voxel_dx[i, zz, yy, xx+1])
		print(voxel_dy[i, zz, yy-1, xx], voxel_dy[i, zz, yy, xx], voxel_dy[i, zz, yy+1, xx])
		print(voxel_dz[i, zz, yy, xx])
	# 	for j in range(5):
	# 		print(voxel_occupy[i, zz-j, yy, xx])
		
		
	# 	# plt.plot(voxel_dz[i, :, yy, xx].numpy())
	# 	# plt.show()
	# 	print()

	return voxel_dis, center_grid




def voxel_grid_to_panorama(voxel_distance,
						   center_grid,
						   direction,
						   panorama_size=[512, 256],
						   min_max_lat=[-np.pi/2, np.pi/2],
						   ray_length=128,
						   ray_step=1.0):
	"""
	Input:
		voxel_distance:		(batch_size, nz, ny, nx)
		center_grid:		(batch_size, 3), each row [0 ~ nx, 0 ~ ny, 0 ~ nz]
		direction:			(batch_size) # rad
		panorama_size:		[width, height]
		min_max_lat:		[min_lat, max_lat]

	Output:
		panorama:			(batch_size, panorama_size[1], panorama_size[0])

	"""

	n, nz, ny, nx = voxel_distance.size()

	ncol, nrow = panorama_size
	min_lat, max_lat = min_max_lat
	cx, cy, cz = center_grid[:, 0], center_grid[:, 1], center_grid[:, 2]
	num_sample = int(np.ceil(ray_length / ray_step))

	delta_angle = direction.float().unsqueeze(dim=1).expand(n, ncol) # shape(n, ncol)
	x = torch.arange(0, ncol, 1, dtype=torch.float32).unsqueeze(dim=0).expand(n, ncol)
	y = torch.arange(0, nrow, 1, dtype=torch.float32).unsqueeze(dim=0).expand(n, nrow)
	lon = x / ncol * 2.0 * np.pi - np.pi + delta_angle
	lat = (1.0 - y / nrow) * (max_lat - min_lat) + min_lat
	sin_lon = torch.sin(lon).view(n, 1, 1, ncol).expand(n, 1, nrow, ncol)
	cos_lon = torch.cos(lon).view(n, 1, 1, ncol).expand(n, 1, nrow, ncol)
	sin_lat = torch.sin(lat).view(n, 1, nrow, 1).expand(n, 1, nrow, ncol)
	cos_lat = torch.cos(lat).view(n, 1, nrow, 1).expand(n, 1, nrow, ncol)

	# Compute the unit vector of each pixel
	vx = cos_lat.mul(sin_lon) # shape (n, 1, nrow, ncol)
	vy = cos_lat.mul(cos_lon) * -1.0 # shape (n, 1, nrow, ncol)
	vz = sin_lat # shape (n, 1, nrow, ncol)

	sample = (torch.arange(0, num_sample, 1, dtype=torch.float32).view(1, num_sample, 1, 1) * ray_step).expand(n, num_sample, nrow, ncol)
	idx_x = torch.clamp(sample * vx.expand(n, num_sample, nrow, ncol) + cx.view(n, 1, 1, 1).expand(n, num_sample, nrow, ncol), 0, nx - 1).long()
	idx_y = torch.clamp(sample * vy.expand(n, num_sample, nrow, ncol) + cy.view(n, 1, 1, 1).expand(n, num_sample, nrow, ncol), 0, ny - 1).long()
	idx_z = torch.clamp(sample * vz.expand(n, num_sample, nrow, ncol) + cz.view(n, 1, 1, 1).expand(n, num_sample, nrow, ncol), 0, nz - 1).long()
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

	num = 9
	step = 10.0
	bs = num
	step = np.array([list(range(-int(num/2), num-int(num/2)))] * 2).T * step
	org_center_loc = np.array([[128, 128]] * num)

	# files = sorted(glob.glob('/home/zoli/xiaohu_new_data/train2_new/*_sate_depth.png'))
	files = sorted(glob.glob('../Pano2StreetVideoOld/data/*_sate_depth.png'))

	for file in files:

		img = np.array(Image.open(file.replace('_sate_depth.png', '_street_rgb.png')))
		img_tensor = torch.from_numpy(img)
		# plt.imshow(img)
		# plt.show()

		direction = float(file.replace('_sate_depth.png', '').split(',')[2]) / 360.0 * np.pi * 2.0

		elevation_batch = np.stack([np.array(Image.open(file).convert('L'))] * num)
		elevation_batch = torch.from_numpy(elevation_batch).float()
		direction_batch = np.array([direction] * num)
		direction_batch = torch.from_numpy(direction_batch).float()

		centor_loc_batch = org_center_loc + np.stack([[np.sin(direction), -np.cos(direction)]] * num) * step
		centor_loc_batch = torch.from_numpy(centor_loc_batch).float()

		for i in range(int(num/bs)):

			voxel_unsigned_distance, center_grid = satellite_elevation_to_voxel_grid(elevation_batch[0+i*bs:bs+i*bs], center_loc=centor_loc_batch[0+i*bs:bs+i*bs], grid_shape=[256]*3, xy_scale=0.25, z_scale=0.5)
			panorama_batch, panorama_voxel_idx_batch = voxel_grid_to_panorama(voxel_unsigned_distance, center_grid, direction_batch[0+i*bs:bs+i*bs])
			warpped_img_batch = warp(img_tensor, panorama_voxel_idx_batch, int(num / 2))

			for j in range(bs):
				img = (panorama_batch[j].numpy())
				img[img>116]=116
				img = img/img.max() * 255

			# 	img = panorama_voxel_idx_batch[j].numpy()
			# 	a = img%256
			# 	a_res = ((img-a)/256).astype(np.int32)
			# 	b = a_res%256
			# 	c = ((a_res-b)/256).astype(np.int32)%256
			# 	img = np.stack([c,b,a], axis=-1)

				plt.imshow(warpped_img_batch[j].numpy())
				plt.savefig('%d.png' % (j+i*bs))
				plt.clf()
				# plt.show()

		quit()

		plt.imshow(np.array(Image.open(file.replace('_sate_depth.png', '_proj_dis.png')).convert('L')))
		plt.show()

		quit()


