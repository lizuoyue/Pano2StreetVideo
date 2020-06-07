import glob, tqdm
import numpy as np
from PIL import Image

if __name__ == '__main__':

	res = []
	files = sorted(glob.glob('/home/zoli/xiaohu_new_data/train2_new/*_proj_depth.png'))
	for file in tqdm.tqdm(files):

		depth = np.array(Image.open(file).convert('L'))
		label = np.array(Image.open(file.replace('_proj_depth.png', '_street_label.png')).convert('RGB')).astype(np.int32)
		label[..., 0] -= 70
		label[..., 1] -= 130
		label[..., 2] -= 180

		sky1 = depth > 250
		sky2 = (label ** 2).sum(axis=-1) == 0
		res.append(((sky1 & sky2).sum() / (sky1 | sky2).sum(), file))

	for item in res:
		print(item)