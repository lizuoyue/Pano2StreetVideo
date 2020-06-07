import glob, tqdm
import numpy as np
from PIL import Image

if __name__ == '__main__':

	files = sorted(glob.glob('/home/zoli/xiaohu_new_data/train2_new/*_sate_depth.png'))
	for file in files:

		depth = np.array(Image.open(file).convert('L'))
		label = np.array(Image.open(file.replace('_sate_depth.png', '_street_label.png')).convert('RGB')).astype(np.int32)
		label[..., 0] -= 70
		label[..., 1] -= 130
		label[..., 2] -= 180

		sky1 = depth > 250
		sky2 = (label ** 2).sum(axis=-1) == 0
		print((sky1 & sky2).sum() / (sk1 | sky2).sum())
