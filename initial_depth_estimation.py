import math
import numpy as np
from scipy.ndimage.interpolation import shift

def get_neighbors(param_filelist, img_filelist, idx, images_per_row):
	neighbor_param_files = []
	neighbor_img_files = []
	if(idx-images_per_row-1 >= 0 and (((idx-images_per_row-1) % images_per_row) < (idx % images_per_row))):
		upper_left_neighbor_params = param_filelist[idx-images_per_row-1]
		upper_left_neighbor_img = img_filelist[idx-images_per_row-1]
		neighbor_param_files.append(upper_left_neighbor_params)
		neighbor_img_files.append(upper_left_neighbor_img)
	if(idx-images_per_row >= 0):
		upper_neighbor_params = param_filelist[idx-images_per_row]
		upper_neighbor_img = img_filelist[idx-images_per_row]
		neighbor_param_files.append(upper_neighbor_params)
		neighbor_img_files.append(upper_neighbor_img)
	if(idx-images_per_row+1 >= 0 and (((idx-images_per_row+1) % images_per_row) > (idx % images_per_row))):
		upper_right_neighbor_params = param_filelist[idx-images_per_row+1]
		upper_right_neighbor_img = img_filelist[idx-images_per_row+1]
		neighbor_param_files.append(upper_right_neighbor_params)
		neighbor_img_files.append(upper_right_neighbor_img)
	if(idx-1 >= 0 and (((idx-1) % images_per_row) < (idx % images_per_row))):
		left_neighbor_params = param_filelist[idx-1]
		left_neighbor_img = img_filelist[idx-1]
		neighbor_param_files.append(left_neighbor_params)
		neighbor_img_files.append(left_neighbor_img)
	if(idx+1 < len(param_filelist) and (((idx+1) % images_per_row) > (idx % images_per_row))):
		right_neighbor_params = param_filelist[idx+1]
		right_neighbor_img = img_filelist[idx+1]
		neighbor_param_files.append(right_neighbor_params)
		neighbor_img_files.append(right_neighbor_img)
	if(idx+images_per_row-1 < len(param_filelist) and (((idx+images_per_row-1) % images_per_row) < (idx % images_per_row))):
		bottom_left_neighbor_params=param_filelist[idx+images_per_row-1]
		bottom_left_neighbor_img = img_filelist[idx+images_per_row-1]
		neighbor_param_files.append(bottom_left_neighbor_params)
		neighbor_img_files.append(bottom_left_neighbor_img)
	if(idx+images_per_row < len(param_filelist)):
		bottom_neighbor_params = param_filelist[idx+images_per_row]
		bottom_neighbor_img = img_filelist[idx+images_per_row]
		neighbor_param_files.append(bottom_neighbor_params)
		neighbor_img_files.append(bottom_neighbor_img)
	if(idx+images_per_row+1 < len(param_filelist) and (((idx+images_per_row+1) % images_per_row) > (idx % images_per_row))):
		bottom_right_neighbor_params = param_filelist[idx+images_per_row+1]
		bottom_right_neighbor_img = img_filelist[idx+images_per_row+1]
		neighbor_param_files.append(bottom_right_neighbor_params)
		neighbor_img_files.append(bottom_right_neighbor_img)

	return (neighbor_param_files, neighbor_img_files)

def cumulative_SAD(current_img_rectified, rectified_neighbors, current_camera_focal_len, neighbor_focal_lens, current_3D_pose, neighbor_3D_poses, horizontal_rec):
	dims = current_img_rectified.shape
	cumulative_SADs = np.zeros((dims[0], dims[1], 30, len(neighbor_focal_lens)))
	for i in range(1, 30):
		print(i)
		Z = float(i)/10
		for j in range(len(neighbor_focal_lens)):
			current_neighbor_pose = neighbor_3D_poses[j, :]
			baseline = math.sqrt((current_3D_pose[0]-current_neighbor_pose[0])**2+(current_3D_pose[1]-current_neighbor_pose[1])**2+(current_3D_pose[2]-current_neighbor_pose[2])**2)
			disparity = current_camera_focal_len*baseline/Z
			horizontally_rectified = horizontal_rec[j]
			if(horizontally_rectified):
				#shifted_img = np.roll(current_img_rectified[:, :, :, j], disparity, axis=1)
				shiftvec = (0, disparity, 0)
				shifted_img = shift(current_img_rectified[:, :, :, j], shift=shiftvec, order=0)
				cost = np.abs(shifted_img-rectified_neighbors[:, :, :, j])
				cumulative_SADs[:, :, i, j] = np.cumsum(np.cumsum(np.sum(cost, axis=2), axis=0), axis=1)
			else:
				#shifted_img = np.roll(current_img_rectified[:, :, :, j], disparity, axis=0)
				shiftvec = (disparity, 0, 0)
				shifted_img = shift(current_img_rectified[:, :, :, j], shift=shiftvec, order=0)
				cost = np.abs(shifted_img-rectified_neighbors[:, :, :, j])
				cumulative_SADs[:, :, i, j] = np.cumsum(np.cumsum(np.sum(cost, axis=2), axis=0), axis=1)
	return cumulative_SADs

def initial_depth_estimate(current_img_rectified, rectified_neighbors, horizontal_rec, current_camera_focal_len, neighbor_focal_lens, current_3D_pose, neighbor_3D_poses, row, col, cumulative_SADs):
	all_depth_errors = np.zeros((30))
	all_depth_errors[0] = float("inf")
	img = current_img_rectified[:, :, :, 0]
	img_height, img_width, channels = img.shape
	
	if(row < 4 or col < 4 or row > img_height-4 or col > img_width-4):
		return 0.0
	
	for i in range(1, 30):
		Z = float(i)/10
		#print("Depth: "+str(Z))
		for j in range(len(neighbor_focal_lens)):
			current_neighbor_pose = neighbor_3D_poses[j, :]
			baseline = math.sqrt((current_3D_pose[0]-current_neighbor_pose[0])**2+(current_3D_pose[1]-current_neighbor_pose[1])**2+(current_3D_pose[2]-current_neighbor_pose[2])**2)
			disparity = current_camera_focal_len*baseline/Z
			horizontally_rectified = horizontal_rec[j]
			if(horizontally_rectified):
				if(disparity > col-4):
					all_depth_errors[i] = float("inf")
					break
				#shifted_img = np.roll(current_img_rectified[:, :, :, j], disparity, axis=1)
				#all_depth_errors[i] += np.sum(np.absolute(shifted_img[row-4:row+4, col-4:col+4, :]-rectified_neighbors[row-4:row+4, col-4:col+4, :, j]))
				all_depth_errors += (cumulative_SADs[row+3, col+3, i, j]-cumulative_SADs[row-4, col+3, i, j]-cumulative_SADs[row+3, col-4, i, j]+cumulative_SADs[row-4, col-4, i, j])
			else:
				if(disparity > row-4):
					all_depth_errors[i] = float("inf")
					break
				#shifted_img = np.roll(current_img_rectified[:, :, :, j], disparity, axis=0)
				#all_depth_errors[i] += np.sum(np.absolute(shifted_img[row-4:row+4, col-4:col+4, :]-rectified_neighbors[row-4:row+4, col-4:col+4, :, j]))
				all_depth_errors += (cumulative_SADs[row+3, col+3, i, j]-cumulative_SADs[row-4, col+3, i, j]-cumulative_SADs[row+3, col-4, i, j]+cumulative_SADs[row-4, col-4, i, j])

	return float(np.argmin(all_depth_errors))/10	
