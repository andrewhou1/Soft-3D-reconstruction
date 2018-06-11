from rectify_images import *
from initial_depth_estimation import *
import os, sys
import numpy as np

def main():
	'''(threeD_pos1, orientation_matrix1, focal_len1, principal_point1) = read_camera_parameters(sys.argv[1])	
	(threeD_pos2, orientation_matrix2, focal_len2, principal_point2) = read_camera_parameters(sys.argv[2])
	img1 = cv2.imread(sys.argv[3])
	img2 = cv2.imread(sys.argv[4])
	(img1_rect, img2_rect) = rectify(threeD_pos1, threeD_pos2, orientation_matrix1, orientation_matrix2, focal_len1, focal_len2, principal_point1, principal_point2, 1024, 1024, img1, img2)'''
	img_and_parameter_dir = sys.argv[1]
	camera_parameter_files = sorted(f for f in os.listdir(img_and_parameter_dir) if f.endswith('.txt'))	
	img_files = sorted(f for f in os.listdir(img_and_parameter_dir) if f.endswith('.jpg'))
	for img_idx in range(len(img_files)):
		(neighbor_param_files, neighbor_img_files) = get_neighbors(camera_parameter_files, img_files, img_idx, 5)
		print(neighbor_param_files)
		print(neighbor_img_files)
		img = cv2.imread(img_and_parameter_dir+neighbor_img_files[0])
		img_height, img_width, channels = img.shape
		current_img_rectified = np.zeros((img_height, img_width, 3, len(neighbor_img_files)))
		rectified_neighbors = np.zeros((img_height, img_width, 3, len(neighbor_img_files)))
		horizontal_rec = np.zeros((len(neighbor_img_files)))
		current_camera_focal_len = 0
		neighbor_focal_lens = np.zeros((len(neighbor_img_files)))
		current_3D_pose = np.zeros((3))
		neighbor_3D_poses = np.zeros((len(neighbor_img_files), 3))
		rectification_matrices = np.zeros((3,3,len(neighbor_img_files)))
		projection_matrices = np.zeros((3, 4, len(neighbor_img_files)))
		distortion_coeffs = np.zeros((1, 5, len(neighbor_img_files)))
		orig_camera_matrix = np.zeros((3, 3))
		for i in range(len(neighbor_param_files)):
			(threeD_pos1, orientation_matrix1, focal_len1, principal_point1) = read_camera_parameters(img_and_parameter_dir+camera_parameter_files[img_idx])	
			(threeD_pos2, orientation_matrix2, focal_len2, principal_point2) = read_camera_parameters(img_and_parameter_dir+neighbor_param_files[i])
			img1 = cv2.imread(img_and_parameter_dir+img_files[img_idx])
			img2 = cv2.imread(img_and_parameter_dir+neighbor_img_files[i])
			(img1_rect, img2_rect, horizontally_rectified, R1, R2, camera_matrix1, P1, distortion_coeffs1) = rectify(threeD_pos1, threeD_pos2, orientation_matrix1, orientation_matrix2, focal_len1, focal_len2, principal_point1, principal_point2, img_height, img_width, img1, img2)		
			current_img_rectified[:, :, :, i] = img1_rect
			rectified_neighbors[:, :, :, i] = img2_rect
			horizontal_rec[i] = horizontally_rectified
			current_camera_focal_len = focal_len1
			current_3D_pose = threeD_pos1
			neighbor_focal_lens[i] = focal_len2
			neighbor_3D_poses[i, :] = threeD_pos2
			rectification_matrices[:, :, i] = R1
			projection_matrices[:, :, i] = P1
			distortion_coeffs[:, :, i] = distortion_coeffs1
			orig_camera_matrix = camera_matrix1
			print(current_img_rectified)
			print(rectified_neighbors)
			print(horizontal_rec) 	
		estimated_depth_map = np.zeros((img_height, img_width))
		cumulative_SADs = cumulative_SAD(current_img_rectified, rectified_neighbors, current_camera_focal_len, neighbor_focal_lens, current_3D_pose, neighbor_3D_poses, horizontal_rec, rectification_matrices)
		guide_img = cv2.imread(img_and_parameter_dir+img_files[img_idx])
		guide_img = cv2.cvtColor(guide_img, cv2.COLOR_BGR2GRAY)
		cumulative_SADs = guided_filter_step(cumulative_SADs, guide_img) 
		print(cumulative_SADs[:, :, 20, 0])
		for j in range(img_height):
			for k in range(img_width):
				estimated_depth_map[j, k] = initial_depth_estimate(current_img_rectified, rectified_neighbors, horizontal_rec, current_camera_focal_len, neighbor_focal_lens, current_3D_pose, neighbor_3D_poses, j, k, cumulative_SADs)
				#if(k == 500):
				#	print(str(j)+', '+str(k)+': '+str(estimated_depth_map[j, k]))			
		print(estimated_depth_map)
		if(img_idx < 10):
			filename = "depth_map_0"+str(img_idx)+".npy"
		else: 
			filename = "depth_map_"+str(img_idx)+".npy"
		np.save(sys.argv[2]+filename, estimated_depth_map)
		'''print(rectification_matrices[:, :, 0])
		fig, ax = plt.subplots(nrows=1, ncols=2)
		plt.subplot(1, 2, 1)
		plt.imshow(current_img_rectified[:, :, :, 0])
		plt.subplot(1, 2, 2)
		plt.imshow(cv2.warpPerspective(current_img_rectified[:, :, :, 0], rectification_matrices[:, :, 0], (img_width, img_height)))
		plt.pause(1000)
		plt.show(block=False)'''
		'''estimated_depth_map = np.load(sys.argv[4])
		for i in range(len(neighbor_focal_lens)):
			estimated_depth_map = cv2.warpPerspective(estimated_depth_map, rectification_matrices[:, :, i], (img_width, img_height))'''

	
		print(rectification_matrices[:, :, 2])
		print(np.linalg.inv(rectification_matrices[:, :, 2]))
		print(np.matmul(rectification_matrices[:, :, 2], np.linalg.inv(rectification_matrices[:, :, 2])))
		max_depth = np.amax(estimated_depth_map)
		min_depth = np.amin(estimated_depth_map)
		#print(max_depth)
		#print(min_depth)
		estimated_depth_map = estimated_depth_map-min_depth
		estimated_depth_map = np.uint8(255*estimated_depth_map.astype(float)/float(max_depth-min_depth))
		if(img_idx < 10):
			filename = "depth_map_0"+str(img_idx)+".png"
		else: 
			filename = "depth_map_"+str(img_idx)+".png"
		cv2.imwrite(sys.argv[3]+filename, estimated_depth_map)
if __name__ == "__main__":
	main() 

