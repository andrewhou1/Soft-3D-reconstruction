import cv2
import numpy as np
import sys, os
from matplotlib import pyplot as plt
import math

def read_camera_parameters(camera_file):
	with open(camera_file) as f:
		parameters = f.readlines()
		parameters = [x.strip() for x in parameters]
		all_camera_params = np.zeros((5, 3))
		for idx, line in enumerate(parameters):
			all_camera_params[idx, :] = np.fromstring(line, dtype=float, sep=' ')
	threeD_pos = all_camera_params[0, :]
	orientation_matrix = all_camera_params[1:4, :]
	focal_len = all_camera_params[4, 0]
	principal_point = all_camera_params[4, 1:3]
	return (threeD_pos, orientation_matrix, focal_len, principal_point)
		

def rectify(threeD_pos1, threeD_pos2, orientation_matrix1, orientation_matrix2, focal_len1, focal_len2, principal_point1, principal_point2, img_height, img_width, img_1, img_2):
	horizontally_rectified = True
	camera_matrix1 = np.zeros((3, 3))
	camera_matrix2 = np.zeros((3, 3))
	camera_matrix1[0, 0] = focal_len1
	camera_matrix1[1, 1] = focal_len1
	camera_matrix1[0, 2] = principal_point1[0]
	camera_matrix1[1, 2] = principal_point1[1]
	camera_matrix1[2, 2] = 1.0
	camera_matrix2[0, 0] = focal_len2
	camera_matrix2[1, 1] = focal_len2
	camera_matrix2[0, 2] = principal_point2[0]
	camera_matrix2[1, 2] = principal_point2[1]
	camera_matrix2[2, 2] = 1.0
	distortion_coeffs1 = np.zeros((1,5))
	distortion_coeffs2 = np.zeros((1,5))
	rotation_matrix = np.matmul(np.linalg.inv(orientation_matrix2), orientation_matrix1)
	translation_vector = threeD_pos2-threeD_pos1
	(R1, R2, P1, P2, Q, roi1, roi2) = cv2.stereoRectify(camera_matrix1, distortion_coeffs1, camera_matrix2, distortion_coeffs2, (img_width, img_height), rotation_matrix, translation_vector, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(img_width, img_height))	
	print(R1)
	print(R2)
	print(P1)
	print(P2)
	print(Q)
	print(roi1)
	print(roi2)
	if(math.fabs(P2[1, 3]) > 0.0000001):
		horizontally_rectified = False
	map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix=camera_matrix1, distCoeffs=distortion_coeffs1, R=R1, newCameraMatrix=P1, size=(img_width, img_height), m1type=cv2.CV_32FC1)	
	map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix=camera_matrix2, distCoeffs=distortion_coeffs2, R=R2, newCameraMatrix=P2, size=(img_width, img_height), m1type=cv2.CV_32FC1)
	img1_rect=cv2.remap(img_1, map1x, map1y, cv2.INTER_LINEAR)
	img2_rect=cv2.remap(img_2, map2x, map2y, cv2.INTER_LINEAR)
	'''fig, ax = plt.subplots(nrows=2, ncols=2)
	plt.subplot(2, 2, 1)
	plt.imshow(img_1)
	plt.subplot(2, 2, 2)
	plt.imshow(img1_rect)
	plt.subplot(2, 2, 3)
	plt.imshow(img_2)
	plt.subplot(2, 2, 4)
	plt.imshow(img2_rect)

	plt.show(block=False)	
	plt.pause(15)
	plt.close()'''
	#return (R1, R2, P1, P2, Q, roi1, roi2)
	return (img1_rect, img2_rect, horizontally_rectified)

'''def main():
	(threeD_pos1, orientation_matrix1, focal_len1, principal_point1) = read_camera_parameters(sys.argv[1])	
	(threeD_pos2, orientation_matrix2, focal_len2, principal_point2) = read_camera_parameters(sys.argv[2])
	img1 = cv2.imread(sys.argv[3])
	img2 = cv2.imread(sys.argv[4])
	(img1_rect, img2_rect) = rectify(threeD_pos1, threeD_pos2, orientation_matrix1, orientation_matrix2, focal_len1, focal_len2, principal_point1, principal_point2, 1024, 1024, img1, img2)	

if __name__ == "__main__":
	main()''' 
