import numpy as np 

sseg_color_list_19 = [(128, 64, 128), (244, 36, 232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), 
	(250,170, 30), (220,220,  0), (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), 
	(  0,  0,142), (  0,  0, 70), (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32)]
sseg_color_list_8 = [(128, 64, 128), ( 70, 70, 70), (153, 153, 153), (107, 142, 35), (70, 130, 180), (220, 20, 60),
	(0, 0, 142), (0, 80, 100)]
	
def apply_color_map(image_array, num_classes=8):
	if num_classes == 19:
		sseg_color_list = sseg_color_list_19
	elif num_classes == 8:
		sseg_color_list = sseg_color_list_8

	color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
	for label_id, color in enumerate(sseg_color_list):
		color_array[image_array == label_id] = color
	return color_array