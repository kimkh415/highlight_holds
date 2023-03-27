Meant to be used in HoldUp app to make climbing problems more conspicuous

# Function descriptions
##### **create_mask**
This function takes an image, x, and y coordinates as input and returns a binary mask that highlights a region around the selected color.

The function first converts the input image to the HSV color space and discretizes the hue channel using a lookup table. Then, it extracts the selected color from the input image and computes the low and high values of hue, saturation, and value (brightness) for the mask. The function uses these values to create a binary mask by thresholding the HSV image using the cv2.inRange() function.

Next, the function applies morphological operations to the mask to remove tiny spots outside and inside the selected region using the cv2.morphologyEx() function with an elliptical kernel. Finally, the function dilates the mask using an elliptical kernel with a specified size.

The mask_kernel_sizes parameter is optional and specifies the sizes of the elliptical kernels used in the morphological operations. If mask_kernel_sizes is not provided, the function uses default values of [2, 15, 30] for the kernel sizes.




