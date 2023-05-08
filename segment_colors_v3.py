from matplotlib import pyplot as plt
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans



def preview(image, desc="test"):
    while 1:
        cv2.imshow(desc, image)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


def pick_colors(image, n, desc="Choose colors"):
    global counter
    while counter < n:
        cv2.imshow(desc, image)
        cv2.setMouseCallback(desc, mouse_points)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


def mouse_points(event, x, y, flags, params):
    global counter
    global clicked_coords_mat
    global im
    global picked_rgb
    # Left button click
    if event == cv2.EVENT_LBUTTONDOWN:
        print(counter)
        print(y, x)
        print(np.flip(im[y, x]))  # reversed(BGR) = RGB
        print(rgb_to_hsv(np.flip(im[y, x])))
        clicked_coords_mat[counter] = y, x
        cur_cols = []
        for i in range(y-2, y+3):
            if i < 0 or i == im.shape[0]:
                continue
            for j in range(x-2, x+3):
                if j < 0 or j == im.shape[1]:
                    continue
                cur_cols.append(np.flip(im[i, j]))
        cur_cols = np.vstack(cur_cols)
        picked_rgb[counter] = np.median(cur_cols, axis=0)
        counter = counter + 1


def rgb_to_hsv(rgb_arr):
    r = rgb_arr[0]
    g = rgb_arr[1]
    b = rgb_arr[2]
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h/2, s*255, v


def increase_saturation(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    s[s > lim] = 255
    s[s <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def decrease_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = value
    v[v < lim] = 0
    v[v >= lim] -= value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# def decrease_saturation(img, value=)




image = cv2.imread('images/new_test.jpg')
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.axis('off')
#plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray, cmap = "gray")
#plt.axis('off')
#plt.show()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

s_cutoff = 40
v_cutoff = 50
#_ , mask = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY)
_ , mask1 = cv2.threshold(hsv[:,:,1], s_cutoff, 255, cv2.THRESH_BINARY)
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))
_ , mask2 = cv2.threshold(hsv[:,:,2], v_cutoff, 255, cv2.THRESH_BINARY)
mask2 = cv2.bitwise_not(mask2)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))
#mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
#plt.imshow(mask, cmap = "gray")
#plt.axis('off')
mask = cv2.add(mask1, mask2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)))
#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)))
cv2.imwrite(f'thresholding_{s_cutoff}_{v_cutoff}.png', cv2.hconcat([image, np.stack((mask1, mask1, mask1), axis=2), np.stack((mask2, mask2, mask2), axis=2), np.stack((mask, mask, mask), axis=2)]))
#cv2.imwrite(f'thresholding_{cutoff}.png', cv2.hconcat([image, np.stack((mask1, mask1, mask1), axis=2)]))





contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
contours_img_before_filtering = mask.copy()
contours_img_before_filtering = cv2.cvtColor(contours_img_before_filtering, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contours_img_before_filtering, contours, -1, (0, 255, 0), 3)
plt.imshow(contours_img_before_filtering)
plt.axis('off')
plt.savefig('contours.png')




filtered_contours = []
df_mean_color = pd.DataFrame()
for idx, contour in enumerate(contours):
    area = int(cv2.contourArea(contour))
    # if area is higher than 3000:
    if area > 200:
        filtered_contours.append(contour)
        # get mean color of contour:
        masked = np.zeros_like(image[:, :, 0])  # This mask is used to get the mean color of the specific bead (contour), for kmeans
        cv2.drawContours(masked, [contour], 0, 255, -1)
        B_mean, G_mean, R_mean, _ = cv2.mean(image, mask=masked)
	h,s,v = rgb_to_hsv([R_mean,G_mean,B_mean])
        df = pd.DataFrame({'B_mean': B_mean, 'G_mean': G_mean, 'R_mean': R_mean, 'H':h,'S':s,'V':v}, index=[idx])
        df_mean_color = pd.concat([df_mean_color, df])


df_mean_color.head()


contours_img_after_filtering = mask.copy()
contours_img_after_filtering = cv2.cvtColor(contours_img_after_filtering, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contours_img_after_filtering, tuple(filtered_contours), -1, (0, 255, 0), 3)
plt.imshow(contours_img_after_filtering)
plt.axis('off')
cv2.imwrite('contours_filtered.png', cv2.hconcat([image, contours_img_after_filtering]))






km = KMeans( n_clusters=8)
df_mean_color['label'] = km.fit_predict(df_mean_color)

background = increase_saturation(image, value=125)
background = decrease_brightness(background, value=75)
gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

def draw_segmented_objects(image, contours, label_cnt_idx, count):
    mask = np.zeros_like(image[:, :, 0])
    cv2.drawContours(mask, [contours[i] for i in label_cnt_idx], -1, (255), -1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100)), iterations=1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    bg = cv2.bitwise_and(gray_bg, gray_bg, mask=cv2.bitwise_not(mask))
    bg = np.stack((bg,)*3, axis=-1)
    out = cv2.add(masked_image, bg)
    out = cv2.putText(out, f'{count} holds', (200, 1200), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 3, color = (255, 255, 255), thickness = 10, lineType = cv2.LINE_AA)
    return out


def draw_segmented_objects2(image, contours, label_cnt_idx, count):
    mask = np.zeros_like(image[:, :, 0])
    cv2.drawContours(mask, [contours[i] for i in label_cnt_idx], -1, (255), -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    out = cv2.putText(masked_image, f'{count} holds', (200, 1200), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 3, color = (255, 255, 255), thickness = 10, lineType = cv2.LINE_AA)
    return out



img = cv2.copyMakeBorder(image.copy(), top=5, bottom=5, left=5, right=5, borderType=cv2.BORDER_CONSTANT, value=[255,255,255])
for label, df_grouped in df_mean_color.groupby('label'):
    num_holds = len(df_grouped)
    masked_image = draw_segmented_objects2(image, contours, df_grouped.index, num_holds)
    masked_image = cv2.copyMakeBorder(masked_image, top=5, bottom=5, left=5, right=5, borderType=cv2.BORDER_CONSTANT, value=[255,255,255])
    img = cv2.hconcat([img, masked_image])


cv2.imwrite('test_segmentation_masks.png', img)








if __name__ == "__main__":
    im = cv2.imread('images/new_test.jpg')

    num_colors = 16  # Hue bins
    # Create a lookup table for the hue values
    hue_scale = np.linspace(0, 255, num_colors + 1).astype(np.uint8)  # see bottom
    hue_lut = np.zeros((1, 256), dtype=np.uint8)
    i = 0
    for i in range(num_colors):
        hue_lut[0, hue_scale[i]:hue_scale[i + 1]] = i * (256 // num_colors)
    hue_lut[0, 255] = i * (256 // num_colors)

    n = 1
    clicked_coords_mat = np.zeros((n, 2), int)
    picked_rgb = np.zeros((n, 3), int)
    counter = 0

    # Manually sample colors (pick colors from pixels around it)
    pick_colors(im, n)
    picked_hsv = rgb_to_hsv(picked_rgb[0])
    picked_h = picked_hsv[0]
    print(f"selected h: {picked_h}")
    picked_color = hue_lut[0, int(picked_h + 0.5)]
    print(f"h converted to discrete color: {picked_color}")
    picked_s = picked_hsv[1]
    picked_v = picked_hsv[2]

    # convert the input image to hsv
    hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # preview(hsv_im, desc="HSV")

    # Apply the hue LUT to the hue channel of the image
    hue_channel = hsv_im[:, :, 0]
    hue_discretized = cv2.LUT(hue_channel, hue_lut)
    hsv_im[:, :, 0] = hue_discretized
    bgr = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
    #preview(bgr, desc="BGR discretized")

    # Create mask on color discretized image
    low_h = picked_color - 1 if picked_color > 0 else 0
    high_h = picked_color + 1 if picked_color < 179 else 179
    high_s = int(np.ceil(picked_s/50.0)) * 50
    low_s = high_s - 50
    high_v = 50 if picked_v < 0.3 else 255
    if high_v == 50:
        high_s = 100
        low_s = 0
    print(f"low h = {low_h}, high h = {high_h}")
    print(f"high s = {high_s}")
    print(f"high v = {high_v}")
    mask = cv2.inRange(hsv_im, np.array([low_h, low_s, 0]), np.array([high_h, high_s, high_v]))
    # mask = cv2.inRange(hsv_im, np.array([0, 50, 20]), np.array([5, 255, 255]))
    # mask = cv2.inRange(hsv_im, np.array([60, 35, 140]), np.array([180, 255, 255]))
    #preview(mask, desc="mask init")
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # open
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # close
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # dilate
    # Open: erosion then dilation - to remove tiny spots else where
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    #preview(mask, desc="mask open")
    # Close: dilation then erosion - to remove tiny spots inside the selected region
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    #preview(mask, desc="mask close")
    mask = cv2.dilate(mask, kernel3, iterations=1)
    #preview(mask, desc="mask final")
    # create an inverse of the mask
    mask_inv = cv2.bitwise_not(mask)
    # Filter only the selected color from the original image
    # res = increase_saturation(im, value=50)
    res = cv2.bitwise_and(im, im, mask=mask)
    # preview(res)
    background = increase_brightness(im, value=75)
    background = decrease_saturation(background, value=50)
    gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background = cv2.bitwise_and(gray_bg, gray_bg, mask=mask_inv)
    # preview(background)
    # convert the one channelled grayscale background to a three channelled image
    background = np.stack((background,)*3, axis=-1)
    # super impose white by addWeighted()
    # add the foreground and the background
    out = cv2.add(res, background)
    preview(out)



"""
The reason for using the 0-179 range instead of the more common 0-360 range is that 
OpenCV represents the hue values as 8-bit integers, which can only hold values from 
0 to 255. By using the 0-179 range, OpenCV can represent the full range of hues using 
just one byte per pixel, which makes it more efficient for image processing operations.
"""
