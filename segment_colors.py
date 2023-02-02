import cv2
import numpy as np


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
        clicked_coords_mat[counter] = y, x
        cur_cols = []
        for i in range(y-1, y+2):
            if i < 0 or i == im.shape[0]:
                continue
            for j in range(x-1, x+2):
                if j < 0 or j == im.shape[1]:
                    continue
                cur_cols.append(np.flip(im[i, j]))
        cur_cols = np.vstack(cur_cols)
        picked_rgb[counter] = np.mean(cur_cols, axis=0)
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


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
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


if __name__ == "__main__":
    im = cv2.imread('images/test.jpg')
    # preview(im, desc="Original")
    # blur = cv2.blur(im, (7,7))

    n = 7
    clicked_coords_mat = np.zeros((n, 2), int)
    picked_rgb = np.zeros((n, 3), int)
    counter = 0

    # Manually sample colors (pick colors from pixels around it)
    pick_colors(im, n)
    print(picked_rgb)
    picked_hsv = [rgb_to_hsv(x) for x in picked_rgb]
    print(picked_hsv)
    hs = [x[0] for x in picked_hsv]
    low_h = min(hs) - 1 if min(hs) >= 1 else min(hs) + 178
    high_h = max(hs) + 1 if max(hs) < 179 else max(hs) - 179
    ss = [x[1] for x in picked_hsv]
    low_s = min(ss) - 5 if min(ss) >= 5 else 0
    high_s = max(ss) + 5 if max(ss) <= 250 else 255
    print(low_h, high_h)
    print(low_s, high_s)

    # convert the input image to hsv
    hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # preview(hsv_im, desc="HSV")
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # hsv hue, saturation and brightness
    # hue int [0,179]; color spectrum think angle
    # saturation int [0,255]; basically color 0=gray
    # brightness int [0, 255]; 0=dark
    # define range of blue color in HSV
    # hvals = np.linspace(0,179,20)
    # for i in range(len(hvals)-1):
    mask = cv2.inRange(hsv_im, np.array([low_h, low_s, 20]), np.array([high_h, high_s, 255]))
    preview(mask, desc="mask")
    # kernel = np.ones((20, 20), np.uint8)
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    preview(mask, desc="mask")
    # create an inverse of the mask
    mask_inv = cv2.bitwise_not(mask)
    # Filter only the selected color from the original image
    res = cv2.bitwise_and(im, im, mask=mask)
    # increase the brightness of selected areas
    res = increase_brightness(res)
    # Filter the regions containing colours other than red from the grayscale image(background)
    background = cv2.bitwise_and(gray_im, gray_im, mask=mask_inv)
    # convert the one channelled grayscale background to a three channelled image
    background = np.stack((background,)*3, axis=-1)
    background = decrease_brightness(background, value=50)
    # add the foreground and the background
    out = cv2.add(res, background)
    preview(out)
