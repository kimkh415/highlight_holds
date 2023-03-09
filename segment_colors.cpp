#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void preview(Mat image, string desc = "test") {
    while (1) {
        imshow(desc, image);
        if (waitKey(10) & 0xFF == 27) {
            break;
        }
    }
    destroyAllWindows();
}

int counter;
Mat clicked_coords_mat;
Mat picked_rgb;

void pick_colors(Mat image, int n, string desc = "Choose colors") {
    counter = 0;
    clicked_coords_mat = Mat(n, 2, CV_32SC1, Scalar(0));
    picked_rgb = Mat(n, 3, CV_32SC1, Scalar(0));
    while (counter < n) {
        imshow(desc, image);
        setMouseCallback(desc, [](int event, int x, int y, int flags, void* userdata) {
            if (event == EVENT_LBUTTONDOWN) {
                int i = counter;
                int* clicked_coords_data = (int*)clicked_coords_mat.data;
                int* picked_rgb_data = (int*)picked_rgb.data;
                cout << i << endl;
                cout << y << " " << x << endl;
                Vec3b bgr = image.at<Vec3f>(y, x);
                cout << (int)bgr[2] << " " << (int)bgr[1] << " " << (int)bgr[0] << endl;
                clicked_coords_data[i * 2] = y;
                clicked_coords_data[i * 2 + 1] = x;
                vector<Vec3b> cur_cols;
                for (int j = y - 1; j <= y + 1; j++) {
                    if (j < 0 || j == image.rows) {
                        continue;
                    }
                    for (int k = x - 1; k <= x + 1; k++) {
                        if (k < 0 || k == image.cols) {
                            continue;
                        }
                        cur_cols.push_back(image.at<Vec3b>(j, k));
                    }
                }
                Mat(cur_cols).reshape(1, cur_cols.size()).convertTo(cur_cols, CV_32SC1);
                Mat mean_rgb = mean(cur_cols);
                picked_rgb_data[i * 3] = mean_rgb.at<int>(0);
                picked_rgb_data[i * 3 + 1] = mean_rgb.at<int>(1);
                picked_rgb_data[i * 3 + 2] = mean_rgb.at<int>(2);
                counter++;
            }
        }, NULL);
        if (waitKey(10) & 0xFF == 27) {
            break;
        }
    }
    destroyAllWindows();
}

vector<int> rgb_to_hsv(Vec3i rgb_arr) {
    float r = rgb_arr[0];
    float g = rgb_arr[1];
    float b = rgb_arr[2];
    r = r / 255.0;
    g = g / 255.0;
    b = b / 255.0;
    float mx = max(r, max(g, b));
    float mn = min(r, min(g, b));
    float df = mx - mn;
    float h, s, v;
    if (mx == mn) {
        h = 0;
    } else if (mx == r) {
        h = fmod(60 * ((g - b) / df) + 360, 360);
    } else if (mx == g) {
        h = fmod(60 * ((b - r) / df) + 120, 360);
    } else if (mx == b) {
        h = fmod(60 * ((r - g) / df) + 240, 360);
    }
    if (mx == 0) {
        s = 0;
    } else {
        s = df / mx;
    }
    v = mx;
    return h / 2, s * 255, v * 255;
}


cv::Mat increase_brightness(cv::Mat img, int value = 30) {
    cv::Mat hsv;
    cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_split;
    cv::split(hsv, hsv_split);
    cv::Mat v_channel = hsv_split[2];

    int lim = 255 - value;
    cv::threshold(v_channel, v_channel, lim, 255, cv::THRESH_TOZERO_INV);
    v_channel = v_channel + value;

    cv::merge(hsv_split, hsv);
    cv::cvtColor(hsv, img, cv::COLOR_HSV2BGR);

    return img;
}


cv::Mat decrease_saturation(cv::Mat img, int value = 30) {
    cv::Mat hsv;
    cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_split;
    cv::split(hsv, hsv_split);
    cv::Mat s_channel = hsv_split[1];

    int lim = value;
    cv::threshold(s_channel, s_channel, lim, 255, cv::THRESH_TOZERO);
    s_channel = s_channel - value;

    cv::merge(hsv_split, hsv);
    cv::cvtColor(hsv, img, cv::COLOR_HSV2BGR);

    return img;
}


// expect inputs
// path to an image file (this should be converted to GUID like id to point to image)
// number of clicks/touches (integer) to sample colors
int main(int argc, char* argv[]) {

    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " [integer] [filename]" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    string fName = argv[2];  //"images/test.jpg"

    cv::Mat im = cv::imread(fName);
    cv::Mat clicked_coords_mat = cv::Mat::zeros(n, 2, CV_32S);
    cv::Mat picked_rgb = cv::Mat::zeros(n, 3, CV_32S);
    int counter = 0;

    pick_colors(im, n);

    std::vector<cv::Vec3f> picked_hsv;
    for (int i = 0; i < n; i++) {
        cv::Vec3b rgb = picked_rgb.at<cv::Vec3b>(i);
        cv::Vec3f hsv = rgb_to_hsv(rgb);
        picked_hsv.push_back(hsv);
    }

    float low_h, high_h, low_s, high_s;
    std::vector<float> hs, ss;
    for (auto hsv : picked_hsv) {
        hs.push_back(hsv[0]);
        ss.push_back(hsv[1]);
    }
    low_h = *std::min_element(hs.begin(), hs.end()) - 1;
    high_h = *std::max_element(hs.begin(), hs.end()) + 1;
    low_s = *std::min_element(ss.begin(), ss.end()) - 10;
    high_s = *std::max_element(ss.begin(), ss.end()) + 10;

    std::cout << low_h << " " << high_h << std::endl;
    std::cout << low_s << " " << high_s << std::endl;

    cv::Mat hsv_im;
    cvtColor(im, hsv_im, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv_im, cv::Scalar(low_h, 50, 20), cv::Scalar(high_h, high_s, 255), mask);
    cv::Mat res;
    cv::bitwise_and(im, im, res, mask);

    cv::imshow("Result", res);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}




