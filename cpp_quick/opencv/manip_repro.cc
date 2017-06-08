#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

inline void sleep(double seconds) {
  int ms = std::round(seconds * 1000);
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int main() {
  string window_name = "KinectFrameCostDebug";
  // cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE );
  // cv::startWindowThread();

  cv::Mat image;
  image = cv::imread("../test.png", 1);
  cv::Mat gray_image;
  cv::cvtColor(image, gray_image, CV_BGR2GRAY);
  cv::Mat gray_image_color;
  cv::cvtColor(gray_image, gray_image_color, CV_GRAY2BGR);
  // gray_image.convertTo(gray_image_color, image.type());

  cout
    << "image: " << image.rows << " x " << image.cols << " - " << image.type() << endl
    << "gray: " << gray_image_color.rows << " x " << gray_image_color.cols << " - " << gray_image_color.type() << endl;

  double duration_sec = 5;
  auto start = Clock::now();
  bool show = true;
  double t = 0;
  cv::Mat out;
  while (t < duration_sec) {
    double factor = (cos(t) + 1) / 2;
    // cout << t << endl;
    // show = !show;
    // if (show) {
    //   cv::imshow(window_name, image);
    // } else {
      cout << factor << endl;
      cv::addWeighted(image, factor, gray_image_color, 1 - factor, 0., out);
      cv::imshow(window_name, out);
    // }
    // sleep(0.1);
    cv::waitKey(2);
    // cv::waitKey(33);
    t = Duration(Clock::now() - start).count();
  }

  return 0;
}
