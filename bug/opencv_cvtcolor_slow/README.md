# OpenCV cvtColor

Encountered some C++ code where it was faster to write simple for-loop rather
than use `cvtColor`:

```cpp
// Offers a simpler flavor of `cvtColor(src, dst, CV_RGB2BGR)`.
// I (Eric) dunno why exactly, but this seems to lower the overall CPU load
// (#6341).
void ConvertRgb8ToBgr8(const cv::Mat& src, cv::Mat* pdst) {
  DRAKE_DEMAND(pdst != nullptr);
  DRAKE_DEMAND(src.isContinuous());
  DRAKE_DEMAND(src.type() == CV_8UC3);
  cv::Mat& dst = *pdst;
  dst.create(src.rows, src.cols, src.type());
  const uint8_t* src_addr = src.data;
  uint8_t* dst_addr = dst.data;
  const int step = 3;
  const int image_size = src.rows * src.cols * step;
  for (int i = 0; i < image_size; i += step) {
    dst_addr[i + 2] = src_addr[i + 0];
    dst_addr[i + 1] = src_addr[i + 1];
    dst_addr[i + 0] = src_addr[i + 2];
  }
}
```

Trying to debug w/ Python first.

Relates:

- <https://answers.opencv.org/question/188664/colour-conversion-from-bgr-to-rgb-in-cv2-is-slower-than-on-python/>

## Python

```sh
./setup.sh python ./repro.py
```
