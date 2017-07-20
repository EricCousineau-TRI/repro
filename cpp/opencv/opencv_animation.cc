// @ref https://stackoverflow.com/questions/33882812/microsoft-visual-studio-c-opencv-animation
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

int main()
{
    // Yellow image
    Mat3b img = imread("path_to_image");

    int step = 5;   // Color step
    int delay = 30; // Animation speed

    bool forward = true;
    Scalar color(0,255,255);

    Mat1b gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat1b edges;
    Canny(gray, edges, 400, 200);

    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    Mat3b canvas(img.rows, img.cols, Vec3b(0,0,0));

    while (true)
    {
        imshow("Draw", canvas);
        if (waitKey(delay) == 27 /*ESC*/) break;

        // Update color 
        color[0] = color[0] + ((forward) ? step : -step);

        // Deal with direction
        if (forward && color[0] > 255) {
            forward = false;
            color[0] = 255;
        }
        if (!forward && color[0] < 0) {
            forward = true;
            color[0] = 0;
        }

        // Update only edge points 
        //canvas.setTo(color, edges);

        // Draw a thick contour
        drawContours(canvas, contours, -1, color, 2);
    }

    return 0;
}
