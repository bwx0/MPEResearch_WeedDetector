#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <iomanip>

cv::Mat image;
cv::Mat edges;
std::vector<cv::Vec2f> lines;

void test(float max_ang) {
    double rho = 3;             // Distance resolution in pixels
    double theta = CV_PI / 120;  // Angular resolution in radians
    int threshold = 220 * 6;    // Minimum number of votes (intersections in Hough grid cell)

    clock_t start_time = clock();

    int runs = 100;

    for (int i = 0; i < runs; ++i) {
        // Detect lines
        cv::HoughLines(edges, lines, rho, theta, threshold, 0, 0, 0, max_ang);
    }

    clock_t end_time = clock();
    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
    double avg_time = elapsed_time / runs;

    // std::cout << "Average execution time per call: " << avg_time << " seconds" << std::endl;
    std::cout << std::fixed << std::setprecision(8) << max_ang << "\t" << avg_time << std::endl;
}

int main() {
    // Load an image
    image = cv::imread("/home/pi/proj/camtest/data/pisize.png");
    if (image.empty()) {
        std::cerr << "Could not read the image." << std::endl;
        return -1;
    }

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detect edges using Canny
    // cv::Mat edges;
    cv::Canny(gray, edges, 50000, 60000, 7);

    // Define parameters for HoughLines

    // Benchmark HoughLines
    int nseg=20;
    for(int i=1;i<=nseg;i++){
        float theta = CV_PI/nseg*i;
        test(theta);
    }

    // Optionally: draw lines on the image for visualization
    // for (size_t i = 0; i < lines.size(); i++) {
    //     float rho = lines[i][0], theta = lines[i][1];
    //     cv::Point pt1, pt2;
    //     double a = cos(theta), b = sin(theta);
    //     double x0 = a * rho, y0 = b * rho;
    //     pt1.x = cvRound(x0 + 1000*(-b));
    //     pt1.y = cvRound(y0 + 1000*(a));
    //     pt2.x = cvRound(x0 - 1000*(-b));
    //     pt2.y = cvRound(y0 - 1000*(a));
    //     cv::line(image, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA);
    // }

    // Display the resulting image
    // cv::imshow("Detected Lines", image);
    // cv::waitKey(0);

    return 0;
}
