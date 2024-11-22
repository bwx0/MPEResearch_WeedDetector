#include "reassembler.h"


int main() {
    printf("hello!\n");
    auto img = cv::imread("../data/w1.png");
    cv::imshow("aa", img);
    //cv::waitKey(100000);
    Reassembler ra;
    ExGIGreenExtractor extractor;
    cv::Mat rimg;
    ra.reassemble(img, rimg, 640, RectSortingMethod::HEIGHT_DESC, true, 1, 8, &extractor);
    cv::imshow("result", rimg);
    cv::waitKey(100000);
    return 0;
}