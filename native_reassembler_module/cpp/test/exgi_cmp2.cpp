#include "../util.h"

int main() {
    auto image = cv::imread("../data/w2.png");
    const auto thresh = 50;

    auto r1 = bgr2ExGI_cvfuncs(image, thresh);
    auto r2 = bgr2ExGI_loop(image, thresh);
    auto r3 = bgr2ExGI_simd(image, thresh);

    cv::imwrite("out/t1.png", r1);
    cv::imwrite("out/t2.png", r2);
    cv::imwrite("out/t3.png", r3);

    return 0;
}