#include<opencv2/opencv.hpp>
#include "../reassembler.h"


void process(const cv::Mat &frame) {
    cv::TickMeter tm;
    tm.start();
    cv::Mat result;
    Reassembler reassembler;
    reassembler.reassemble(frame, result);
    tm.stop();
    printf("time=%f  sz=%d %d\n", tm.getTimeMilli(), result.size[0], result.size[1]);

    cv::imshow("s", frame);
    cv::imshow("t", result);
}

int main() {
    auto videoFile = R"(D:\projects\data_topdown\d2.MP4)";
    cv::VideoCapture cap(videoFile);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file " << videoFile << "." << std::endl;
        return 0;
    }

    try {
        cv::Mat frame;
        while (true) {
            bool ret = cap.read(frame);

            if (!ret) {
                std::cerr << "Reached end of video or failed to read frame." << std::endl;
                break;
            }
            process(frame);

            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
