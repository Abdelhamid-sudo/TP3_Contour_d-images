#pragma once

#include <opencv2/core.hpp>

#include <string>

struct Score {
    int found     = 0;  // contours_detectes
    int truth     = 0;  // contours_reference
    int hits      = 0;  // contours_corrects
    int falsePos  = 0;  // faux_positifs
    int falseNeg  = 0;  // faux_negatifs
    double perf   = 0;  // P
    double fpRate = 0;  // TFP
    double fnRate = 0;  // TFN
};

cv::Mat readGroundTruth(const std::string& path);

Score compare(const cv::Mat& found, const cv::Mat& truth);

cv::Mat colorize(const cv::Mat& gray, const cv::Mat& found, const cv::Mat& truth);
