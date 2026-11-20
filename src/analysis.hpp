#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct Dataset {
    std::vector<std::string> names;
    std::vector<cv::Mat> gray;
    std::vector<cv::Mat> truth;
};

int runAnalysis(const Dataset& data);
