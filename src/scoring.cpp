#include "scoring.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
cv::Mat grow(const cv::Mat& mask)
{
    cv::Mat out;
    cv::dilate(mask, out, cv::Mat::ones(3, 3, CV_8U));
    return out;
}

}

cv::Mat readGroundTruth(const std::string& path)
{
    const cv::Mat file = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (file.empty())
        return file;
    cv::Mat mask;
    cv::threshold(file, mask, 127, 255, cv::THRESH_BINARY_INV);
    return mask;
}

// tolerance 3x3. Les corrects sont comptes cote reference pour avoir FN = ref - corrects
// (sinon un contour epais de Sobel compte plusieurs fois)
Score compare(const cv::Mat& found, const cv::Mat& truth)
{
    Score s;
    s.found    = cv::countNonZero(found);
    s.truth    = cv::countNonZero(truth);
    s.hits     = cv::countNonZero(truth & grow(found));
    s.falseNeg = s.truth - s.hits;
    s.falsePos = cv::countNonZero(found & ~grow(truth));

    const double all = s.hits + s.falsePos + s.falseNeg;
    if (all > 0) {
        s.perf   = s.hits / all;
        s.fpRate = s.falsePos / all;
        s.fnRate = s.falseNeg / all;
    }
    return s;
}

// vert = correct, rouge = faux positif, bleu = manque
cv::Mat colorize(const cv::Mat& gray, const cv::Mat& found, const cv::Mat& truth)
{
    cv::Mat dim, out;
    gray.convertTo(dim, CV_8U, 0.5);
    cv::cvtColor(dim, out, cv::COLOR_GRAY2BGR);

    const cv::Mat truthZone = grow(truth);
    out.setTo(cv::Scalar(255, 0, 0), truth & ~grow(found));
    out.setTo(cv::Scalar(0, 0, 255), found & ~truthZone);
    out.setTo(cv::Scalar(0, 255, 0), found & truthZone);
    return out;
}
