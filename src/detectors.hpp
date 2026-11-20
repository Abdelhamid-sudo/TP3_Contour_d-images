#pragma once

#include <opencv2/core.hpp>

// choix du seuil : manuel, auto global (gain * moyenne), auto local (texture retiree)
enum ThresholdMode { MANUAL = 0, GLOBAL = 1, LOCAL = 2 };

struct SobelConfig {
    int smooth    = 3;
    int kernel    = 3;
    int threshold = 70;   // seuil manuel, relatif au max (0-255)
    int mode      = MANUAL;
    double gain   = 3.6;  // seuil auto = gain * moyenne du gradient
    double alpha  = 1.0;
    double sigma  = 4.0;
    static double defaultGain(int m) { return m == LOCAL ? 8.0 : 3.6; }
};

struct LaplaceConfig {
    int smooth    = 9;
    int kernel    = 5;
    int threshold = 30;   // pente minimale au passage par zero
    int mode      = MANUAL;
    double gain   = 2.0;
    static double defaultGain(int) { return 2.0; }
};

struct CannyConfig {
    int smooth   = 9;
    int low      = 60;
    int high     = 120;
    int aperture = 3;
    int mode     = MANUAL;
    double gain  = 4.0;   // seuil haut = gain * moyenne de |gx|+|gy|
    double alpha = 0.5;
    double sigma = 8.0;
    static double defaultGain(int m) { return m == LOCAL ? 5.4 : 4.0; }
};

namespace edges {
cv::Mat sobel(const cv::Mat& gray, const SobelConfig& cfg);
cv::Mat laplace(const cv::Mat& gray, const LaplaceConfig& cfg);
cv::Mat canny(const cv::Mat& gray, const CannyConfig& cfg);

cv::Mat sobelResponse(const cv::Mat& gray, const SobelConfig& cfg);
cv::Mat laplaceSlopes(const cv::Mat& gray, const LaplaceConfig& cfg);
void cannyGradients(const cv::Mat& gray, const CannyConfig& cfg, cv::Mat& dx, cv::Mat& dy);
cv::Mat cannyAuto(const cv::Mat& dx, const cv::Mat& dy, double gain);
}
