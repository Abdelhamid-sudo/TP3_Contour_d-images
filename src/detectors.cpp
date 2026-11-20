#include "detectors.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace {
cv::Mat gaussian(const cv::Mat& gray, int size)
{
    if (size <= 1)
        return gray;
    cv::Mat out;
    cv::GaussianBlur(gray, out, cv::Size(size, size), 0);
    return out;
}

// seuil manuel : fraction du maximum de l'image
cv::Mat relativeThreshold(const cv::Mat& response, int level)
{
    double peak = 0;
    cv::minMaxLoc(response, nullptr, &peak);
    cv::Mat mask;
    cv::compare(response, peak * level / 255.0, mask, cv::CMP_GT);
    return mask;
}

// seuil local : on retire alpha * la moyenne du gradient autour du pixel
// -> la texture disparait, un contour isole reste
cv::Mat suppressSurround(const cv::Mat& response, double alpha, double sigma)
{
    if (alpha <= 0)
        return response;
    cv::Mat surround;
    cv::GaussianBlur(response, surround, cv::Size(0, 0), sigma);
    cv::Mat out = response - alpha * surround;
    return cv::max(out, 0.0);
}

// passages par zero du laplacien (valeur = pente, sur le pixel le plus proche de 0)
cv::Mat zeroCrossings(const cv::Mat& lap)
{
    cv::Mat slope = cv::Mat::zeros(lap.size(), CV_32F);
    for (int y = 0; y < lap.rows; ++y) {
        const float* row = lap.ptr<float>(y);
        const float* below = y + 1 < lap.rows ? lap.ptr<float>(y + 1) : nullptr;
        for (int x = 0; x < lap.cols; ++x) {
            const float a = row[x];
            const float* nb[2] = {x + 1 < lap.cols ? &row[x + 1] : nullptr, below ? &below[x] : nullptr};
            const cv::Point pos[2] = {{x + 1, y}, {x, y + 1}};
            for (int k = 0; k < 2; ++k) {
                if (!nb[k])
                    continue;
                const float b = *nb[k];
                if (a * b >= 0)
                    continue;
                const float s = std::abs(a - b);
                float& cell = std::abs(a) <= std::abs(b) ? slope.at<float>(y, x) : slope.at<float>(pos[k]);
                cell = std::max(cell, s);
            }
        }
    }
    return slope;
}

}

namespace edges {
cv::Mat sobelResponse(const cv::Mat& gray, const SobelConfig& cfg)
{
    const cv::Mat src = gaussian(gray, cfg.smooth);
    cv::Mat dx, dy, norm;
    cv::Sobel(src, dx, CV_32F, 1, 0, cfg.kernel);
    cv::Sobel(src, dy, CV_32F, 0, 1, cfg.kernel);
    cv::magnitude(dx, dy, norm);
    return cfg.mode == LOCAL ? suppressSurround(norm, cfg.alpha, cfg.sigma) : norm;
}

cv::Mat sobel(const cv::Mat& gray, const SobelConfig& cfg)
{
    const cv::Mat response = sobelResponse(gray, cfg);
    if (cfg.mode == MANUAL)
        return relativeThreshold(response, cfg.threshold);
    cv::Mat mask;
    cv::compare(response, cfg.gain * cv::mean(response)[0], mask, cv::CMP_GT);
    return mask;
}

cv::Mat laplaceSlopes(const cv::Mat& gray, const LaplaceConfig& cfg)
{
    cv::Mat lap;
    cv::Laplacian(gaussian(gray, cfg.smooth), lap, CV_32F, cfg.kernel);
    return zeroCrossings(lap);
}

cv::Mat laplace(const cv::Mat& gray, const LaplaceConfig& cfg)
{
    cv::Mat lap;
    cv::Laplacian(gaussian(gray, cfg.smooth), lap, CV_32F, cfg.kernel);
    const cv::Mat slope = zeroCrossings(lap);

    double limit = 0;
    if (cfg.mode == MANUAL) {
        double peak = 0;
        cv::minMaxLoc(cv::abs(lap), nullptr, &peak);
        limit = peak * cfg.threshold / 255.0;
    } else {
        const cv::Mat crossings = slope > 0;
        limit = cfg.gain * cv::mean(slope, crossings)[0];  // moyenne sur les passages par zero
    }
    cv::Mat mask;
    cv::compare(slope, limit, mask, cv::CMP_GT);
    return mask;
}

// meme gradient que celui calcule dans cv::Canny
void cannyGradients(const cv::Mat& gray, const CannyConfig& cfg, cv::Mat& dx, cv::Mat& dy)
{
    const cv::Mat src = gaussian(gray, cfg.smooth);
    cv::Sobel(src, dx, CV_16S, 1, 0, cfg.aperture);
    cv::Sobel(src, dy, CV_16S, 0, 1, cfg.aperture);
    if (cfg.mode != LOCAL)
        return;

    // on garde la direction, on diminue seulement la norme
    cv::Mat fx, fy, norm, factor;
    dx.convertTo(fx, CV_32F);
    dy.convertTo(fy, CV_32F);
    cv::magnitude(fx, fy, norm);
    cv::divide(suppressSurround(norm, cfg.alpha, cfg.sigma), norm + 1e-6, factor);
    cv::multiply(fx, factor, fx);
    cv::multiply(fy, factor, fy);
    fx.convertTo(dx, CV_16S);
    fy.convertTo(dy, CV_16S);
}

// seuil haut = gain * moyenne(|gx| + |gy|), seuil bas = haut / 2
cv::Mat cannyAuto(const cv::Mat& dx, const cv::Mat& dy, double gain)
{
    const double high = gain * (cv::mean(cv::abs(dx))[0] + cv::mean(cv::abs(dy))[0]);
    cv::Mat out;
    cv::Canny(dx, dy, out, high / 2, high);
    return out;
}

cv::Mat canny(const cv::Mat& gray, const CannyConfig& cfg)
{
    if (cfg.mode == MANUAL) {
        cv::Mat out;
        cv::Canny(gaussian(gray, cfg.smooth), out, cfg.low, cfg.high, cfg.aperture);
        return out;
    }
    cv::Mat dx, dy;
    cannyGradients(gray, cfg, dx, dy);
    return cannyAuto(dx, dy, cfg.gain);
}

}
