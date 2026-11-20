#include "analysis.hpp"

#include "detectors.hpp"
#include "scoring.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>

namespace fs = std::filesystem;

namespace {
double average(const std::vector<double>& v)
{
    return v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

std::vector<double> ranks(const std::vector<double>& v)
{
    std::vector<size_t> order(v.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });
    std::vector<double> r(v.size());
    for (size_t i = 0; i < order.size();) {
        size_t j = i;
        while (j + 1 < order.size() && v[order[j + 1]] == v[order[i]])
            ++j;
        for (size_t k = i; k <= j; ++k)
            r[order[k]] = (i + j) / 2.0 + 1;
        i = j + 1;
    }
    return r;
}

double pearson(const std::vector<double>& a, const std::vector<double>& b)
{
    const double ma = average(a), mb = average(b);
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
    }
    return sab / std::sqrt(saa * sbb);
}

// correlation de Spearman = Pearson sur les rangs
double spearman(const std::vector<double>& a, const std::vector<double>& b)
{
    return pearson(ranks(a), ranks(b));
}

// fonction beta incomplete (Numerical Recipes), pour la p-value
double betaContinuedFraction(double a, double b, double x)
{
    const double tiny = 1e-30;
    double c = 1, d = 1 - (a + b) * x / (a + 1);
    d = 1 / (std::abs(d) < tiny ? tiny : d);
    double h = d;
    for (int m = 1; m <= 200; ++m) {
        for (int step = 0; step < 2; ++step) {
            const double num = step == 0 ? m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
                                         : -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1));
            d = 1 + num * d;
            d = 1 / (std::abs(d) < tiny ? tiny : d);
            c = 1 + num / c;
            if (std::abs(c) < tiny)
                c = tiny;
            h *= d * c;
        }
    }
    return h;
}

double incompleteBeta(double a, double b, double x)
{
    if (x <= 0)
        return 0;
    if (x >= 1)
        return 1;
    const double front = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log(1 - x));
    if (x < (a + 1) / (a + b + 2))
        return front * betaContinuedFraction(a, b, x) / a;
    return 1 - front * betaContinuedFraction(b, a, 1 - x) / b;
}

double pValue(double r, int n)
{
    if (std::abs(r) >= 1)
        return 0;
    const double df = n - 2;
    const double t2 = r * r * df / (1 - r * r);
    return incompleteBeta(df / 2, 0.5, df / (df + t2));
}

const char* stars(double p)
{
    return p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "";
}

std::vector<double> geomspace(double lo, double hi, int n)
{
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = lo * std::pow(hi / lo, double(i) / (n - 1));
    return v;
}

double percentile(const std::vector<float>& sorted, double q)
{
    const double pos = q / 100 * (sorted.size() - 1);
    const size_t i = static_cast<size_t>(pos);
    if (i + 1 >= sorted.size())
        return sorted.back();
    return sorted[i] + (pos - i) * (sorted[i + 1] - sorted[i]);
}

// validation croisee : pour chaque image, la constante est choisie sur les autres images
double leaveOneOut(const std::vector<std::vector<double>>& table)
{
    const size_t n = table.size(), m = table[0].size();
    std::vector<double> total(m, 0);
    for (const auto& row : table)
        for (size_t j = 0; j < m; ++j)
            total[j] += row[j];
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t best = 0;
        for (size_t j = 1; j < m; ++j)
            if (total[j] - table[i][j] > total[best] - table[i][best])
                best = j;
        sum += table[i][best];
    }
    return sum / n;
}

size_t bestColumn(const std::vector<std::vector<double>>& table)
{
    size_t best = 0;
    double bestSum = -1;
    for (size_t j = 0; j < table[0].size(); ++j) {
        double s = 0;
        for (const auto& row : table)
            s += row[j];
        if (s > bestSum) {
            bestSum = s;
            best = j;
        }
    }
    return best;
}

struct Response {
    std::string name;
    std::vector<cv::Mat> map;
    std::vector<std::vector<float>> values;
    std::function<cv::Mat(size_t, double)> binarize;
    double lo, hi;
};

std::vector<float> sortedValues(const cv::Mat& m, bool positiveOnly)
{
    std::vector<float> v;
    v.reserve(m.total());
    for (int y = 0; y < m.rows; ++y) {
        const float* row = m.ptr<float>(y);
        for (int x = 0; x < m.cols; ++x)
            if (!positiveOnly || row[x] > 0)
                v.push_back(row[x]);
    }
    std::sort(v.begin(), v.end());
    return v;
}

cv::Mat above(const cv::Mat& m, double t)
{
    cv::Mat mask;
    cv::compare(m, t, mask, cv::CMP_GT);
    return mask;
}

class Log {
public:
    explicit Log(const std::string& path) : file_(path) {}
    void operator()(const char* text)
    {
        std::fputs(text, stdout);
        file_ << text;
    }
    template <typename... A>
    void operator()(const char* fmt, A... args)
    {
        char buf[512];
        std::snprintf(buf, sizeof buf, fmt, args...);
        std::fputs(buf, stdout);
        std::fflush(stdout);
        file_ << buf;
    }
private:
    std::ofstream file_;
};

}

int runAnalysis(const Dataset& data)
{
    const size_t n = data.gray.size();
    if (n < 3)
        return 1;
    fs::create_directories("results/analysis");
    Log log("results/analysis/analysis.txt");
    const cv::Mat square3 = cv::Mat::ones(3, 3, CV_8U), square7 = cv::Mat::ones(7, 7, CV_8U);

    SobelConfig sobelCfg;
    sobelCfg.mode = GLOBAL;
    LaplaceConfig lapCfg;
    CannyConfig cannyCfg;
    cannyCfg.mode = GLOBAL;

    Response sob{"Sobel", {}, {}, nullptr, 4, 400};
    Response lap{"Laplace", {}, {}, nullptr, 1, 400};
    Response can{"Canny", {}, {}, nullptr, 20, 500};
    std::vector<cv::Mat> cannyInput(n);
    for (size_t i = 0; i < n; ++i) {
        sob.map.push_back(edges::sobelResponse(data.gray[i], sobelCfg));
        lap.map.push_back(edges::laplaceSlopes(data.gray[i], lapCfg));
        cv::Mat dx, dy, ax, ay;
        edges::cannyGradients(data.gray[i], cannyCfg, dx, dy);
        cv::Mat(cv::abs(dx)).convertTo(ax, CV_32F);
        cv::Mat(cv::abs(dy)).convertTo(ay, CV_32F);
        can.map.push_back(ax + ay);
        cv::GaussianBlur(data.gray[i], cannyInput[i], cv::Size(cannyCfg.smooth, cannyCfg.smooth), 0);
        sob.values.push_back(sortedValues(sob.map[i], false));
        lap.values.push_back(sortedValues(lap.map[i], true));
        can.values.push_back(sortedValues(can.map[i], false));
    }
    sob.binarize = [&](size_t i, double t) { return above(sob.map[i], t); };
    lap.binarize = [&](size_t i, double t) { return above(lap.map[i], t); };
    can.binarize = [&](size_t i, double t) {
        cv::Mat out;
        cv::Canny(cannyInput[i], out, t / 2, t, cannyCfg.aperture);
        return out;
    };
    Response* detectors[3] = {&sob, &lap, &can};
    auto P = [&](const Response& d, size_t i, double t) { return compare(d.binarize(i, t), data.truth[i]).perf; };

    // 1. caracteristiques des images
    log("=== 1. Image features and performance P (manual default settings) ===\n");
    std::vector<double> contours(n), texture(n), ratio(n), density(n), contrast(n);
    std::vector<double> manualP[3];
    std::ofstream featCsv("results/analysis/features.csv");
    featCsv << "image;Gc_contours;T_texture;R;densite;contraste;P_sobel;P_laplace;P_canny\n";
    for (size_t i = 0; i < n; ++i) {
        const cv::Mat& m = sob.map[i];
        cv::Mat localMax, nearContour;
        cv::dilate(m, localMax, square3);
        cv::dilate(data.truth[i], nearContour, square7);
        const cv::Mat farFromContour = nearContour == 0;
        contours[i] = cv::mean(localMax, data.truth[i])[0];
        texture[i] = cv::mean(m, farFromContour)[0];
        ratio[i] = contours[i] / texture[i];
        density[i] = 100.0 * cv::countNonZero(data.truth[i]) / data.truth[i].total();
        cv::Scalar mu, sd;
        cv::meanStdDev(data.gray[i], mu, sd);
        contrast[i] = sd[0];
        manualP[0].push_back(compare(edges::sobel(data.gray[i], SobelConfig()), data.truth[i]).perf);
        manualP[1].push_back(compare(edges::laplace(data.gray[i], LaplaceConfig()), data.truth[i]).perf);
        manualP[2].push_back(compare(edges::canny(data.gray[i], CannyConfig()), data.truth[i]).perf);
        char row[256];
        std::snprintf(row, sizeof row, "%s;%.2f;%.2f;%.3f;%.3f;%.2f;%.4f;%.4f;%.4f\n", data.names[i].c_str(), contours[i],
                      texture[i], ratio[i], density[i], contrast[i], manualP[0][i], manualP[1][i], manualP[2][i]);
        featCsv << row;
    }
    const std::pair<const char*, std::vector<double>*> features[5] = {
        {"Gc contours", &contours}, {"T texture", &texture}, {"R = Gc/T", &ratio}, {"density %", &density}, {"contrast", &contrast}};
    for (const auto& f : features) {
        const auto& v = *f.second;
        log("%-12s min %7.2f  max %7.2f  |", f.first, *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()));
        for (int d = 0; d < 3; ++d) {
            const double rho = spearman(v, manualP[d]), p = pValue(rho, static_cast<int>(n));
            log("  %s rho %+.2f (p %.3f)%-3s", detectors[d]->name.c_str(), rho, p, stars(p));
        }
        log("\n");
    }
    std::vector<double> sortedR = ratio;
    std::sort(sortedR.begin(), sortedR.end());
    const double cut1 = sortedR[(n - 1) / 3], cut2 = sortedR[2 * (n - 1) / 3];
    const char* groupNames[3] = {"strong texture (low R)", "medium R", "weak texture (high R)"};
    for (int g = 0; g < 3; ++g) {
        log("%-24s", groupNames[g]);
        for (int d = 0; d < 3; ++d) {
            std::vector<double> sel;
            for (size_t i = 0; i < n; ++i) {
                const int group = ratio[i] <= cut1 ? 0 : ratio[i] <= cut2 ? 1 : 2;
                if (group == g)
                    sel.push_back(manualP[d][i]);
            }
            log("  %s P %.1f %%", detectors[d]->name.c_str(), 100 * average(sel));
        }
        log("\n");
    }

    // 2-3. meilleur seuil par image et regles automatiques
    log("\n=== 2-3. Best threshold per image and automatic rules (mean P, leave-one-out) ===\n");
    std::ofstream thrCsv("results/analysis/best_thresholds.csv");
    thrCsv << "image;sobel_best;sobel_mean;laplace_best;laplace_mean;canny_best;canny_mean\n";
    std::vector<std::vector<double>> bestThr(3), meanStat(3);
    for (int d = 0; d < 3; ++d) {
        Response& det = *detectors[d];
        std::printf("  %s...\n", det.name.c_str());
        std::vector<double> sMax(n), sMean(n), sMed(n), ones(n, 1.0);
        for (size_t i = 0; i < n; ++i) {
            const auto& v = det.values[i];
            sMax[i] = v.back();
            sMean[i] = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            sMed[i] = percentile(v, 50);
        }
        meanStat[d] = sMean;

        auto ruleTable = [&](const std::vector<double>& stat) {
            const auto ks = geomspace(det.lo / *std::max_element(stat.begin(), stat.end()),
                                      det.hi / *std::min_element(stat.begin(), stat.end()), 70);
            std::vector<std::vector<double>> table(n, std::vector<double>(ks.size()));
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < ks.size(); ++j)
                    table[i][j] = P(det, i, ks[j] * stat[i]);
            return table;
        };
        const auto fixedTable = ruleTable(ones);
        const auto grid = geomspace(det.lo, det.hi, 70);
        double oracle = 0;
        for (size_t i = 0; i < n; ++i) {
            const size_t j = std::max_element(fixedTable[i].begin(), fixedTable[i].end()) - fixedTable[i].begin();
            bestThr[d].push_back(grid[j]);
            oracle += fixedTable[i][j];
        }
        oracle /= n;

        std::vector<std::vector<double>> pctTable(n);
        for (size_t i = 0; i < n; ++i)
            for (double q = 50; q <= 99.5; q += 0.5)
                pctTable[i].push_back(P(det, i, percentile(det.values[i], q)));

        std::vector<double> otsu(n);
        for (size_t i = 0; i < n; ++i) {
            const auto& v = det.values[i];
            cv::Mat u;
            cv::Mat(v).convertTo(u, CV_8U, 255.0 / v.back());
            cv::Mat ignored;
            const double t8 = cv::threshold(u, ignored, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            otsu[i] = P(det, i, t8 / 255.0 * v.back());
        }

        const double rhoThr = spearman(bestThr[d], sMean);
        log("%s: best threshold from %.0f to %.0f, correlation with the mean gradient rho %.2f (p %.4f)\n", det.name.c_str(),
            *std::min_element(bestThr[d].begin(), bestThr[d].end()), *std::max_element(bestThr[d].begin(), bestThr[d].end()),
            rhoThr, pValue(rhoThr, static_cast<int>(n)));
        log("   fixed %.1f | k x max %.1f | k x mean %.1f | k x median %.1f | percentile %.1f | Otsu %.1f | oracle %.1f\n",
            100 * leaveOneOut(fixedTable), 100 * leaveOneOut(ruleTable(sMax)), 100 * leaveOneOut(ruleTable(sMean)),
            100 * leaveOneOut(ruleTable(sMed)), 100 * leaveOneOut(pctTable), 100 * average(otsu), 100 * oracle);
    }
    for (size_t i = 0; i < n; ++i) {
        thrCsv << data.names[i];
        for (int d = 0; d < 3; ++d)
            thrCsv << ';' << bestThr[d][i] << ';' << meanStat[d][i];
        thrCsv << '\n';
    }

    // 4. seuil local : recherche de alpha, sigma, k
    log("\n=== 4. Local threshold: response - alpha x Gaussian mean (sigma) of the neighbourhood, threshold = k x mean ===\n");
    const double alphas[5] = {0, 0.5, 1, 1.5, 2};
    const double sigmas[3] = {2, 4, 8};
    const auto gains = geomspace(1, 12, 26);
    struct Combo { double alpha, sigma, k; };
    std::vector<Combo> combos;
    for (double a : alphas)
        for (double s : sigmas)
            if (a > 0 || s == 2)
                for (double k : gains)
                    combos.push_back({a, s, k});

    for (int d : {0, 2}) {
        std::printf("  %s local...\n", detectors[d]->name.c_str());
        std::vector<std::vector<double>> table(n, std::vector<double>(combos.size()));
        for (size_t i = 0; i < n; ++i) {
            for (size_t c = 0; c < combos.size(); c += gains.size()) {
                if (d == 0) {
                    SobelConfig cfg;
                    cfg.mode = combos[c].alpha > 0 ? LOCAL : GLOBAL;
                    cfg.alpha = combos[c].alpha;
                    cfg.sigma = combos[c].sigma;
                    const cv::Mat r = edges::sobelResponse(data.gray[i], cfg);
                    const double m = cv::mean(r)[0];
                    for (size_t j = 0; j < gains.size(); ++j)
                        table[i][c + j] = compare(above(r, gains[j] * m), data.truth[i]).perf;
                } else {
                    CannyConfig cfg;
                    cfg.mode = combos[c].alpha > 0 ? LOCAL : GLOBAL;
                    cfg.alpha = combos[c].alpha;
                    cfg.sigma = combos[c].sigma;
                    cv::Mat dx, dy;
                    edges::cannyGradients(data.gray[i], cfg, dx, dy);
                    for (size_t j = 0; j < gains.size(); ++j)
                        table[i][c + j] = compare(edges::cannyAuto(dx, dy, gains[j]), data.truth[i]).perf;
                }
            }
        }
        const size_t best = bestColumn(table);
        double all = 0;
        for (const auto& row : table)
            all += row[best];
        log("%s: best alpha %.1f, sigma %.0f, k %.1f -> P %.1f %% (leave-one-out %.1f %%)\n", detectors[d]->name.c_str(),
            combos[best].alpha, combos[best].sigma, combos[best].k, 100 * all / n, 100 * leaveOneOut(table));
    }
    log("\nOutput: results/analysis/analysis.txt, features.csv, best_thresholds.csv\n");
    return 0;
}
