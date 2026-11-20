#include "analysis.hpp"
#include "detectors.hpp"
#include "scoring.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
const std::array<std::string, 3> LABELS = {"Sobel", "Laplace", "Canny"};
const std::array<std::string, 3> SLUGS  = {"sobel", "laplace", "canny"};
const int SHOW_ALL = 3;
const int MAX_WIDTH = 1280;
const char* const VIEW = "Edges";
const char* const PANEL = "Settings";

struct Sample {
    std::string name;
    cv::Mat gray;
    cv::Mat truth;
};

struct Settings {
    SobelConfig sobel;
    LaplaceConfig laplace;
    CannyConfig canny;
};

std::vector<std::string> imageNames()
{
    std::vector<std::string> out;
    for (const auto& entry : fs::directory_iterator("data/images"))
        if (entry.path().extension() == ".pgm")
            out.push_back(entry.path().stem().string());
    std::sort(out.begin(), out.end());
    return out;
}

bool loadSample(const std::string& name, Sample& s)
{
    s.name = name;
    s.gray = cv::imread("data/images/" + name + ".pgm", cv::IMREAD_GRAYSCALE);
    s.truth = readGroundTruth("data/reference/" + name + "_gt_binary.pgm");
    return !s.gray.empty() && !s.truth.empty();
}

std::vector<Sample> loadAll()
{
    std::vector<Sample> all;
    for (const std::string& name : imageNames()) {
        Sample s;
        if (loadSample(name, s))
            all.push_back(s);
        else
            std::cerr << "Skipped " << name << " (image or ground truth unreadable)\n";
    }
    return all;
}

std::array<cv::Mat, 3> runAll(const cv::Mat& gray, const Settings& st)
{
    return {edges::sobel(gray, st.sobel), edges::laplace(gray, st.laplace), edges::canny(gray, st.canny)};
}

std::string scoreLine(const Score& s)
{
    char buf[96];
    std::snprintf(buf, sizeof buf, "P %.1f%%  TFP %.1f%%  TFN %.1f%%", 100 * s.perf, 100 * s.fpRate, 100 * s.fnRate);
    return buf;
}

cv::Mat tile(const cv::Mat& img, const std::string& title, const std::string& caption = "")
{
    const int header = 56;
    cv::Mat out(img.rows + header, img.cols, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::Mat body = out(cv::Rect(0, header, img.cols, img.rows));
    if (img.channels() == 1)
        cv::cvtColor(img, body, cv::COLOR_GRAY2BGR);
    else
        img.copyTo(body);
    cv::putText(out, title, {8, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(out, caption, {8, 46}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(120, 220, 255), 1, cv::LINE_AA);
    return out;
}

const int KERNELS[4] = {1, 3, 5, 7};
const int APERTURES[3] = {3, 5, 7};

int slot(const int* values, int count, int v)
{
    for (int i = 0; i < count; ++i)
        if (values[i] == v)
            return i;
    return 0;
}

int bar(const char* name)
{
    return std::max(0, cv::getTrackbarPos(name, PANEL));
}

void addBar(const char* name, int max, int start)
{
    cv::createTrackbar(name, PANEL, nullptr, max);
    cv::setTrackbarPos(name, PANEL, start);
}

int tenth(double v) { return static_cast<int>(std::lround(10 * v)); }

void buildPanel()
{
    const Settings d;
    cv::namedWindow(PANEL, cv::WINDOW_NORMAL);
    cv::resizeWindow(PANEL, 480, 760);
    addBar("detector", SHOW_ALL, SHOW_ALL);
    addBar("S smooth", 7, (d.sobel.smooth - 1) / 2);
    addBar("S kernel", 3, slot(KERNELS, 4, d.sobel.kernel));
    addBar("S mode", 2, d.sobel.mode);
    addBar("S thresh", 255, d.sobel.threshold);
    addBar("S gain x10", 200, tenth(d.sobel.gain));
    addBar("S alpha x10", 30, tenth(d.sobel.alpha));
    addBar("S sigma", 20, static_cast<int>(d.sobel.sigma));
    addBar("L smooth", 7, (d.laplace.smooth - 1) / 2);
    addBar("L kernel", 3, slot(KERNELS, 4, d.laplace.kernel));
    addBar("L mode", 1, d.laplace.mode);
    addBar("L thresh", 255, d.laplace.threshold);
    addBar("L gain x10", 100, tenth(d.laplace.gain));
    addBar("C smooth", 7, (d.canny.smooth - 1) / 2);
    addBar("C aperture", 2, slot(APERTURES, 3, d.canny.aperture));
    addBar("C mode", 2, d.canny.mode);
    addBar("C low", 1000, d.canny.low);
    addBar("C high", 1000, d.canny.high);
    addBar("C gain x10", 200, tenth(d.canny.gain));
    addBar("C alpha x10", 30, tenth(d.canny.alpha));
    addBar("C sigma", 20, static_cast<int>(d.canny.sigma));
}

Settings readPanel()
{
    Settings st;
    st.sobel.smooth      = 2 * bar("S smooth") + 1;
    st.sobel.kernel      = KERNELS[std::min(bar("S kernel"), 3)];
    st.sobel.mode        = bar("S mode");
    st.sobel.threshold   = bar("S thresh");
    st.sobel.gain        = bar("S gain x10") / 10.0;
    st.sobel.alpha       = bar("S alpha x10") / 10.0;
    st.sobel.sigma       = std::max(1, bar("S sigma"));
    st.laplace.smooth    = 2 * bar("L smooth") + 1;
    st.laplace.kernel    = KERNELS[std::min(bar("L kernel"), 3)];
    st.laplace.mode      = bar("L mode");
    st.laplace.threshold = bar("L thresh");
    st.laplace.gain      = bar("L gain x10") / 10.0;
    st.canny.smooth      = 2 * bar("C smooth") + 1;
    st.canny.aperture    = APERTURES[std::min(bar("C aperture"), 2)];
    st.canny.mode        = bar("C mode");
    st.canny.low         = bar("C low");
    st.canny.high        = bar("C high");
    st.canny.gain        = bar("C gain x10") / 10.0;
    st.canny.alpha       = bar("C alpha x10") / 10.0;
    st.canny.sigma       = std::max(1, bar("C sigma"));
    return st;
}

// quand on change de mode, le gain revient a la valeur reglee pour ce mode
void resetGainsOnModeChange(const Settings& st, std::array<int, 3>& lastModes)
{
    if (st.sobel.mode != lastModes[0])
        cv::setTrackbarPos("S gain x10", PANEL, tenth(SobelConfig::defaultGain(st.sobel.mode)));
    if (st.laplace.mode != lastModes[1])
        cv::setTrackbarPos("L gain x10", PANEL, tenth(LaplaceConfig::defaultGain(st.laplace.mode)));
    if (st.canny.mode != lastModes[2])
        cv::setTrackbarPos("C gain x10", PANEL, tenth(CannyConfig::defaultGain(st.canny.mode)));
    lastModes = {st.sobel.mode, st.laplace.mode, st.canny.mode};
}

std::vector<int> signature(int index, bool overlay, int which, const Settings& st)
{
    return {index, overlay, which,
            st.sobel.smooth, st.sobel.kernel, st.sobel.mode, st.sobel.threshold, tenth(st.sobel.gain),
            tenth(st.sobel.alpha), tenth(st.sobel.sigma),
            st.laplace.smooth, st.laplace.kernel, st.laplace.mode, st.laplace.threshold, tenth(st.laplace.gain),
            st.canny.smooth, st.canny.aperture, st.canny.mode, st.canny.low, st.canny.high, tenth(st.canny.gain),
            tenth(st.canny.alpha), tenth(st.canny.sigma)};
}

const char* const MODE_NAMES[3] = {"manual", "auto global", "auto local"};

// mode interactif
int interactive()
{
    const std::vector<std::string> names = imageNames();
    if (names.empty()) {
        std::cerr << "No .pgm file in data/images\n";
        return 1;
    }

    cv::namedWindow(VIEW, cv::WINDOW_AUTOSIZE);
    buildPanel();
    std::cout << "detector trackbar: 0 Sobel, 1 Laplace, 2 Canny, 3 all\n"
                 "Keys in the Edges window: d / a next / previous image, o toggle overlay,\n"
                 "                          w write the current views to results/snapshots/, Esc quit\n";

    int index = 0;
    bool overlay = true;
    std::vector<int> shown;
    Sample current;
    std::array<cv::Mat, 3> views;
    std::array<int, 3> lastModes = {MANUAL, MANUAL, MANUAL};

    while (true) {
        const int which = bar("detector");
        resetGainsOnModeChange(readPanel(), lastModes);
        const Settings st = readPanel();
        const std::vector<int> sig = signature(index, overlay, which, st);

        if (sig != shown) {
            shown = sig;
            if (!loadSample(names[index], current)) {
                std::cerr << "Cannot read " << names[index] << " or its ground truth\n";
                return 1;
            }
            const std::array<cv::Mat, 3> maps = runAll(current.gray, st);
            const std::string where = "  " + current.name + " [" + std::to_string(index + 1) + "/" +
                                      std::to_string(names.size()) + "]";
            const int modes[3] = {st.sobel.mode, std::min(st.laplace.mode, 1), st.canny.mode};
            std::vector<cv::Mat> tiles;
            for (int d = 0; d < 3; ++d) {
                views[d] = overlay ? colorize(current.gray, maps[d], current.truth) : maps[d];
                if (which == d || which == SHOW_ALL)
                    tiles.push_back(tile(views[d], LABELS[d] + " (" + MODE_NAMES[modes[d]] + ")" + where,
                                         scoreLine(compare(maps[d], current.truth))));
            }
            cv::Mat canvas;
            cv::hconcat(tiles, canvas);
            if (canvas.cols > MAX_WIDTH) {
                const double f = double(MAX_WIDTH) / canvas.cols;
                cv::resize(canvas, canvas, cv::Size(), f, f, cv::INTER_AREA);
            }
            cv::imshow(VIEW, canvas);
        }

        const int key = cv::waitKey(30) & 0xFF;
        if (key == 27 || cv::getWindowProperty(VIEW, cv::WND_PROP_VISIBLE) < 1)
            break;
        const int n = static_cast<int>(names.size());
        if (key == 'd')
            index = (index + 1) % n;
        else if (key == 'a')
            index = (index + n - 1) % n;
        else if (key == 'o')
            overlay = !overlay;
        else if (key == 'w') {
            fs::create_directories("results/snapshots");
            const std::string stem = "results/snapshots/" + current.name;
            cv::imwrite(stem + "_input.png", current.gray);
            cv::imwrite(stem + "_truth.png", current.truth);
            for (int d = 0; d < 3; ++d)
                cv::imwrite(stem + "_" + SLUGS[d] + ".png", views[d]);
            std::cout << "Written " << stem << "_*.png\n";
        }
    }
    cv::destroyAllWindows();
    return 0;
}

// traitement par lot
Settings settingsFor(int mode)
{
    Settings st;
    st.sobel.mode = mode;
    st.sobel.gain = SobelConfig::defaultGain(mode);
    st.laplace.mode = std::min(mode, 1);
    st.laplace.gain = LaplaceConfig::defaultGain(mode);
    st.canny.mode = mode;
    st.canny.gain = CannyConfig::defaultGain(mode);
    return st;
}

const char* const MODE_SLUGS[3] = {"manual", "global", "local"};

int batch(int mode)
{
    const std::string out = std::string("results/") + MODE_SLUGS[mode] + "/";
    const std::vector<Sample> samples = loadAll();
    if (samples.empty()) {
        std::cerr << "No usable image in data/\n";
        return 1;
    }
    for (const std::string& s : SLUGS)
        fs::create_directories(out + "maps/" + s);
    fs::create_directories(out + "overlays");

    std::ofstream table(out + "scores.csv");
    table << "image";
    for (const std::string& s : SLUGS)
        for (const char* col : {"detectes", "reference", "corrects", "faux_positifs", "faux_negatifs", "P", "TFP", "TFN"})
            table << ';' << s << '_' << col;
    table << '\n';

    const Settings st = settingsFor(mode);
    std::array<Score, 3> total{};
    std::array<std::array<double, 3>, 3> rates{};

    for (const Sample& smp : samples) {
        const std::array<cv::Mat, 3> maps = runAll(smp.gray, st);
        std::vector<cv::Mat> tiles = {tile(smp.gray, smp.name), tile(smp.truth, "ground truth")};
        std::array<double, 3> perf{};
        table << smp.name;
        for (int d = 0; d < 3; ++d) {
            const Score s = compare(maps[d], smp.truth);
            perf[d] = s.perf;
            cv::imwrite(out + "maps/" + SLUGS[d] + "/" + smp.name + ".png", maps[d]);
            tiles.push_back(tile(colorize(smp.gray, maps[d], smp.truth), LABELS[d], scoreLine(s)));

            char row[160];
            std::snprintf(row, sizeof row, ";%d;%d;%d;%d;%d;%.4f;%.4f;%.4f", s.found, s.truth, s.hits,
                          s.falsePos, s.falseNeg, s.perf, s.fpRate, s.fnRate);
            table << row;

            total[d].found += s.found;
            total[d].truth += s.truth;
            total[d].hits += s.hits;
            total[d].falsePos += s.falsePos;
            total[d].falseNeg += s.falseNeg;
            rates[d][0] += s.perf;
            rates[d][1] += s.fpRate;
            rates[d][2] += s.fnRate;
        }
        table << '\n';

        cv::Mat strip;
        cv::hconcat(tiles, strip);
        cv::imwrite(out + "overlays/" + smp.name + ".png", strip);
        std::printf("%-12s P  Sobel %5.1f%%  Laplace %5.1f%%  Canny %5.1f%%\n", smp.name.c_str(),
                    100 * perf[0], 100 * perf[1], 100 * perf[2]);
    }

    const double n = static_cast<double>(samples.size());
    table << "moyenne";
    for (int d = 0; d < 3; ++d) {
        char row[160];
        std::snprintf(row, sizeof row, ";%.1f;%.1f;%.1f;%.1f;%.1f;%.4f;%.4f;%.4f", total[d].found / n,
                      total[d].truth / n, total[d].hits / n, total[d].falsePos / n, total[d].falseNeg / n,
                      rates[d][0] / n, rates[d][1] / n, rates[d][2] / n);
        table << row;
    }
    table << '\n';

    std::printf("\nMean over %zu images      P       TFP      TFN\n", samples.size());
    for (int d = 0; d < 3; ++d)
        std::printf("%-20s %7.2f%% %7.2f%% %7.2f%%\n", LABELS[d].c_str(), 100 * rates[d][0] / n,
                    100 * rates[d][1] / n, 100 * rates[d][2] / n);
    std::printf("Thresholds: %s (Laplace has no local mode, it uses the global one)\n", MODE_NAMES[mode]);
    std::printf("Output: %sscores.csv, %smaps/, %soverlays/\n", out.c_str(), out.c_str(), out.c_str());
    return 0;
}

template <typename Detect>
void sweepOne(const std::vector<Sample>& samples, const std::string& file, const std::string& param,
              const std::vector<int>& values, Detect detect)
{
    std::ofstream out("results/sweep/" + file + ".csv");
    out << param << ";P;TFP;TFN\n";
    for (int v : values) {
        double p = 0, fp = 0, fn = 0;
        for (const Sample& smp : samples) {
            const Score s = compare(detect(smp.gray, v), smp.truth);
            p += s.perf;
            fp += s.fpRate;
            fn += s.fnRate;
        }
        const double n = static_cast<double>(samples.size());
        char row[96];
        std::snprintf(row, sizeof row, "%d;%.4f;%.4f;%.4f\n", v, p / n, fp / n, fn / n);
        out << row;
        std::printf("%-8s %s=%-4d P %.1f%%\n", file.c_str(), param.c_str(), v, 100 * p / n);
    }
}

std::vector<int> range(int from, int to, int step)
{
    std::vector<int> v;
    for (int x = from; x <= to; x += step)
        v.push_back(x);
    return v;
}

// balayage des parametres
int sweep()
{
    const std::vector<Sample> samples = loadAll();
    if (samples.empty())
        return 1;
    fs::create_directories("results/sweep");

    sweepOne(samples, "sobel", "seuil", range(10, 250, 10), [](const cv::Mat& g, int v) {
        SobelConfig c;
        c.threshold = v;
        return edges::sobel(g, c);
    });
    sweepOne(samples, "laplace", "seuil", range(5, 150, 5), [](const cv::Mat& g, int v) {
        LaplaceConfig c;
        c.threshold = v;
        return edges::laplace(g, c);
    });
    sweepOne(samples, "canny", "seuil_haut", range(40, 400, 20), [](const cv::Mat& g, int v) {
        CannyConfig c;
        c.high = v;
        c.low = v / 2;
        return edges::canny(g, c);
    });
    sweepOne(samples, "canny_flou", "taille_flou", range(1, 15, 2), [](const cv::Mat& g, int v) {
        CannyConfig c;
        c.smooth = v;
        return edges::canny(g, c);
    });
    std::printf("Output: results/sweep/*.csv\n");
    return 0;
}

}

int main(int argc, char** argv)
{
    std::error_code ec;
    if (!fs::exists("data/images", ec))
        fs::current_path(PROJECT_DIR, ec);

    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode.empty())
        return interactive();
    if (mode == "batch") {
        const std::string which = argc > 2 ? argv[2] : "manual";
        for (int m = 0; m < 3; ++m)
            if (which == MODE_SLUGS[m])
                return batch(m);
        if (which == "all") {
            for (int m = 0; m < 3; ++m)
                if (batch(m) != 0)
                    return 1;
            return 0;
        }
    }
    if (mode == "sweep")
        return sweep();
    if (mode == "analyse") {
        Dataset data;
        for (const Sample& smp : loadAll()) {
            data.names.push_back(smp.name);
            data.gray.push_back(smp.gray);
            data.truth.push_back(smp.truth);
        }
        return runAnalysis(data);
    }

    std::cerr << "Usage: edge_eval [batch [manual | global | local | all] | sweep | analyse]\n";
    return 1;
}
