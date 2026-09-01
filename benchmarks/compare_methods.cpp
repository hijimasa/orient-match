// Compare OrientMatch against other rotation-capable template matchers.
//
// Protocol, per case:
//   - take a Kodak image, grayscale, landscape;
//   - crop a CROP_W x CROP_H window from its centre after rotating the source by theta;
//   - take the template from the theta = 0 crop, at a fixed offset from its centre;
//   - the template therefore appears in the scene at angle theta, at a known position;
//   - optionally degrade the scene (noise, occlusion, illumination, JPEG, scale);
//   - run every method on exactly the same scene and template.
//
// A case counts as a success when the angle is within 5 degrees and the centre within
// 5 pixels, the rule used by the evaluation this protocol comes from.
//
// Output is one CSV row per (case, method) on stdout; summarize.py turns it into tables.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "orient_match/orient_match.hpp"
#include "baselines.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double angleError(double a, double b) {
    const double d = std::fmod(std::abs(a - b), 360.0);
    return std::min(d, 360.0 - d);
}

/** Rotate the source about its centre, then cut a fixed window out of the middle. */
cv::Mat rotatedCrop(const cv::Mat &src, double theta, int w, int h) {
    const cv::Mat m = cv::getRotationMatrix2D(
        cv::Point2f((src.cols - 1) / 2.0F, (src.rows - 1) / 2.0F), theta, 1.0);
    cv::Mat rotated;
    cv::warpAffine(src, rotated, m, src.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    return rotated({(src.cols - w) / 2, (src.rows - h) / 2, w, h}).clone();
}

/** Grey out an axis-aligned band covering `target` of the rotated template's area. */
void occludeByCoverage(cv::Mat &image, double cx, double cy, int t, double theta,
                       double target) {
    const double r = theta * CV_PI / 180.0, c = std::cos(r), s = std::sin(r);
    const double hw = t / 2.0;
    const double corner[4][2] = {{-hw, -hw}, {hw, -hw}, {hw, hw}, {-hw, hw}};
    std::vector<cv::Point2f> quad(4);
    for (int k = 0; k < 4; ++k) {
        const double u = corner[k][0], v = corner[k][1];
        quad[k] = cv::Point2f((float)(cx + u * c + v * s), (float)(cy - u * s + v * c));
    }
    const double width = t * std::sqrt(2.0) + 4.0;
    double lo = 0.0, hi = width;
    for (int i = 0; i < 50; ++i) {
        const double height = 0.5 * (lo + hi);
        const std::vector<cv::Point2f> rect = {
            {(float)(cx - width / 2), (float)(cy - height / 2)},
            {(float)(cx + width / 2), (float)(cy - height / 2)},
            {(float)(cx + width / 2), (float)(cy + height / 2)},
            {(float)(cx - width / 2), (float)(cy + height / 2)}};
        std::vector<cv::Point2f> hull;
        if (cv::intersectConvexConvex(quad, rect, hull, true) / (double)(t * t) < target) {
            lo = height;
        } else {
            hi = height;
        }
    }
    const double height = 0.5 * (lo + hi);
    cv::Rect box(cvRound(cx - width / 2), cvRound(cy - height / 2), cvRound(width),
                 cvRound(height));
    box &= cv::Rect(0, 0, image.cols, image.rows);
    if (box.area() > 0) image(box).setTo(cv::Scalar(128));
}

void degrade(cv::Mat &image, const std::string &kind, double param, cv::RNG &rng) {
    if (kind == "noise") {
        cv::Mat n(image.size(), CV_32F), f;
        rng.fill(n, cv::RNG::NORMAL, 0, param);
        image.convertTo(f, CV_32F);
        f += n;
        f.convertTo(image, CV_8U);
    } else if (kind == "illum") {
        cv::Mat f;
        image.convertTo(f, CV_32F);
        f = f * param + (param < 1.0 ? 40.0 : -40.0);
        f.convertTo(image, CV_8U);
    } else if (kind == "jpeg") {
        std::vector<uchar> buf;
        cv::imencode(".jpg", image, buf, {cv::IMWRITE_JPEG_QUALITY, (int)param});
        image = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    } else if (kind == "scale") {
        cv::Mat s;
        cv::resize(image, s, cv::Size(), param, param, cv::INTER_LINEAR);
        cv::Mat out = cv::Mat::zeros(image.size(), CV_8U);
        const int w = std::min(image.cols, s.cols), h = std::min(image.rows, s.rows);
        s({(s.cols - w) / 2, (s.rows - h) / 2, w, h})
            .copyTo(out({(image.cols - w) / 2, (image.rows - h) / 2, w, h}));
        image = out;
    }
}

struct Condition {
    const char *name;
    const char *kind;
    double param;
    bool negative;
};

/** OrientMatch behind the same prepare/match split as the baselines. */
struct PreparedOrientMatch : bench::Prepared {
    orient_match::Matcher matcher;
    // The same search budget as the coarse-to-fine NCC control: 0.5x coarse level,
    // a 3 degree bank, 1 degree refinement, 5 candidates.
    explicit PreparedOrientMatch(const cv::Mat &templ)
        : matcher(templ, orient_match::MatcherOptions{}) {}

    bench::Res match(const cv::Mat &image) override {
        const orient_match::MatchResult r = matcher.match(image);
        if (r.status != orient_match::MatchStatus::ok) return {};
        return {r.angle_deg, r.center.x - (image.cols - 1) / 2.0,
                r.center.y - (image.rows - 1) / 2.0, r.score, true};
    }
};

std::unique_ptr<bench::Prepared> prepare(const std::string &method, const cv::Mat &templ) {
    if (method == "orient_match") return std::make_unique<PreparedOrientMatch>(templ);
    if (method == "bfncc") return std::make_unique<bench::PreparedBruteForce>(templ, 1);
    if (method == "bfncc_pyr")
        return std::make_unique<bench::PreparedCoarseToFine>(templ, 0.5, 3, 5);
    if (method == "fmt") return std::make_unique<bench::PreparedFourierMellin>(templ);
    if (method == "orb") return std::make_unique<bench::PreparedFeatures>(templ, false);
    if (method == "sift") return std::make_unique<bench::PreparedFeatures>(templ, true);
    return nullptr;
}

}  // namespace

int main(int argc, char **argv) {
    const int n_images = argc > 1 ? std::atoi(argv[1]) : 24;
    const int t_size = argc > 2 ? std::atoi(argv[2]) : 96;
    const std::string only = argc > 3 ? argv[3] : "";
    const int CROP_W = 384, CROP_H = 256;

    const std::vector<std::string> methods = {"orient_match", "bfncc_pyr", "bfncc",
                                              "fmt", "orb", "sift"};
    // Angles with a fractional part, so the truth never sits exactly on the 1 degree
    // refinement grid; otherwise the angle error measures the grid, not the method.
    const std::vector<double> angles = {0.0, 37.25, 78.5, 124.75, 163.0, 201.25,
                                        248.5, 292.75, 331.0};
    const std::vector<cv::Point> offsets = {{30, -20}, {0, 0}, {-50, 25}};
    const std::vector<Condition> conditions = {
        {"clean", "", 0, false},
        {"noise25", "noise", 25, false},
        {"noise50", "noise", 50, false},
        {"occlusion25", "occlusion", 0.25, false},
        {"occlusion50", "occlusion", 0.50, false},
        {"illum0.5", "illum", 0.5, false},
        {"jpeg20", "jpeg", 20, false},
        {"scale0.95", "scale", 0.95, false},
        {"scale1.05", "scale", 1.05, false},
        {"negative", "", 0, true},
        {"negative_noise25", "noise", 25, true},
    };

    std::printf("condition,image,tsize,offx,offy,true_angle,method,"
                "det_angle,angle_err,pos_err,score,success,ms_total,ms_frame\n");

    std::vector<cv::Mat> sources(n_images + 1);
    for (int i = 1; i <= n_images; ++i) {
        char path[256];
        std::snprintf(path, sizeof path, "datasets/kodak/kodim%02d.png", i);
        cv::Mat src = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (src.empty()) {
            std::fprintf(stderr, "missing %s -- run benchmarks/fetch_kodak.sh\n", path);
            return 1;
        }
        if (src.rows > src.cols) cv::transpose(src, src);
        sources[i] = src;
    }

    // Warm up every method once: the first call of a process pays for thread pools and
    // OpenCV's lazily allocated buffers, which would otherwise land on whichever case
    // happened to come first.
    {
        const cv::Mat warm_scene = rotatedCrop(sources[1], 0.0, CROP_W, CROP_H);
        const cv::Mat warm_templ =
            warm_scene({CROP_W / 2 - t_size / 2, CROP_H / 2 - t_size / 2, t_size, t_size})
                .clone();
        for (const std::string &method : methods) {
            if (!only.empty() && only != method) continue;
            prepare(method, warm_templ)->match(warm_scene);
        }
    }

    // The order methods run in within a case is rotated, so that none of them is
    // systematically measured right after the same neighbour. Running them in a fixed
    // order lets one method's memory traffic bias the next one's timing every time.
    std::size_t rotation = 0;

    cv::RNG rng(12345);
    for (int i = 1; i <= n_images; ++i) {
        const cv::Mat base = rotatedCrop(sources[i], 0.0, CROP_W, CROP_H);
        const cv::Mat other =
            rotatedCrop(sources[i % n_images + 1], 0.0, CROP_W, CROP_H);

        for (const Condition &cond : conditions) {
            for (const cv::Point &off : offsets) {
                const cv::Rect roi(CROP_W / 2 + off.x - t_size / 2,
                                   CROP_H / 2 + off.y - t_size / 2, t_size, t_size);
                if ((roi & cv::Rect(0, 0, CROP_W, CROP_H)) != roi) continue;
                // Skip placements whose rotated template would leave the scene.
                if (std::hypot((double)off.x, (double)off.y) + t_size / std::sqrt(2.0) >
                    std::min(CROP_W, CROP_H) / 2.0) {
                    continue;
                }
                const cv::Mat templ = (cond.negative ? other : base)(roi).clone();

                for (const double theta : angles) {
                    cv::Mat scene = rotatedCrop(sources[i], theta, CROP_W, CROP_H);
                    const double r = theta * CV_PI / 180.0;
                    double tdx = off.x * std::cos(r) + off.y * std::sin(r);
                    double tdy = -off.x * std::sin(r) + off.y * std::cos(r);
                    if (std::strcmp(cond.kind, "occlusion") == 0) {
                        occludeByCoverage(scene, (CROP_W - 1) / 2.0 + tdx,
                                          (CROP_H - 1) / 2.0 + tdy, t_size, theta,
                                          cond.param);
                    } else {
                        degrade(scene, cond.kind, cond.param, rng);
                    }
                    if (std::strcmp(cond.kind, "scale") == 0) {
                        tdx *= cond.param;
                        tdy *= cond.param;
                    }

                    for (std::size_t mi = 0; mi < methods.size(); ++mi) {
                        const std::string &method =
                            methods[(mi + rotation) % methods.size()];
                        if (!only.empty() && only != method) continue;
                        // ms_total is the one-shot cost: template preparation plus the
                        // search. ms_frame is the search alone, which is what a stream of
                        // frames against a fixed template actually pays.
                        const auto t0 = Clock::now();
                        std::unique_ptr<bench::Prepared> m = prepare(method, templ);
                        const auto t1 = Clock::now();
                        const bench::Res res = m->match(scene);
                        const auto t2 = Clock::now();
                        const double ms =
                            std::chrono::duration<double, std::milli>(t2 - t0).count();
                        const double ms_frame =
                            std::chrono::duration<double, std::milli>(t2 - t1).count();
                        if (cond.negative) {
                            // No pose to be right about; these rows exist to compare the
                            // score of a present object with that of an absent one.
                            std::printf("%s,kodim%02d,%d,%d,%d,%g,%s,%.3f,-1,-1,%.6g,0,%.2f,%.2f\n",
                                        cond.name, i, t_size, off.x, off.y, theta,
                                        method.c_str(), res.angle, res.score, ms, ms_frame);
                            continue;
                        }
                        const double aerr = res.valid ? angleError(res.angle, theta) : 180.0;
                        const double perr =
                            res.valid ? std::hypot(res.dx - tdx, res.dy - tdy) : 1e9;
                        const int ok = (res.valid && aerr <= 5.0 && perr <= 5.0) ? 1 : 0;
                        std::printf("%s,kodim%02d,%d,%d,%d,%g,%s,%.3f,%.3f,%.3f,%.6g,%d,%.2f,%.2f\n",
                                    cond.name, i, t_size, off.x, off.y, theta,
                                    method.c_str(), res.angle, aerr, perr, res.score, ok,
                                    ms, ms_frame);
                    }
                    ++rotation;
                }
            }
        }
        std::fflush(stdout);
        std::fprintf(stderr, "kodim%02d done\n", i);
    }
    return 0;
}
