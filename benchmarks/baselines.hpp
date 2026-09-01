// Rotation-capable template matchers to compare OrientMatch against.
//
// These are ordinary, widely used approaches, implemented here so that every method in
// the comparison sees exactly the same images, the same ground truth and the same
// success rule. They are adapted from the evaluation harness of
// https://github.com/hijimasa/radon-template-matching (MIT, same author), with the
// result convention changed to match this repository.
//
// The Radon/sinogram method of that repository is deliberately left out: it is not a
// common approach, and comparing against it would say little about how OrientMatch
// stands next to what people actually use.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace bench {

/** One detection. Position is the template centre relative to the image centre. */
struct Res {
    double angle = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    double score = 0.0;
    bool valid = false;
};

constexpr double kSentinel = -1.99;  // score of a correlation map with no valid entry

inline double wrap360(double a) {
    a = std::fmod(a, 360.0);
    return a < 0.0 ? a + 360.0 : a;
}

inline void sanitize(cv::Mat &res, double sentinel) {
    cv::patchNaNs(res, static_cast<float>(sentinel));
    res.setTo(sentinel, (res < -1.001F) | (res > 1.001F));
}

/** Template placed in the centre of its circumscribed square, with a validity mask. */
struct RotCanvas {
    cv::Mat image;
    cv::Mat mask;
    int d = 0;
};

inline RotCanvas makeRotCanvas(const cv::Mat &templ) {
    RotCanvas rc;
    int d = static_cast<int>(std::ceil(std::hypot((double)templ.rows, (double)templ.cols)));
    if (d % 2) ++d;
    rc.d = d;
    rc.image = cv::Mat::zeros(d, d, CV_8U);
    rc.mask = cv::Mat::zeros(d, d, CV_8U);
    const cv::Rect at((d - templ.cols) / 2, (d - templ.rows) / 2, templ.cols, templ.rows);
    templ.copyTo(rc.image(at));
    rc.mask(at).setTo(255);
    return rc;
}

/** Peak of a masked, zero-mean normalized correlation. */
inline void maskedPeak(const cv::Mat &image, const cv::Mat &rt, const cv::Mat &rm,
                       double &score, cv::Point &loc) {
    cv::Mat res;
    cv::matchTemplate(image, rt, res, cv::TM_CCOEFF_NORMED, rm);
    sanitize(res, -2.0);
    cv::minMaxLoc(res, nullptr, &score, nullptr, &loc);
}

inline void rotateCanvas(const RotCanvas &rc, const cv::Mat &src, const cv::Mat &srcmask,
                         int d, double angle, cv::Mat &rt, cv::Mat &rm) {
    const cv::Mat m = cv::getRotationMatrix2D(
        cv::Point2f((d - 1) / 2.0F, (d - 1) / 2.0F), angle, 1.0);
    cv::warpAffine(src, rt, m, cv::Size(d, d), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(0));
    cv::warpAffine(srcmask, rm, m, cv::Size(d, d), cv::INTER_NEAREST, cv::BORDER_CONSTANT,
                   cv::Scalar(0));
    (void)rc;
}

// -----------------------------------------------------------------------------
// Exhaustive rotated-bank NCC: the textbook way to add rotation to matchTemplate.
// -----------------------------------------------------------------------------
inline Res bruteForceNcc(const cv::Mat &image, const cv::Mat &templ, int angle_step = 1) {
    const RotCanvas rc = makeRotCanvas(templ);
    if (rc.d > image.cols || rc.d > image.rows) return {};

    const int n = (360 + angle_step - 1) / angle_step;
    std::vector<double> scores(n, -2.0);
    std::vector<cv::Point> locs(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < n; ++i) {
        cv::Mat rt, rm;
        rotateCanvas(rc, rc.image, rc.mask, rc.d, i * angle_step, rt, rm);
        maskedPeak(image, rt, rm, scores[i], locs[i]);
    }
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (scores[i] > scores[best]) best = i;
    }
    if (scores[best] <= kSentinel) return {};
    return {static_cast<double>(best * angle_step),
            locs[best].x + (rc.d - 1) / 2.0 - (image.cols - 1) / 2.0,
            locs[best].y + (rc.d - 1) / 2.0 - (image.rows - 1) / 2.0,
            scores[best], true};
}

// -----------------------------------------------------------------------------
// The same NCC, but with OrientMatch's search structure: a coarse angle scan on a
// downsampled image, then the top-k candidates refined at full resolution. Holding the
// search budget equal is what makes this the control for the scoring function.
// -----------------------------------------------------------------------------
inline Res coarseToFineNcc(const cv::Mat &image, const cv::Mat &templ,
                           double coarse_scale = 0.5, int coarse_step = 3, int top_k = 5) {
    const RotCanvas rc = makeRotCanvas(templ);
    const int d = rc.d;
    if (d > image.cols || d > image.rows) return {};

    cv::Mat small;
    cv::resize(image, small, cv::Size(), coarse_scale, coarse_scale, cv::INTER_AREA);
    const int ds = static_cast<int>(std::lround(d * coarse_scale));
    if (ds < 8 || ds > small.cols || ds > small.rows) {
        return bruteForceNcc(image, templ, coarse_step);
    }
    cv::Mat cimg, cmask;
    cv::resize(rc.image, cimg, cv::Size(ds, ds), 0, 0, cv::INTER_AREA);
    cv::resize(rc.mask, cmask, cv::Size(ds, ds), 0, 0, cv::INTER_NEAREST);

    const int n = (360 + coarse_step - 1) / coarse_step;
    std::vector<double> cscore(n, -2.0);
    std::vector<cv::Point> cloc(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < n; ++i) {
        cv::Mat rt, rm;
        rotateCanvas(rc, cimg, cmask, ds, i * coarse_step, rt, rm);
        maskedPeak(small, rt, rm, cscore[i], cloc[i]);
    }

    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return cscore[a] > cscore[b]; });
    order.resize(std::min<std::size_t>(order.size(), std::max(1, top_k)));

    const int pad = static_cast<int>(2.0 / coarse_scale) + 3;
    std::vector<cv::Rect> rois(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        const double cx = (cloc[order[i]].x + (ds - 1) / 2.0) / coarse_scale;
        const double cy = (cloc[order[i]].y + (ds - 1) / 2.0) / coarse_scale;
        const int w = std::min(image.cols, d + 2 * pad);
        const int h = std::min(image.rows, d + 2 * pad);
        rois[i] = {std::clamp((int)std::lround(cx - w / 2.0), 0, image.cols - w),
                   std::clamp((int)std::lround(cy - h / 2.0), 0, image.rows - h), w, h};
    }

    const int per = 2 * coarse_step + 1;
    const int tasks = static_cast<int>(order.size()) * per;
    std::vector<double> fscore(tasks, -2.0);
    std::vector<cv::Point> floc(tasks);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < tasks; ++i) {
        const int ci = i / per, k = i % per;
        const int a = ((order[ci] * coarse_step - coarse_step + k) % 360 + 360) % 360;
        cv::Mat rt, rm;
        rotateCanvas(rc, rc.image, rc.mask, d, a, rt, rm);
        maskedPeak(image(rois[ci]), rt, rm, fscore[i], floc[i]);
    }
    int best = 0;
    for (int i = 1; i < tasks; ++i) {
        if (fscore[i] > fscore[best]) best = i;
    }
    if (fscore[best] <= kSentinel) return {};
    const int ci = best / per;
    const int angle = ((order[ci] * coarse_step - coarse_step + best % per) % 360 + 360) % 360;
    return {static_cast<double>(angle),
            floc[best].x + rois[ci].x + (d - 1) / 2.0 - (image.cols - 1) / 2.0,
            floc[best].y + rois[ci].y + (d - 1) / 2.0 - (image.rows - 1) / 2.0,
            fscore[best], true};
}

// -----------------------------------------------------------------------------
// Fourier-Mellin: rotation from phase correlation of log-polar amplitude spectra.
// The amplitude spectrum is point symmetric, so theta and theta+180 are indistinguishable;
// both are scored and the better one kept.
// -----------------------------------------------------------------------------
inline cv::Mat logPolarMagnitude(const cv::Mat &gray) {
    cv::Mat f;
    gray.convertTo(f, CV_32F);
    cv::Mat window;
    cv::createHanningWindow(window, f.size(), CV_32F);
    f = f.mul(window);

    cv::Mat planes[] = {f, cv::Mat::zeros(f.size(), CV_32F)};
    cv::Mat complex;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex);
    cv::split(complex, planes);
    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);
    cv::log(mag + 1.0F, mag);

    const int cx = mag.cols / 2, cy = mag.rows / 2;
    cv::Mat q0(mag, {0, 0, cx, cy}), q1(mag, {cx, 0, cx, cy});
    cv::Mat q2(mag, {0, cy, cx, cy}), q3(mag, {cx, cy, cx, cy});
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    cv::Mat lp;
    cv::warpPolar(mag, lp, {256, 360}, cv::Point2f(cx, cy), std::min(cx, cy),
                  cv::INTER_LINEAR + cv::WARP_POLAR_LOG);
    return lp;
}

inline double maskedNccAt(const cv::Mat &image, const cv::Mat &templ, double angle,
                          double &dx, double &dy) {
    const RotCanvas rc = makeRotCanvas(templ);
    if (rc.d > image.cols || rc.d > image.rows) { dx = dy = 0; return -2.0; }
    cv::Mat rt, rm;
    rotateCanvas(rc, rc.image, rc.mask, rc.d, angle, rt, rm);
    double score;
    cv::Point loc;
    maskedPeak(image, rt, rm, score, loc);
    dx = loc.x + (rc.d - 1) / 2.0 - (image.cols - 1) / 2.0;
    dy = loc.y + (rc.d - 1) / 2.0 - (image.rows - 1) / 2.0;
    return score;
}

inline Res fourierMellin(const cv::Mat &image, const cv::Mat &templ) {
    cv::Mat canvas = cv::Mat::zeros(image.size(), CV_8U);
    templ.copyTo(canvas({(image.cols - templ.cols) / 2, (image.rows - templ.rows) / 2,
                         templ.cols, templ.rows}));
    const cv::Mat lp_image = logPolarMagnitude(image);
    const cv::Mat lp_templ = logPolarMagnitude(canvas);
    const cv::Point2d shift = cv::phaseCorrelate(lp_templ, lp_image);
    const double angle = shift.y / lp_image.rows * 360.0;

    double dx1, dy1, dx2, dy2;
    const double s1 = maskedNccAt(image, templ, wrap360(angle), dx1, dy1);
    const double s2 = maskedNccAt(image, templ, wrap360(angle + 180.0), dx2, dy2);
    if (s1 <= kSentinel && s2 <= kSentinel) return {};
    if (s1 >= s2) return {wrap360(angle), dx1, dy1, s1, true};
    return {wrap360(angle + 180.0), dx2, dy2, s2, true};
}

// -----------------------------------------------------------------------------
// Feature matching plus a RANSAC similarity fit: the usual alternative when the pose
// range is wide. Its score is the inlier count, not a correlation.
// -----------------------------------------------------------------------------
inline Res features(const cv::Mat &image, const cv::Mat &templ, bool use_sift) {
    cv::Ptr<cv::Feature2D> detector =
        use_sift ? cv::Ptr<cv::Feature2D>(cv::SIFT::create())
                 : cv::Ptr<cv::Feature2D>(cv::ORB::create(3000));

    std::vector<cv::KeyPoint> k_image, k_templ;
    cv::Mat d_image, d_templ;
    detector->detectAndCompute(image, cv::noArray(), k_image, d_image);
    detector->detectAndCompute(templ, cv::noArray(), k_templ, d_templ);
    if (d_templ.rows < 3 || d_image.rows < 3) return {};
    if (use_sift) {
        // OpenCV's SIFT detects on a 2x upsampled image, which leaves every keypoint at
        // x + 0.25. It cancels in the rotation but not in the translation.
        for (auto &k : k_image) k.pt -= cv::Point2f(0.25F, 0.25F);
        for (auto &k : k_templ) k.pt -= cv::Point2f(0.25F, 0.25F);
    }

    cv::BFMatcher matcher(use_sift ? cv::NORM_L2 : cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(d_templ, d_image, knn, 2);
    std::vector<cv::Point2f> p_templ, p_image;
    for (const auto &m : knn) {
        if (m.size() < 2 || m[0].distance >= 0.75F * m[1].distance) continue;  // Lowe ratio
        p_templ.push_back(k_templ[m[0].queryIdx].pt);
        p_image.push_back(k_image[m[0].trainIdx].pt);
    }
    if (p_templ.size() < 3) return {};

    cv::Mat inliers;
    const cv::Mat m = cv::estimateAffinePartial2D(p_templ, p_image, inliers, cv::RANSAC, 3.0);
    if (m.empty()) return {};

    // Image y points down, so negate to match getRotationMatrix2D's sign.
    const double angle =
        wrap360(-std::atan2(m.at<double>(1, 0), m.at<double>(0, 0)) * 180.0 / CV_PI);
    const double tcx = (templ.cols - 1) / 2.0, tcy = (templ.rows - 1) / 2.0;
    const double x = m.at<double>(0, 0) * tcx + m.at<double>(0, 1) * tcy + m.at<double>(0, 2);
    const double y = m.at<double>(1, 0) * tcx + m.at<double>(1, 1) * tcy + m.at<double>(1, 2);
    return {angle, x - (image.cols - 1) / 2.0, y - (image.rows - 1) / 2.0,
            static_cast<double>(cv::countNonZero(inliers)), true};
}

// -----------------------------------------------------------------------------
// Prepared forms.
//
// Every method here has per-template work that a real application would do once and
// reuse over a stream of frames - which is the whole point of OrientMatch's Matcher.
// Timing only the one-shot call would charge each method for that work on every frame
// and would understate all of them, unevenly. These wrappers hoist it so the same
// methods can also be timed per frame.
// -----------------------------------------------------------------------------

struct Prepared {
    virtual ~Prepared() = default;
    virtual Res match(const cv::Mat &image) = 0;
};

/** Shared by the two NCC baselines: the rotated template bank, built once. */
struct RotBank {
    std::vector<cv::Mat> image;
    std::vector<cv::Mat> mask;
    int d = 0;
    int step = 1;

    void build(const cv::Mat &src, const cv::Mat &srcmask, int size, int angle_step) {
        d = size;
        step = angle_step;
        const int n = (360 + angle_step - 1) / angle_step;
        image.resize(n);
        mask.resize(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < n; ++i) {
            const cv::Mat m = cv::getRotationMatrix2D(
                cv::Point2f((d - 1) / 2.0F, (d - 1) / 2.0F), i * angle_step, 1.0);
            cv::warpAffine(src, image[i], m, cv::Size(d, d), cv::INTER_LINEAR,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::warpAffine(srcmask, mask[i], m, cv::Size(d, d), cv::INTER_NEAREST,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
        }
    }
};

struct PreparedBruteForce : Prepared {
    RotCanvas rc;
    RotBank bank;
    int step;

    PreparedBruteForce(const cv::Mat &templ, int angle_step) : step(angle_step) {
        rc = makeRotCanvas(templ);
        bank.build(rc.image, rc.mask, rc.d, angle_step);
    }

    Res match(const cv::Mat &image) override {
        if (rc.d > image.cols || rc.d > image.rows) return {};
        const int n = static_cast<int>(bank.image.size());
        std::vector<double> scores(n, -2.0);
        std::vector<cv::Point> locs(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < n; ++i) {
            maskedPeak(image, bank.image[i], bank.mask[i], scores[i], locs[i]);
        }
        int best = 0;
        for (int i = 1; i < n; ++i) {
            if (scores[i] > scores[best]) best = i;
        }
        if (scores[best] <= kSentinel) return {};
        return {static_cast<double>(best * step),
                locs[best].x + (rc.d - 1) / 2.0 - (image.cols - 1) / 2.0,
                locs[best].y + (rc.d - 1) / 2.0 - (image.rows - 1) / 2.0,
                scores[best], true};
    }
};

struct PreparedCoarseToFine : Prepared {
    RotCanvas rc;
    RotBank coarse;
    RotBank fine;
    double coarse_scale;
    int coarse_step;
    int top_k;
    int ds = 0;

    PreparedCoarseToFine(const cv::Mat &templ, double scale, int step, int k)
        : coarse_scale(scale), coarse_step(step), top_k(k) {
        rc = makeRotCanvas(templ);
        ds = static_cast<int>(std::lround(rc.d * coarse_scale));
        cv::Mat cimg, cmask;
        cv::resize(rc.image, cimg, cv::Size(ds, ds), 0, 0, cv::INTER_AREA);
        cv::resize(rc.mask, cmask, cv::Size(ds, ds), 0, 0, cv::INTER_NEAREST);
        coarse.build(cimg, cmask, ds, coarse_step);
        fine.build(rc.image, rc.mask, rc.d, 1);
    }

    Res match(const cv::Mat &image) override {
        const int d = rc.d;
        if (d > image.cols || d > image.rows) return {};
        cv::Mat small;
        cv::resize(image, small, cv::Size(), coarse_scale, coarse_scale, cv::INTER_AREA);
        if (ds < 8 || ds > small.cols || ds > small.rows) return {};

        const int n = static_cast<int>(coarse.image.size());
        std::vector<double> cscore(n, -2.0);
        std::vector<cv::Point> cloc(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < n; ++i) {
            maskedPeak(small, coarse.image[i], coarse.mask[i], cscore[i], cloc[i]);
        }
        std::vector<int> order(n);
        for (int i = 0; i < n; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return cscore[a] > cscore[b]; });
        order.resize(std::min<std::size_t>(order.size(), std::max(1, top_k)));

        const int pad = static_cast<int>(2.0 / coarse_scale) + 3;
        std::vector<cv::Rect> rois(order.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            const double cx = (cloc[order[i]].x + (ds - 1) / 2.0) / coarse_scale;
            const double cy = (cloc[order[i]].y + (ds - 1) / 2.0) / coarse_scale;
            const int w = std::min(image.cols, d + 2 * pad);
            const int h = std::min(image.rows, d + 2 * pad);
            rois[i] = {std::clamp((int)std::lround(cx - w / 2.0), 0, image.cols - w),
                       std::clamp((int)std::lround(cy - h / 2.0), 0, image.rows - h), w, h};
        }
        const int per = 2 * coarse_step + 1;
        const int tasks = static_cast<int>(order.size()) * per;
        std::vector<double> fscore(tasks, -2.0);
        std::vector<cv::Point> floc(tasks);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < tasks; ++i) {
            const int ci = i / per, k = i % per;
            const int a = ((order[ci] * coarse_step - coarse_step + k) % 360 + 360) % 360;
            maskedPeak(image(rois[ci]), fine.image[a], fine.mask[a], fscore[i], floc[i]);
        }
        int best = 0;
        for (int i = 1; i < tasks; ++i) {
            if (fscore[i] > fscore[best]) best = i;
        }
        if (fscore[best] <= kSentinel) return {};
        const int ci = best / per;
        const int angle =
            ((order[ci] * coarse_step - coarse_step + best % per) % 360 + 360) % 360;
        return {static_cast<double>(angle),
                floc[best].x + rois[ci].x + (d - 1) / 2.0 - (image.cols - 1) / 2.0,
                floc[best].y + rois[ci].y + (d - 1) / 2.0 - (image.rows - 1) / 2.0,
                fscore[best], true};
    }
};

struct PreparedFeatures : Prepared {
    cv::Ptr<cv::Feature2D> detector;
    std::vector<cv::KeyPoint> k_templ;
    cv::Mat d_templ;
    cv::Size templ_size;
    bool sift;

    PreparedFeatures(const cv::Mat &templ, bool use_sift)
        : templ_size(templ.size()), sift(use_sift) {
        detector = use_sift ? cv::Ptr<cv::Feature2D>(cv::SIFT::create())
                            : cv::Ptr<cv::Feature2D>(cv::ORB::create(3000));
        detector->detectAndCompute(templ, cv::noArray(), k_templ, d_templ);
        if (use_sift) {
            for (auto &k : k_templ) k.pt -= cv::Point2f(0.25F, 0.25F);
        }
    }

    Res match(const cv::Mat &image) override {
        if (d_templ.rows < 3) return {};
        std::vector<cv::KeyPoint> k_image;
        cv::Mat d_image;
        detector->detectAndCompute(image, cv::noArray(), k_image, d_image);
        if (d_image.rows < 3) return {};
        if (sift) {
            for (auto &k : k_image) k.pt -= cv::Point2f(0.25F, 0.25F);
        }
        cv::BFMatcher matcher(sift ? cv::NORM_L2 : cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(d_templ, d_image, knn, 2);
        std::vector<cv::Point2f> p_templ, p_image;
        for (const auto &m : knn) {
            if (m.size() < 2 || m[0].distance >= 0.75F * m[1].distance) continue;
            p_templ.push_back(k_templ[m[0].queryIdx].pt);
            p_image.push_back(k_image[m[0].trainIdx].pt);
        }
        if (p_templ.size() < 3) return {};
        cv::Mat inliers;
        const cv::Mat m =
            cv::estimateAffinePartial2D(p_templ, p_image, inliers, cv::RANSAC, 3.0);
        if (m.empty()) return {};
        const double angle =
            wrap360(-std::atan2(m.at<double>(1, 0), m.at<double>(0, 0)) * 180.0 / CV_PI);
        const double tcx = (templ_size.width - 1) / 2.0, tcy = (templ_size.height - 1) / 2.0;
        const double x =
            m.at<double>(0, 0) * tcx + m.at<double>(0, 1) * tcy + m.at<double>(0, 2);
        const double y =
            m.at<double>(1, 0) * tcx + m.at<double>(1, 1) * tcy + m.at<double>(1, 2);
        return {angle, x - (image.cols - 1) / 2.0, y - (image.rows - 1) / 2.0,
                static_cast<double>(cv::countNonZero(inliers)), true};
    }
};

struct PreparedFourierMellin : Prepared {
    cv::Mat templ;
    PreparedFourierMellin(const cv::Mat &t) : templ(t.clone()) {}
    Res match(const cv::Mat &image) override { return fourierMellin(image, templ); }
};

}  // namespace bench
