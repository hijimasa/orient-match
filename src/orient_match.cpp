#include "orient_match/orient_match.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace orient_match {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr float kInvalidScore = -2.0F;
constexpr std::size_t kMaxCoarseAngleCandidates = 4096;
constexpr std::size_t kMaxFineAngleOffsets = 4097;
constexpr std::size_t kMaxFineTasks = 1000000;
// The orientation field has near-unit length, so the support area is the largest image
// energy a position can contribute. A position holding less than this fraction of it is
// textureless and carries no orientation information; flooring the denominator there
// stops the normalized score from amplifying correlation round-off into a false peak.
constexpr double kEnergyFloorRatio = 1e-6;
// Target step of the global angle scan. Over the reference scenes - rotation canvases
// from 64 to 500 pixels, coarse scales 0.50 and 0.35, cluttered backgrounds holding a
// mirrored look-alike - scanning this coarsely and recovering the angle locally
// reproduced the pose of an exhaustive scan exactly. The first differences appeared at a
// 24 degree step, so this leaves a factor of two in hand.
constexpr double kScanStepDeg = 12.0;
// The coarse level chooses the angle at reduced resolution. Over the reference scenes its
// choice sat up to two coarse steps from the full-resolution optimum, so the leading
// candidate is refined one ring wider than the gap between bank angles.
constexpr int kCoarseAngleUncertaintySteps = 2;

struct OrientationCanvas {
    cv::Mat x;
    cv::Mat y;
    cv::Mat mask;
    int size = 0;
};

struct RotatedTemplate {
    cv::Mat x;
    cv::Mat y;
    cv::Mat mask;
    // Constants of the normalized score, filled in by rotateCanvas.
    double energy = 0.0;        // sum of |z|^2 over the template
    double energy_floor = 0.0;  // smallest image energy the denominator will believe
};

struct ImagePlanes {
    cv::Mat x;
    cv::Mat y;
    cv::Mat energy;
};

/**
 * Transforms of one image's planes, shared by every angle of the coarse scan.
 *
 * cv::matchTemplate transforms the image again for every template it is given, which
 * is by far the largest redundancy in a rotation search: the image does not change
 * while the angle does. Transforming it once and reusing the spectra removes that.
 */
struct ImageSpectra {
    cv::Size image;
    cv::Size dft;
    cv::Mat x;
    cv::Mat y;
    cv::Mat energy;
};

/** Gradients shared by the interleaved field and the split planes. */
struct Gradients {
    cv::Mat x;
    cv::Mat y;
    cv::Mat magnitude;
    float epsilon = 0.0F;
};

struct Peak {
    double score = kInvalidScore;
    cv::Point location{};
};

void validateSingleChannel(const cv::Mat &image, const char *name) {
    if (image.empty()) {
        throw std::invalid_argument(std::string(name) + " must not be empty");
    }
    if (image.channels() != 1) {
        throw std::invalid_argument(std::string(name) + " must be single-channel");
    }
    if (image.rows < 3 || image.cols < 3) {
        throw std::invalid_argument(std::string(name) + " must be at least 3x3 pixels");
    }
}

void validateOptions(const MatcherOptions &o) {
    const auto finite = [](double value) { return std::isfinite(value); };
    if (!finite(o.blur_sigma) || o.blur_sigma < 0.0) {
        throw std::invalid_argument("blur_sigma must be finite and >= 0");
    }
    if (!finite(o.eps_ratio) || o.eps_ratio < 0.0) {
        throw std::invalid_argument("eps_ratio must be finite and >= 0");
    }
    if (!finite(o.window_sigma) || o.window_sigma < 0.0) {
        throw std::invalid_argument("window_sigma must be finite and >= 0");
    }
    if (!finite(o.coarse_scale) || o.coarse_scale <= 0.0 || o.coarse_scale > 1.0) {
        throw std::invalid_argument("coarse_scale must be in (0, 1]");
    }
    if (!finite(o.angle_start_deg)) {
        throw std::invalid_argument("angle_start_deg must be finite");
    }
    if (!finite(o.angle_extent_deg) || o.angle_extent_deg <= 0.0 ||
        o.angle_extent_deg > 360.0) {
        throw std::invalid_argument("angle_extent_deg must be in (0, 360]");
    }
    if (!finite(o.coarse_angle_step_deg) || o.coarse_angle_step_deg <= 0.0 ||
        o.coarse_angle_step_deg > 360.0) {
        throw std::invalid_argument("coarse_angle_step_deg must be in (0, 360]");
    }
    if (!finite(o.fine_angle_step_deg) || o.fine_angle_step_deg <= 0.0 ||
        o.fine_angle_step_deg > o.coarse_angle_step_deg) {
        throw std::invalid_argument(
            "fine_angle_step_deg must be in (0, coarse_angle_step_deg]");
    }
    if (o.refine_top_k < 1) {
        throw std::invalid_argument("refine_top_k must be >= 1");
    }
    if (o.angle_scan_stride < 0) {
        throw std::invalid_argument("angle_scan_stride must be >= 0");
    }
    if (!finite(o.min_score) || o.min_score > 1.0) {
        throw std::invalid_argument("min_score must be finite and <= 1");
    }
    if (!finite(o.coarse_gate_ratio) || o.coarse_gate_ratio < 0.0 ||
        o.coarse_gate_ratio > 1.0) {
        throw std::invalid_argument("coarse_gate_ratio must be in [0, 1]");
    }
    if (!finite(o.angle_start_deg + o.angle_extent_deg) ||
        o.angle_start_deg + o.angle_extent_deg <= o.angle_start_deg) {
        throw std::invalid_argument(
            "angle range is not representable; use a smaller angle_start_deg");
    }
    if (o.coarse_angle_step_deg < o.angle_extent_deg &&
        o.angle_start_deg + o.coarse_angle_step_deg <= o.angle_start_deg) {
        throw std::invalid_argument(
            "coarse_angle_step_deg is too small relative to angle_start_deg");
    }
    if (!finite(o.angle_start_deg + o.fine_angle_step_deg) ||
        o.angle_start_deg + o.fine_angle_step_deg <= o.angle_start_deg) {
        throw std::invalid_argument(
            "fine_angle_step_deg is too small relative to angle_start_deg");
    }
}

double wrap360(double angle) {
    angle = std::fmod(angle, 360.0);
    return angle < 0.0 ? angle + 360.0 : angle;
}

void applyGaussianWindow(cv::Mat &field, double sigma) {
    if (sigma == 0.0) {
        return;
    }
    const int height = field.rows;
    const int width = field.cols;
    const double denominator = 2.0 * sigma * sigma;
    for (int y = 0; y < height; ++y) {
        const double ny = 2.0 * y / (height - 1.0) - 1.0;
        auto *row = field.ptr<cv::Vec2f>(y);
        for (int x = 0; x < width; ++x) {
            const double nx = 2.0 * x / (width - 1.0) - 1.0;
            const float weight =
                static_cast<float>(std::exp(-(nx * nx + ny * ny) / denominator));
            row[x] *= weight;
        }
    }
}

cv::Mat binaryMask(const cv::Mat &input, cv::Size expected_size) {
    if (input.empty()) {
        return cv::Mat(expected_size, CV_32F, cv::Scalar(1.0F));
    }
    if (input.size() != expected_size || input.channels() != 1) {
        throw std::invalid_argument(
            "template_mask must be single-channel and match the template size");
    }
    cv::Mat nonzero;
    cv::compare(input, 0, nonzero, cv::CMP_NE);
    nonzero.convertTo(nonzero, CV_32F, 1.0 / 255.0);
    if (cv::countNonZero(nonzero) == 0) {
        throw std::invalid_argument("template_mask must contain at least one non-zero pixel");
    }
    return nonzero;
}

OrientationCanvas makeCanvas(const cv::Mat &template_image, const cv::Mat &template_mask,
                             const MatcherOptions &options) {
    cv::Mat field = orientationField(template_image, options.blur_sigma, options.eps_ratio);
    cv::Mat mask = binaryMask(template_mask, template_image.size());
    field.setTo(cv::Scalar(0.0F, 0.0F), mask == 0.0F);
    applyGaussianWindow(field, options.window_sigma);

    int size = static_cast<int>(
        std::ceil(std::hypot(static_cast<double>(template_image.rows),
                            static_cast<double>(template_image.cols))));
    // Preserve the template's pixel-center parity when both axes agree. This keeps
    // integer translations exactly representable for the common even-by-even and
    // odd-by-odd cases.
    if ((template_image.rows & 1) == (template_image.cols & 1) &&
        (size & 1) != (template_image.rows & 1)) {
        ++size;
    }
    const double tx = (size - template_image.cols) / 2.0;
    const double ty = (size - template_image.rows) / 2.0;
    const cv::Matx23d transform(1.0, 0.0, tx, 0.0, 1.0, ty);

    cv::Mat centered_field;
    cv::warpAffine(field, centered_field, transform, cv::Size(size, size), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(0.0F, 0.0F));
    cv::Mat centered_mask;
    cv::warpAffine(mask, centered_mask, transform, cv::Size(size, size), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(0.0F));

    cv::Mat channels[2];
    cv::split(centered_field, channels);
    return {channels[0], channels[1], centered_mask, size};
}

OrientationCanvas resizeCanvas(const OrientationCanvas &source, double scale) {
    const int size = static_cast<int>(std::lround(source.size * scale));
    if (size < 3) {
        throw std::invalid_argument(
            "coarse_scale is too small; the template canvas must remain at least 3x3");
    }
    OrientationCanvas result;
    result.size = size;
    cv::resize(source.x, result.x, cv::Size(size, size), 0.0, 0.0, cv::INTER_AREA);
    cv::resize(source.y, result.y, cv::Size(size, size), 0.0, 0.0, cv::INTER_AREA);
    cv::resize(source.mask, result.mask, cv::Size(size, size), 0.0, 0.0, cv::INTER_AREA);
    return result;
}

RotatedTemplate rotateCanvas(const OrientationCanvas &source, double angle_deg) {
    const float center = static_cast<float>((source.size - 1) / 2.0);
    const cv::Mat transform =
        cv::getRotationMatrix2D(cv::Point2f(center, center), angle_deg, 1.0);

    cv::Mat warped_x;
    cv::Mat warped_y;
    RotatedTemplate result;
    cv::warpAffine(source.x, warped_x, transform, cv::Size(source.size, source.size),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.0F));
    cv::warpAffine(source.y, warped_y, transform, cv::Size(source.size, source.size),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.0F));
    cv::Mat warped_mask;
    cv::warpAffine(source.mask, warped_mask, transform, cv::Size(source.size, source.size),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.0F));

    // The template planes already contain the interpolated mask amplitude. For cosine
    // normalization, image energy must instead be measured on a geometric support: using
    // the fractional amplitude here can violate Cauchy-Schwarz and produce scores > 1.
    cv::Mat support;
    cv::compare(warped_mask, 0.0F, support, cv::CMP_GT);
    support.convertTo(result.mask, CV_32F, 1.0 / 255.0);
    result.energy_floor = std::max(kEnergyFloorRatio * cv::sum(result.mask)[0], 1e-12);

    // OpenCV image coordinates have y pointing down. Rotating the sample positions
    // therefore rotates the orientation vector values by the negative mathematical angle.
    const double radians = -angle_deg * kPi / 180.0;
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);
    result.x = warped_x * cosine - warped_y * sine;
    result.y = warped_x * sine + warped_y * cosine;
    result.energy = std::max(
        cv::sum(result.x.mul(result.x))[0] + cv::sum(result.y.mul(result.y))[0], 1e-12);
    return result;
}

Gradients computeGradients(const cv::Mat &image, double blur_sigma, double eps_ratio) {
    cv::Mat values;
    image.convertTo(values, CV_32F);
    if (blur_sigma > 0.0) {
        cv::GaussianBlur(values, values, cv::Size(), blur_sigma, blur_sigma,
                         cv::BORDER_DEFAULT);
    }
    Gradients gradients;
    cv::Sobel(values, gradients.x, CV_32F, 1, 0, 3);
    cv::Sobel(values, gradients.y, CV_32F, 0, 1, 3);
    cv::magnitude(gradients.x, gradients.y, gradients.magnitude);
    gradients.epsilon =
        static_cast<float>(eps_ratio * cv::mean(gradients.magnitude)[0] + 1e-6);
    return gradients;
}

ImagePlanes makeImagePlanes(const cv::Mat &image, const MatcherOptions &options) {
    const Gradients gradients =
        computeGradients(image, options.blur_sigma, options.eps_ratio);

    // The correlation stages consume the planes, never the interleaved field, so the
    // normalization is written straight into them. Going through orientationField()
    // would merge two channels only to split them again, and would allocate about ten
    // full-size temporaries per call.
    ImagePlanes planes;
    planes.x.create(image.size(), CV_32F);
    planes.y.create(image.size(), CV_32F);
    planes.energy.create(image.size(), CV_32F);
    for (int y = 0; y < image.rows; ++y) {
        const float *gradient_x = gradients.x.ptr<float>(y);
        const float *gradient_y = gradients.y.ptr<float>(y);
        const float *magnitude = gradients.magnitude.ptr<float>(y);
        float *plane_x = planes.x.ptr<float>(y);
        float *plane_y = planes.y.ptr<float>(y);
        float *plane_energy = planes.energy.ptr<float>(y);
        for (int x = 0; x < image.cols; ++x) {
            const float denominator = magnitude[x] + gradients.epsilon;
            plane_x[x] = gradient_x[x] / denominator;
            plane_y[x] = gradient_y[x] / denominator;
            plane_energy[x] = plane_x[x] * plane_x[x] + plane_y[x] * plane_y[x];
        }
    }
    return planes;
}

ImageSpectra makeSpectra(const ImagePlanes &planes) {
    ImageSpectra spectra;
    spectra.image = planes.x.size();
    spectra.dft = {cv::getOptimalDFTSize(spectra.image.width),
                   cv::getOptimalDFTSize(spectra.image.height)};
    const cv::Mat *sources[] = {&planes.x, &planes.y, &planes.energy};
    cv::Mat *targets[] = {&spectra.x, &spectra.y, &spectra.energy};
    for (int i = 0; i < 3; ++i) {
        cv::Mat padded = cv::Mat::zeros(spectra.dft, CV_32F);
        sources[i]->copyTo(padded(cv::Rect(cv::Point(0, 0), spectra.image)));
        cv::dft(padded, *targets[i], 0, spectra.image.height);
    }
    return spectra;
}

/**
 * Correlate the cached image spectra with one rotated template.
 *
 * `energy` is filled in only when the caller passes an empty matrix. The support mask,
 * and therefore that term, is identical for an angle and that angle plus 180 degrees,
 * so half of the coarse scan can reuse the result of its counterpart.
 */
void correlate(const ImageSpectra &spectra, const RotatedTemplate &templ,
               cv::Mat &numerator, cv::Mat &energy) {
    const cv::Rect valid(0, 0, spectra.image.width - templ.x.cols + 1,
                         spectra.image.height - templ.x.rows + 1);
    const cv::Rect corner(0, 0, templ.x.cols, templ.x.rows);
    cv::Mat padded = cv::Mat::zeros(spectra.dft, CV_32F);
    cv::Mat spectrum;
    cv::Mat product;
    cv::Mat accumulator;
    cv::Mat inverse;

    templ.x.copyTo(padded(corner));
    cv::dft(padded, spectrum, 0, corner.height);
    cv::mulSpectrums(spectra.x, spectrum, accumulator, 0, true);
    templ.y.copyTo(padded(corner));
    cv::dft(padded, spectrum, 0, corner.height);
    cv::mulSpectrums(spectra.y, spectrum, product, 0, true);
    // CCS spectra add linearly, so both numerator terms share one inverse transform.
    accumulator += product;
    cv::idft(accumulator, inverse, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT, valid.height);
    numerator = inverse(valid).clone();

    if (energy.empty()) {
        templ.mask.copyTo(padded(corner));
        cv::dft(padded, spectrum, 0, corner.height);
        cv::mulSpectrums(spectra.energy, spectrum, product, 0, true);
        cv::idft(product, inverse, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT, valid.height);
        energy = inverse(valid).clone();
    }
}

/** Normalize the correlation maps and take the best pose in a single pass. */
Peak findPeak(const cv::Mat &numerator, const cv::Mat &energy,
              const RotatedTemplate &templ) {
    Peak peak;
    for (int y = 0; y < numerator.rows; ++y) {
        const float *numerator_row = numerator.ptr<float>(y);
        const float *energy_row = energy.ptr<float>(y);
        for (int x = 0; x < numerator.cols; ++x) {
            const double image_energy =
                std::max(static_cast<double>(energy_row[x]), templ.energy_floor);
            const double score =
                numerator_row[x] / std::sqrt(image_energy * templ.energy);
            if (!(score > peak.score) || !std::isfinite(score) || score < -1.001 ||
                score > 1.001) {
                continue;
            }
            peak.score = score;
            peak.location = {x, y};
        }
    }
    return peak;
}

/** Score one rotated template against a window whose transforms are already cached. */
Peak findPeak(const ImageSpectra &spectra, const RotatedTemplate &templ) {
    cv::Mat numerator;
    cv::Mat energy;
    correlate(spectra, templ, numerator, energy);
    return findPeak(numerator, energy, templ);
}

bool separated(const Peak &left, const Peak &right, int separation) {
    return std::abs(left.location.x - right.location.x) >= separation ||
           std::abs(left.location.y - right.location.y) >= separation;
}

/**
 * Peaks that actually scored, best first.
 *
 * Angles the global scan stepped over keep an invalid score and are dropped here, so that
 * the candidate list cannot fill up with angles nothing was ever measured at.
 */
std::vector<std::size_t> rankPeaks(const std::vector<Peak> &peaks) {
    std::vector<std::size_t> indices;
    indices.reserve(peaks.size());
    for (std::size_t i = 0; i < peaks.size(); ++i) {
        if (peaks[i].score > kInvalidScore) {
            indices.push_back(i);
        }
    }
    std::stable_sort(indices.begin(), indices.end(), [&](std::size_t left, std::size_t right) {
        return peaks[left].score > peaks[right].score;
    });
    return indices;
}

/**
 * Take the best candidates, skipping any that sits within `separation` of one already
 * taken. Neighbouring angles of a single position score almost alike, so without this
 * test the list fills up with one position seen several times: the refinement budget is
 * spent re-examining the same place, and the runner-up score says nothing about how
 * strongly that place won.
 */
std::vector<std::size_t> selectCandidates(const std::vector<Peak> &peaks,
                                          const std::vector<std::size_t> &ranked,
                                          int count, int separation) {
    std::vector<std::size_t> chosen;
    chosen.reserve(static_cast<std::size_t>(count));
    for (std::size_t index : ranked) {
        if (chosen.size() >= static_cast<std::size_t>(count)) {
            break;
        }
        const bool clear = std::all_of(
            chosen.begin(), chosen.end(), [&](std::size_t other) {
                return separated(peaks[other], peaks[index], separation);
            });
        if (clear) {
            chosen.push_back(index);
        }
    }
    return chosen;
}

/**
 * Best score at least `separation` away from the winner, or 0 when the winner stands
 * alone. The gap between the two is what distinguishes a present object, which wins its
 * position outright, from an empty scene, where look-alike peaks tie.
 */
double runnerUpScore(const std::vector<Peak> &peaks,
                     const std::vector<std::size_t> &ranked, int separation) {
    const Peak &winner = peaks[ranked.front()];
    for (std::size_t index : ranked) {
        if (separated(winner, peaks[index], separation)) {
            return std::max(peaks[index].score, 0.0);
        }
    }
    return 0.0;
}

cv::Rect refinementRoi(const cv::Mat &image, double center_x, double center_y,
                       int template_size, int padding) {
    const int width = std::min(image.cols, template_size + 2 * padding);
    const int height = std::min(image.rows, template_size + 2 * padding);
    int x = static_cast<int>(std::lround(center_x - (width - 1) / 2.0));
    int y = static_cast<int>(std::lround(center_y - (height - 1) / 2.0));
    x = std::clamp(x, 0, image.cols - width);
    y = std::clamp(y, 0, image.rows - height);
    return {x, y, width, height};
}

std::vector<double> coarseAngles(const MatcherOptions &options) {
    const long double start = options.angle_start_deg;
    const long double end = start + static_cast<long double>(options.angle_extent_deg);
    const long double step = options.coarse_angle_step_deg;
    std::vector<double> angles;
    for (std::size_t i = 0; i <= kMaxCoarseAngleCandidates; ++i) {
        const long double candidate = start + static_cast<long double>(i) * step;
        if (candidate >= end) {
            return angles;
        }
        if (angles.size() == kMaxCoarseAngleCandidates) {
            break;
        }
        const double angle = static_cast<double>(candidate);
        if (!std::isfinite(angle) || (!angles.empty() && angle <= angles.back())) {
            throw std::invalid_argument(
                "coarse angle samples are not representable with double precision");
        }
        angles.push_back(angle);
    }
    throw std::invalid_argument("angle range contains more than 4096 coarse samples");
}

/**
 * For each coarse angle, the index of an earlier angle exactly half a turn away.
 *
 * The template rectangle is centered on the rotation canvas, so its support is mapped
 * onto itself by a half turn and the two angles share their image-energy correlation.
 * The rasterized masks are compared rather than assumed equal, so an angle set that
 * does not have the property simply gets no twin.
 */
std::vector<int> maskTwins(const std::vector<double> &angles,
                           const std::vector<RotatedTemplate> &templates) {
    std::vector<int> twins(angles.size(), -1);
    for (std::size_t i = 0; i < angles.size(); ++i) {
        const double target = angles[i] - 180.0;
        const double tolerance = 1e-9 * std::max(1.0, std::abs(target));
        const auto first = angles.begin();
        const auto it = std::lower_bound(first, first + i, target - tolerance);
        if (it == first + i || std::abs(*it - target) > tolerance) {
            continue;
        }
        const std::size_t candidate = static_cast<std::size_t>(it - first);
        if (cv::norm(templates[i].mask, templates[candidate].mask, cv::NORM_INF) == 0.0) {
            twins[i] = static_cast<int>(candidate);
        }
    }
    return twins;
}

std::vector<double> fineOffsets(double radius, double step) {
    const long double ratio = static_cast<long double>(radius) / step;
    constexpr std::size_t kMaxRings = (kMaxFineAngleOffsets - 1) / 2;
    if (!std::isfinite(ratio) || ratio > static_cast<long double>(kMaxRings)) {
        throw std::invalid_argument("fine angle range contains more than 4097 samples");
    }

    const std::size_t rings = static_cast<std::size_t>(std::floor(ratio));
    const double tolerance =
        8.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, radius);
    std::vector<double> offsets{0.0};
    offsets.reserve(std::min(kMaxFineAngleOffsets, 2 * rings + 3));
    bool has_radius = false;
    for (std::size_t i = 1; i <= rings; ++i) {
        double value = static_cast<double>(static_cast<long double>(i) * step);
        if (std::abs(value - radius) <= tolerance) {
            value = radius;
            has_radius = true;
        }
        offsets.push_back(-value);
        offsets.push_back(value);
    }
    if (!has_radius) {
        offsets.push_back(-radius);
        offsets.push_back(radius);
    }
    if (offsets.size() > kMaxFineAngleOffsets) {
        throw std::invalid_argument("fine angle range contains more than 4097 samples");
    }
    std::sort(offsets.begin(), offsets.end());
    return offsets;
}

/** Offsets in the ring between `inner` and `outer` degrees, at `step` spacing. */
std::vector<double> outerOffsets(double inner, double outer, double step) {
    const double tolerance =
        8.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, outer);
    std::vector<double> offsets;
    for (std::size_t i = 1; offsets.size() + 2 <= kMaxFineAngleOffsets; ++i) {
        const double value = static_cast<double>(static_cast<long double>(i) * step);
        if (!std::isfinite(value) || value > outer + tolerance) {
            break;
        }
        if (value > inner + tolerance) {
            offsets.push_back(-value);
            offsets.push_back(value);
        }
    }
    std::sort(offsets.begin(), offsets.end());
    return offsets;
}

/** Bank indices within `radius` steps of `center`, wrapping only over a full circle. */
std::vector<int> neighbourAngles(int center, int radius, int count, bool wrap) {
    std::vector<int> indices;
    indices.reserve(static_cast<std::size_t>(2 * radius + 1));
    for (int offset = -radius; offset <= radius; ++offset) {
        int index = center + offset;
        if (wrap) {
            index = (index % count + count) % count;
        } else if (index < 0 || index >= count) {
            continue;
        }
        indices.push_back(index);
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

/**
 * Pair up scanned angles that can share an image-energy correlation.
 *
 * Returns (owner, sharer) pairs, where sharer is -1 when the angle's half-turn
 * counterpart is not itself scanned. Pairing them inside one task, rather than caching
 * every correlation map, keeps the extra memory to one map per thread.
 */
std::vector<std::pair<int, int>> pairScannedAngles(const std::vector<int> &scan,
                                                   const std::vector<int> &mask_twin) {
    const auto at = [](const std::vector<int> &v, int i) {
        return v[static_cast<std::size_t>(i)];
    };
    std::vector<char> scanned(mask_twin.size(), 0);
    std::vector<int> sharer(mask_twin.size(), -1);
    for (const int i : scan) {
        scanned[static_cast<std::size_t>(i)] = 1;
    }
    for (const int i : scan) {
        const int twin = at(mask_twin, i);
        if (twin >= 0 && scanned[static_cast<std::size_t>(twin)] != 0) {
            sharer[static_cast<std::size_t>(twin)] = i;
        }
    }
    std::vector<std::pair<int, int>> tasks;
    tasks.reserve(scan.size());
    for (const int i : scan) {
        const int twin = at(mask_twin, i);
        if (twin >= 0 && scanned[static_cast<std::size_t>(twin)] != 0) {
            continue;  // this angle is handled as its twin's sharer
        }
        tasks.emplace_back(i, at(sharer, i));
    }
    return tasks;
}

/** Full-resolution refinement window around a canvas position found at the coarse level. */
cv::Rect fullResolutionRoi(const cv::Mat &image, cv::Point coarse_location, int coarse_size,
                           int canvas_size, double coarse_scale) {
    // Inverse of OpenCV's half-pixel resize mapping.
    const double center_x =
        (coarse_location.x + (coarse_size - 1) / 2.0 + 0.5) / coarse_scale - 0.5;
    const double center_y =
        (coarse_location.y + (coarse_size - 1) / 2.0 + 0.5) / coarse_scale - 0.5;
    const int padding = static_cast<int>(std::ceil(2.0 / coarse_scale)) + 3;
    return refinementRoi(image, center_x, center_y, canvas_size, padding);
}

/**
 * How many bank angles the global search skips.
 *
 * The score varies smoothly with angle, so a pose is still the strongest thing in the
 * image when it is viewed several degrees off. Finding the position from a sparse set of
 * angles and recovering the angle locally therefore costs a fraction of scanning every
 * angle over the whole image, and lands on the same pose.
 */
int resolveScanStride(const MatcherOptions &options, std::size_t count) {
    const int limit = static_cast<int>(std::max<std::size_t>(count, 1));
    if (options.angle_scan_stride > 0) {
        return std::min(options.angle_scan_stride, limit);
    }
    const int stride = static_cast<int>(kScanStepDeg / options.coarse_angle_step_deg);
    return std::clamp(stride, 1, limit);
}

}  // namespace

cv::Mat orientationField(const cv::Mat &image, double blur_sigma, double eps_ratio) {
    validateSingleChannel(image, "image");
    if (!std::isfinite(blur_sigma) || blur_sigma < 0.0) {
        throw std::invalid_argument("blur_sigma must be finite and >= 0");
    }
    if (!std::isfinite(eps_ratio) || eps_ratio < 0.0) {
        throw std::invalid_argument("eps_ratio must be finite and >= 0");
    }

    const Gradients gradients = computeGradients(image, blur_sigma, eps_ratio);
    const cv::Mat denominator = gradients.magnitude + gradients.epsilon;
    cv::Mat components[] = {gradients.x / denominator, gradients.y / denominator};
    cv::Mat result;
    cv::merge(components, 2, result);
    return result;
}

std::string_view statusMessage(MatchStatus status) noexcept {
    switch (status) {
        case MatchStatus::ok:
            return "ok";
        case MatchStatus::template_larger_than_image:
            return "template rotation canvas is larger than the image";
        case MatchStatus::no_finite_score:
            return "no finite correlation score";
        case MatchStatus::below_min_score:
            return "best pose scored below min_score";
    }
    return "unknown status";
}

class Matcher::Impl {
public:
    Impl(const cv::Mat &template_image, const cv::Mat &template_mask,
         MatcherOptions matcher_options)
        : options(std::move(matcher_options)), template_size(template_image.size()) {
        validateSingleChannel(template_image, "template_image");
        validateOptions(options);
        full = makeCanvas(template_image, template_mask, options);
        const double template_energy =
            cv::sum(full.x.mul(full.x))[0] + cv::sum(full.y.mul(full.y))[0];
        if (!std::isfinite(template_energy) || template_energy <= 1e-12) {
            throw std::invalid_argument("template_image has no usable gradient energy");
        }
        coarse = resizeCanvas(full, options.coarse_scale);
        coarse_angles = coarseAngles(options);
        fine_offsets = fineOffsets(options.coarse_angle_step_deg,
                                   options.fine_angle_step_deg);
        outer_fine_offsets = outerOffsets(
            options.coarse_angle_step_deg,
            kCoarseAngleUncertaintySteps * options.coarse_angle_step_deg,
            options.fine_angle_step_deg);
        scan_stride = resolveScanStride(options, coarse_angles.size());
        const std::size_t refined_candidates = std::min(
            coarse_angles.size(), static_cast<std::size_t>(options.refine_top_k));
        if (refined_candidates > kMaxFineTasks / fine_offsets.size()) {
            throw std::invalid_argument("angle options produce more than 1000000 fine tasks");
        }

        coarse_templates.reserve(coarse_angles.size());
        for (double angle : coarse_angles) {
            coarse_templates.push_back(rotateCanvas(coarse, angle));
        }
        coarse_mask_twin = maskTwins(coarse_angles, coarse_templates);
    }

    MatcherOptions options;
    cv::Size template_size;
    OrientationCanvas full;
    OrientationCanvas coarse;
    std::vector<double> coarse_angles;
    std::vector<RotatedTemplate> coarse_templates;
    // For each coarse angle, the earlier angle whose support mask is the same one.
    std::vector<int> coarse_mask_twin;
    // Fine offsets around a candidate's angle, and the ring beyond them that only the
    // leading candidate is searched over.
    std::vector<double> fine_offsets;
    std::vector<double> outer_fine_offsets;
    int scan_stride = 1;
};

Matcher::Matcher(const cv::Mat &template_image, const MatcherOptions &options)
    : impl_(std::make_shared<Impl>(template_image, cv::Mat(), options)) {}

Matcher::Matcher(const cv::Mat &template_image, const cv::Mat &template_mask,
                 const MatcherOptions &options)
    : impl_(std::make_shared<Impl>(template_image, template_mask, options)) {}

cv::Size Matcher::templateSize() const noexcept { return impl_->template_size; }

int Matcher::canvasSize() const noexcept { return impl_->full.size; }

const MatcherOptions &Matcher::options() const noexcept { return impl_->options; }

MatchResult Matcher::match(const cv::Mat &image) const {
    const MatcherOptions &options = impl_->options;
    validateSingleChannel(image, "image");
    if (impl_->full.size > image.cols || impl_->full.size > image.rows) {
        return {MatchStatus::template_larger_than_image};
    }

    cv::Mat coarse_image;
    cv::resize(image, coarse_image, cv::Size(), options.coarse_scale, options.coarse_scale,
               cv::INTER_AREA);
    if (impl_->coarse.size > coarse_image.cols || impl_->coarse.size > coarse_image.rows) {
        return {MatchStatus::template_larger_than_image};
    }

    const ImagePlanes coarse_planes = makeImagePlanes(coarse_image, options);
    const ImageSpectra coarse_spectra = makeSpectra(coarse_planes);
    const int coarse_count = static_cast<int>(impl_->coarse_templates.size());
    const bool full_circle = options.angle_extent_deg == 360.0;

    // ---- global search: every position, but only every scan_stride-th angle ----
    std::vector<int> scan;
    scan.reserve(static_cast<std::size_t>(coarse_count / impl_->scan_stride + 1));
    for (int i = 0; i < coarse_count; i += impl_->scan_stride) {
        scan.push_back(i);
    }

    // The two angles of a half-turn pair share their image-energy correlation.
    const std::vector<std::pair<int, int>> coarse_tasks =
        pairScannedAngles(scan, impl_->coarse_mask_twin);

    // Angles left out of the scan keep an invalid score and are dropped by rankPeaks.
    std::vector<Peak> coarse_peaks(static_cast<std::size_t>(coarse_count));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int t = 0; t < static_cast<int>(coarse_tasks.size()); ++t) {
        const auto &task = coarse_tasks[static_cast<std::size_t>(t)];
        cv::Mat numerator;
        cv::Mat energy;
        for (const int i : {task.first, task.second}) {
            if (i < 0) {
                continue;
            }
            const std::size_t index = static_cast<std::size_t>(i);
            const RotatedTemplate &templ = impl_->coarse_templates[index];
            correlate(coarse_spectra, templ, numerator, energy);
            coarse_peaks[index] = findPeak(numerator, energy, templ);
        }
    }

    // ---- candidate positions ----
    // Two positions closer than half a canvas are the same detection seen twice.
    const int separation = std::max(1, impl_->coarse.size / 2);
    const std::vector<std::size_t> ranked = rankPeaks(coarse_peaks);
    if (ranked.empty()) {
        return {MatchStatus::no_finite_score};
    }
    const std::vector<std::size_t> candidates =
        selectCandidates(coarse_peaks, ranked, options.refine_top_k, separation);

    MatchResult diagnostics;
    diagnostics.margin =
        coarse_peaks[ranked.front()].score - runnerUpScore(coarse_peaks, ranked, separation);

    // ---- local angle search over the gap the global scan stepped across ----
    // A small window around each candidate is cheap enough to try every bank angle on,
    // which is what makes the sparse global scan safe.
    struct Candidate {
        cv::Point location{};
        int angle_index = 0;
        double score = kInvalidScore;
    };
    struct AngleTask {
        std::size_t candidate = 0;
        int angle_index = 0;
    };
    std::vector<Candidate> refined(candidates.size());
    for (std::size_t c = 0; c < candidates.size(); ++c) {
        refined[c] = {coarse_peaks[candidates[c]].location,
                      static_cast<int>(candidates[c]), coarse_peaks[candidates[c]].score};
    }
    if (impl_->scan_stride > 1) {
        const double half_gap = 0.5 * impl_->scan_stride * options.coarse_angle_step_deg;
        const int pad = static_cast<int>(std::ceil(
                            impl_->coarse.size * std::sin(half_gap * kPi / 180.0))) + 2;
        const int radius = impl_->scan_stride / 2;

        std::vector<cv::Rect> windows(candidates.size());
        std::vector<ImageSpectra> spectra(candidates.size());
        std::vector<AngleTask> local;
        for (std::size_t c = 0; c < candidates.size(); ++c) {
            const cv::Point &location = refined[c].location;
            cv::Rect window(location.x - pad, location.y - pad,
                            impl_->coarse.size + 2 * pad, impl_->coarse.size + 2 * pad);
            windows[c] = window & cv::Rect(0, 0, coarse_planes.x.cols, coarse_planes.x.rows);
            for (const int index : neighbourAngles(refined[c].angle_index, radius,
                                                   coarse_count, full_circle)) {
                local.push_back({c, index});
            }
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int c = 0; c < static_cast<int>(windows.size()); ++c) {
            const cv::Rect &window = windows[static_cast<std::size_t>(c)];
            spectra[static_cast<std::size_t>(c)] =
                makeSpectra({coarse_planes.x(window), coarse_planes.y(window),
                             coarse_planes.energy(window)});
        }
        std::vector<Peak> peaks(local.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < static_cast<int>(local.size()); ++i) {
            const AngleTask &task = local[static_cast<std::size_t>(i)];
            peaks[static_cast<std::size_t>(i)] = findPeak(
                spectra[task.candidate],
                impl_->coarse_templates[static_cast<std::size_t>(task.angle_index)]);
        }
        std::vector<double> best_score(candidates.size(), kInvalidScore);
        for (std::size_t i = 0; i < local.size(); ++i) {
            const AngleTask &task = local[i];
            if (peaks[i].score <= best_score[task.candidate]) {
                continue;
            }
            best_score[task.candidate] = peaks[i].score;
            const cv::Rect &window = windows[task.candidate];
            refined[task.candidate] = {
                cv::Point(window.x + peaks[i].location.x, window.y + peaks[i].location.y),
                task.angle_index, peaks[i].score};
        }
    }

    // The coarse score is read after the local angle refinement, where it is a much closer
    // estimate of the final score than the sparse scan alone can give.
    for (const Candidate &candidate : refined) {
        diagnostics.coarse_score = std::max(diagnostics.coarse_score, candidate.score);
    }

    // Nothing in the image comes close enough to be worth refining: the full-resolution
    // field and the fine search are skipped.
    if (options.min_score >= 0.0 &&
        diagnostics.coarse_score < options.coarse_gate_ratio * options.min_score) {
        diagnostics.status = MatchStatus::below_min_score;
        return diagnostics;
    }

    // ---- full-resolution refinement ----
    const ImagePlanes full_planes = makeImagePlanes(image, options);
    std::vector<cv::Rect> rois(refined.size());
    std::vector<ImageSpectra> roi_spectra(refined.size());
    for (std::size_t i = 0; i < refined.size(); ++i) {
        rois[i] = fullResolutionRoi(image, refined[i].location, impl_->coarse.size,
                                    impl_->full.size, options.coarse_scale);
    }
    // Every angle of a candidate is correlated against the same window, so the window is
    // transformed once here rather than inside each of those correlations.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < static_cast<int>(rois.size()); ++i) {
        const cv::Rect &roi = rois[static_cast<std::size_t>(i)];
        roi_spectra[static_cast<std::size_t>(i)] = makeSpectra(
            {full_planes.x(roi), full_planes.y(roi), full_planes.energy(roi)});
    }

    struct FineTask {
        std::size_t candidate = 0;
        double angle = 0.0;
    };
    const double range_end = options.angle_start_deg + options.angle_extent_deg;
    const auto in_range = [&](double angle) {
        return full_circle || (angle >= options.angle_start_deg && angle < range_end);
    };

    std::vector<FineTask> tasks;
    for (std::size_t i = 0; i < refined.size(); ++i) {
        const double base =
            impl_->coarse_angles[static_cast<std::size_t>(refined[i].angle_index)];
        for (double offset : impl_->fine_offsets) {
            if (in_range(base + offset)) {
                tasks.push_back({i, base + offset});
            }
        }
    }

    std::vector<Peak> fine_peaks(tasks.size());
    const auto run_tasks = [&](std::size_t first) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = static_cast<int>(first); i < static_cast<int>(tasks.size()); ++i) {
            const FineTask &task = tasks[static_cast<std::size_t>(i)];
            fine_peaks[static_cast<std::size_t>(i)] =
                findPeak(roi_spectra[task.candidate], rotateCanvas(impl_->full, task.angle));
        }
    };
    run_tasks(0);

    const auto leading_task = [&]() {
        std::size_t best = 0;
        for (std::size_t i = 1; i < tasks.size(); ++i) {
            if (fine_peaks[i].score > fine_peaks[best].score) {
                best = i;
            }
        }
        return best;
    };

    // The coarse level can be a step or two off the full-resolution optimum, so the
    // leading candidate - and only that one - is searched one ring further out.
    if (!tasks.empty() && !impl_->outer_fine_offsets.empty()) {
        const std::size_t candidate = tasks[leading_task()].candidate;
        const double base =
            impl_->coarse_angles[static_cast<std::size_t>(refined[candidate].angle_index)];
        const std::size_t first = tasks.size();
        for (double offset : impl_->outer_fine_offsets) {
            if (in_range(base + offset)) {
                tasks.push_back({candidate, base + offset});
            }
        }
        fine_peaks.resize(tasks.size());
        run_tasks(first);
    }

    MatchResult best = diagnostics;
    best.score = kInvalidScore;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        const Peak &peak = fine_peaks[i];
        if (peak.score <= best.score) {
            continue;
        }
        const cv::Rect &roi = rois[tasks[i].candidate];
        best.status = MatchStatus::ok;
        best.angle_deg = wrap360(tasks[i].angle);
        best.center.x = roi.x + peak.location.x + (impl_->full.size - 1) / 2.0;
        best.center.y = roi.y + peak.location.y + (impl_->full.size - 1) / 2.0;
        best.score = peak.score;
    }
    if (best.score <= kInvalidScore + 0.01) {
        diagnostics.status = MatchStatus::no_finite_score;
        return diagnostics;
    }
    if (options.min_score >= 0.0 && best.score < options.min_score) {
        best.status = MatchStatus::below_min_score;
    }
    return best;
}

}  // namespace orient_match
