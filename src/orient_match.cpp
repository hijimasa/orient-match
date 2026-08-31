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
};

struct ImagePlanes {
    cv::Mat x;
    cv::Mat y;
    cv::Mat energy;
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
    const int size = std::max(3, static_cast<int>(std::lround(source.size * scale)));
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
    cv::warpAffine(source.mask, result.mask, transform, cv::Size(source.size, source.size),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.0F));

    // OpenCV image coordinates have y pointing down. Rotating the sample positions
    // therefore rotates the orientation vector values by the negative mathematical angle.
    const double radians = -angle_deg * kPi / 180.0;
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);
    result.x = warped_x * cosine - warped_y * sine;
    result.y = warped_x * sine + warped_y * cosine;
    return result;
}

ImagePlanes makeImagePlanes(const cv::Mat &image, const MatcherOptions &options) {
    cv::Mat field = orientationField(image, options.blur_sigma, options.eps_ratio);
    cv::Mat channels[2];
    cv::split(field, channels);
    return {channels[0], channels[1], channels[0].mul(channels[0]) +
                                               channels[1].mul(channels[1])};
}

Peak findPeak(const ImagePlanes &image, const RotatedTemplate &templ) {
    cv::Mat numerator_x;
    cv::Mat numerator_y;
    cv::Mat image_energy;
    cv::matchTemplate(image.x, templ.x, numerator_x, cv::TM_CCORR);
    cv::matchTemplate(image.y, templ.y, numerator_y, cv::TM_CCORR);
    cv::matchTemplate(image.energy, templ.mask, image_energy, cv::TM_CCORR);

    const double template_energy = std::max(
        cv::sum(templ.x.mul(templ.x))[0] + cv::sum(templ.y.mul(templ.y))[0], 1e-12);
    cv::Mat denominator;
    cv::max(image_energy, 1e-12F, denominator);
    cv::sqrt(denominator * template_energy, denominator);

    cv::Mat scores;
    cv::divide(numerator_x + numerator_y, denominator, scores);
    cv::patchNaNs(scores, kInvalidScore);
    const cv::Mat invalid = (scores < -1.001F) | (scores > 1.001F);
    scores.setTo(kInvalidScore, invalid);

    Peak peak;
    cv::minMaxLoc(scores, nullptr, &peak.score, nullptr, &peak.location);
    return peak;
}

std::vector<std::size_t> topIndices(const std::vector<Peak> &peaks, int count) {
    std::vector<std::size_t> indices(peaks.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::stable_sort(indices.begin(), indices.end(), [&](std::size_t left, std::size_t right) {
        return peaks[left].score > peaks[right].score;
    });
    indices.resize(std::min(indices.size(), static_cast<std::size_t>(count)));
    return indices;
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

std::vector<double> fineOffsets(double radius, double step) {
    std::vector<double> offsets;
    for (double value = -radius; value <= radius + 1e-9; value += step) {
        offsets.push_back(value);
    }
    offsets.push_back(0.0);
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end(), [](double left, double right) {
                      return std::abs(left - right) < 1e-9;
                  }),
                  offsets.end());
    return offsets;
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

    cv::Mat values;
    image.convertTo(values, CV_32F);
    if (blur_sigma > 0.0) {
        cv::GaussianBlur(values, values, cv::Size(), blur_sigma, blur_sigma,
                         cv::BORDER_DEFAULT);
    }

    cv::Mat gradient_x;
    cv::Mat gradient_y;
    cv::Sobel(values, gradient_x, CV_32F, 1, 0, 3);
    cv::Sobel(values, gradient_y, CV_32F, 0, 1, 3);
    cv::Mat magnitude;
    cv::magnitude(gradient_x, gradient_y, magnitude);
    const double epsilon = eps_ratio * cv::mean(magnitude)[0] + 1e-6;

    cv::Mat denominator = magnitude + epsilon;
    cv::Mat components[] = {gradient_x / denominator, gradient_y / denominator};
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

        const double end = options.angle_start_deg + options.angle_extent_deg;
        for (double angle = options.angle_start_deg; angle < end - 1e-9;
             angle += options.coarse_angle_step_deg) {
            coarse_angles.push_back(angle);
            coarse_templates.push_back(rotateCanvas(coarse, angle));
        }
        if (coarse_angles.empty()) {
            coarse_angles.push_back(options.angle_start_deg);
            coarse_templates.push_back(rotateCanvas(coarse, options.angle_start_deg));
        }
    }

    MatcherOptions options;
    cv::Size template_size;
    OrientationCanvas full;
    OrientationCanvas coarse;
    std::vector<double> coarse_angles;
    std::vector<RotatedTemplate> coarse_templates;
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
    validateSingleChannel(image, "image");
    if (impl_->full.size > image.cols || impl_->full.size > image.rows) {
        return {MatchStatus::template_larger_than_image};
    }

    cv::Mat coarse_image;
    cv::resize(image, coarse_image, cv::Size(), impl_->options.coarse_scale,
               impl_->options.coarse_scale, cv::INTER_AREA);
    if (impl_->coarse.size > coarse_image.cols || impl_->coarse.size > coarse_image.rows) {
        return {MatchStatus::template_larger_than_image};
    }

    const ImagePlanes coarse_planes = makeImagePlanes(coarse_image, impl_->options);
    std::vector<Peak> coarse_peaks(impl_->coarse_templates.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < static_cast<int>(impl_->coarse_templates.size()); ++i) {
        coarse_peaks[static_cast<std::size_t>(i)] =
            findPeak(coarse_planes, impl_->coarse_templates[static_cast<std::size_t>(i)]);
    }

    const std::vector<std::size_t> candidates =
        topIndices(coarse_peaks, impl_->options.refine_top_k);
    const ImagePlanes full_planes = makeImagePlanes(image, impl_->options);
    const int padding = static_cast<int>(std::ceil(2.0 / impl_->options.coarse_scale)) + 3;
    std::vector<cv::Rect> rois(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const Peak &peak = coarse_peaks[candidates[i]];
        const double coarse_center_x = peak.location.x + (impl_->coarse.size - 1) / 2.0;
        const double coarse_center_y = peak.location.y + (impl_->coarse.size - 1) / 2.0;
        // Inverse of OpenCV's half-pixel resize mapping.
        const double center_x =
            (coarse_center_x + 0.5) / impl_->options.coarse_scale - 0.5;
        const double center_y =
            (coarse_center_y + 0.5) / impl_->options.coarse_scale - 0.5;
        rois[i] = refinementRoi(image, center_x, center_y, impl_->full.size, padding);
    }

    struct FineTask {
        std::size_t candidate = 0;
        double angle = 0.0;
    };
    std::vector<FineTask> tasks;
    const auto offsets = fineOffsets(impl_->options.coarse_angle_step_deg,
                                     impl_->options.fine_angle_step_deg);
    const bool full_circle = std::abs(impl_->options.angle_extent_deg - 360.0) < 1e-9;
    const double range_end =
        impl_->options.angle_start_deg + impl_->options.angle_extent_deg;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const double base = impl_->coarse_angles[candidates[i]];
        for (double offset : offsets) {
            const double angle = base + offset;
            if (!full_circle &&
                (angle < impl_->options.angle_start_deg - 1e-9 || angle >= range_end - 1e-9)) {
                continue;
            }
            tasks.push_back({i, angle});
        }
    }

    std::vector<Peak> fine_peaks(tasks.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < static_cast<int>(tasks.size()); ++i) {
        const FineTask &task = tasks[static_cast<std::size_t>(i)];
        const RotatedTemplate rotated = rotateCanvas(impl_->full, task.angle);
        const cv::Rect &roi = rois[task.candidate];
        const ImagePlanes roi_planes{full_planes.x(roi), full_planes.y(roi),
                                     full_planes.energy(roi)};
        fine_peaks[static_cast<std::size_t>(i)] = findPeak(roi_planes, rotated);
    }

    MatchResult best;
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
        return {MatchStatus::no_finite_score};
    }
    return best;
}

}  // namespace orient_match
