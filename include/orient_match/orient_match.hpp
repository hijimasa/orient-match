#pragma once

#include <memory>
#include <string_view>

#include <opencv2/core.hpp>

namespace orient_match {

/** Configuration fixed when a Matcher is constructed. */
struct MatcherOptions {
    // Gradient-orientation field z = grad(I) / (|grad(I)| + eps).
    double blur_sigma = 1.0;
    double eps_ratio = 0.1;

    // Gaussian window applied to the template orientation field. Set to 0 to disable.
    double window_sigma = 1.0;

    // Coarse image level and pose search.
    double coarse_scale = 0.5;
    double angle_start_deg = 0.0;
    double angle_extent_deg = 360.0;  // Half-open range [start, start + extent).
    double coarse_angle_step_deg = 3.0;
    double fine_angle_step_deg = 1.0;
    int refine_top_k = 5;
};

enum class MatchStatus {
    ok,
    template_larger_than_image,
    no_finite_score,
};

/** Best pose found for the template. Coordinates use OpenCV's pixel-center convention. */
struct MatchResult {
    MatchStatus status = MatchStatus::no_finite_score;
    cv::Point2d center{};       // Absolute template-center position in the input image.
    double angle_deg = 0.0;    // Counter-clockwise, normalized to [0, 360).
    double score = -1.0;       // Normalized orientation correlation, approximately [-1, 1].

    [[nodiscard]] bool valid() const noexcept { return status == MatchStatus::ok; }
    explicit operator bool() const noexcept { return valid(); }
};

/** Human-readable text for a MatchStatus value. */
[[nodiscard]] std::string_view statusMessage(MatchStatus status) noexcept;

/**
 * Compute the dense, softly gated gradient-orientation field.
 *
 * The input must be a non-empty, single-channel image. The result is CV_32FC2,
 * where channel 0 is the x component and channel 1 is the y component.
 */
[[nodiscard]] cv::Mat orientationField(const cv::Mat &image,
                                       double blur_sigma = 1.0,
                                       double eps_ratio = 0.1);

/**
 * Reusable single-template matcher for translation and in-plane rotation.
 *
 * Construction extracts the template field and precomputes the coarse rotated
 * template bank. match() is const and can be called concurrently for different
 * input images.
 */
class Matcher {
public:
    explicit Matcher(const cv::Mat &template_image,
                     const MatcherOptions &options = {});
    Matcher(const cv::Mat &template_image, const cv::Mat &template_mask,
            const MatcherOptions &options = {});

    [[nodiscard]] MatchResult match(const cv::Mat &image) const;

    [[nodiscard]] cv::Size templateSize() const noexcept;
    [[nodiscard]] int canvasSize() const noexcept;
    [[nodiscard]] const MatcherOptions &options() const noexcept;

private:
    class Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace orient_match
