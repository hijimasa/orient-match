#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

namespace orient_match {

/** Configuration fixed when a Matcher is constructed. */
struct MatcherOptions {
    // Gradient-orientation field z = grad(I) / (|grad(I)| + eps).
    double blur_sigma = 1.0;
    double eps_ratio = 0.1;

    // Gaussian window applied to the template orientation field. Set to 0 to disable.
    double window_sigma = 1.0;

    // Coarse image level and pose search. Configurations producing more than
    // 4096 coarse angles, 4097 fine offsets, or 1,000,000 fine tasks are rejected.
    // The scaled template rotation canvas must remain at least 3x3 pixels.
    double coarse_scale = 0.5;
    double angle_start_deg = 0.0;
    double angle_extent_deg = 360.0;  // Half-open range [start, start + extent).
    double coarse_angle_step_deg = 3.0;
    double fine_angle_step_deg = 1.0;

    // How many candidate positions the refinement stages examine. Candidates are kept
    // apart from one another, so this counts distinct places in the image rather than
    // several angles of the same place - and it is therefore also the largest number of
    // matches matchAll() can report.
    int refine_top_k = 5;

    // How far apart two candidate places must be at the coarse level, as a fraction of
    // the rotation canvas. It keeps the refinement budget on distinct places instead of
    // on several angles of one; lower it when instances stand close together, knowing the
    // budget then goes to near-duplicates. Two instances closer than this are reported as
    // one, whatever max_overlap says.
    double candidate_separation = 0.5;

    // matchAll() drops a pose whose template rectangle overlaps a better one by more than
    // this, as intersection over union. Unlike candidate_separation this is measured on
    // the refined poses, using the template's own rectangle at its detected angle rather
    // than the square canvas around it.
    double max_overlap = 0.5;

    // Only every angle_scan_stride-th bank angle takes part in the global search over the
    // whole image; the angles in between are examined locally, around the positions that
    // search returns. Since the global search is by far the most expensive stage, this is
    // the main speed control. 0 picks a stride from the canvas size, such that one scan
    // step moves the template rim by a few coarse pixels. 1 searches every angle globally.
    int angle_scan_stride = 0;

    // Acceptance test for "is the template present at all?".
    //
    // A negative min_score accepts every pose, which is the default. When it is set, a
    // best pose scoring below it is reported as below_min_score instead of ok. The right
    // value is template- and application-specific: calibrate it on images that do and do
    // not contain the object, using MatchResult::score and MatchResult::margin.
    //
    // The coarse stage also gives up early, skipping all full-resolution work, once its
    // best score falls below coarse_gate_ratio * min_score. A true match scores lower at
    // the coarse level than it finally does, so this ratio is the assumed floor on that
    // shrinkage: over the reference scenes it never fell below 0.79 at coarse_scale 0.5,
    // or below 0.70 at coarse_scale 0.35. Lower the ratio towards 0 to make an early
    // rejection less likely, at the cost of searching empty images in full.
    double min_score = -1.0;
    double coarse_gate_ratio = 0.6;
};

enum class MatchStatus {
    ok,
    template_larger_than_image,
    no_finite_score,
    below_min_score,
};

/** Best pose found for the template. Coordinates use OpenCV's pixel-center convention. */
struct MatchResult {
    MatchStatus status = MatchStatus::no_finite_score;
    cv::Point2d center{};       // Absolute template-center position in the input image.
    double angle_deg = 0.0;    // Counter-clockwise, normalized to [0, 360).
    double score = -1.0;       // Normalized orientation correlation, approximately [-1, 1].

    // Diagnostics from the coarse level, for calibrating min_score. Both are filled in for
    // rejected results too; when the coarse level rejects early, the pose fields above are
    // left unset, because no pose was refined.
    //
    // coarse_score is the best coarse-resolution score once the angle has been refined
    // locally. It always runs below the final score, by the factor coarse_gate_ratio
    // bounds, and it is what the early rejection tests.
    //
    // margin is how far the best angle of the global scan stands above the best one at
    // least one rotation-canvas radius away. A present object wins its position outright;
    // an empty scene leaves look-alike peaks near a tie.
    double coarse_score = -1.0;
    double margin = 0.0;

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

    /**
     * Every distinct pose found, best first.
     *
     * At most refine_top_k entries, each at or above min_score and none overlapping a
     * better one by more than max_overlap. The list is empty when nothing qualifies;
     * match() is the form that reports why. With min_score left at its default no pose is
     * ever rejected on score, so the list runs to refine_top_k entries and its tail is
     * whatever the image happened to offer - set min_score before relying on the length.
     */
    [[nodiscard]] std::vector<MatchResult> matchAll(const cv::Mat &image) const;

    [[nodiscard]] cv::Size templateSize() const noexcept;
    [[nodiscard]] int canvasSize() const noexcept;
    [[nodiscard]] const MatcherOptions &options() const noexcept;

private:
    class Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace orient_match
