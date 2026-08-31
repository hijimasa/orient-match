#include "orient_match/orient_match.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

double angleError(double left, double right) {
    double difference = std::fmod(std::abs(left - right), 360.0);
    return std::min(difference, 360.0 - difference);
}

cv::Mat makeTemplate() {
    cv::Mat templ(40, 48, CV_8U, cv::Scalar(25));
    cv::rectangle(templ, cv::Rect(5, 6, 28, 8), cv::Scalar(210), cv::FILLED);
    cv::circle(templ, cv::Point(34, 28), 7, cv::Scalar(150), cv::FILLED);
    cv::line(templ, cv::Point(7, 34), cv::Point(28, 18), cv::Scalar(245), 3);
    cv::rectangle(templ, cv::Rect(8, 17, 8, 12), cv::Scalar(75), cv::FILLED);
    return templ;
}

cv::Mat placeTemplate(const cv::Mat &templ, cv::Size image_size, cv::Point2d center,
                      double angle_deg) {
    const cv::Point2f source_center((templ.cols - 1) / 2.0F, (templ.rows - 1) / 2.0F);
    cv::Mat transform = cv::getRotationMatrix2D(source_center, angle_deg, 1.0);
    transform.at<double>(0, 2) += center.x - source_center.x;
    transform.at<double>(1, 2) += center.y - source_center.y;
    cv::Mat image;
    cv::warpAffine(templ, image, transform, image_size, cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(25));
    return image;
}

orient_match::MatcherOptions testOptions() {
    orient_match::MatcherOptions options;
    options.coarse_scale = 0.5;
    options.coarse_angle_step_deg = 5.0;
    options.fine_angle_step_deg = 1.0;
    options.refine_top_k = 5;
    return options;
}

void poseRegression() {
    const cv::Mat templ = makeTemplate();
    const orient_match::Matcher matcher(templ, testOptions());

    struct Case {
        cv::Point2d center;
        double angle;
    };
    const Case cases[] = {
        {{119.5, 99.5}, 0.0},
        {{143.5, 76.5}, 37.0},
        {{82.5, 126.5}, 213.0},
    };

    for (const Case &test_case : cases) {
        const cv::Mat image =
            placeTemplate(templ, cv::Size(240, 200), test_case.center, test_case.angle);
        const orient_match::MatchResult result = matcher.match(image);
        expect(result.valid(), "pose regression should return a valid result");
        if (!result) {
            continue;
        }
        expect(angleError(result.angle_deg, test_case.angle) <= 1.01,
               "angle error should be at most one fine-search step");
        expect(cv::norm(result.center - test_case.center) <= 1.1,
               "center error should be at most 1.1 pixels");
        expect(result.score > 0.75, "clean-pose score should be high");
    }
}

void maskRegression() {
    cv::Mat templ = makeTemplate();
    cv::Mat mask(templ.size(), CV_8U, cv::Scalar(0));
    cv::rectangle(mask, cv::Rect(3, 3, 38, 34), cv::Scalar(255), cv::FILLED);
    const orient_match::Matcher matcher(templ, mask, testOptions());
    const cv::Point2d center(126.5, 91.5);
    const cv::Mat image = placeTemplate(templ, cv::Size(240, 200), center, 71.0);
    const auto result = matcher.match(image);
    expect(result.valid(), "masked template should produce a result");
    if (result) {
        expect(angleError(result.angle_deg, 71.0) <= 1.01,
               "masked-template angle should be accurate");
        expect(cv::norm(result.center - center) <= 1.1,
               "masked-template center should be accurate");
    }
}

void partialAngleRangeRegression() {
    const cv::Mat templ = makeTemplate();
    orient_match::MatcherOptions options = testOptions();
    options.angle_start_deg = 350.0;
    options.angle_extent_deg = 20.0;
    const orient_match::Matcher matcher(templ, options);
    const cv::Point2d center(108.5, 113.5);
    const cv::Mat image = placeTemplate(templ, cv::Size(240, 200), center, 3.0);
    const auto result = matcher.match(image);
    expect(result.valid(), "wrapped partial angle range should produce a result");
    if (result) {
        expect(angleError(result.angle_deg, 3.0) <= 1.01,
               "wrapped partial angle range should find the correct angle");
        expect(cv::norm(result.center - center) <= 1.1,
               "wrapped partial angle range should find the correct center");
    }
}

void statusAndValidationRegression() {
    const cv::Mat templ = makeTemplate();
    const orient_match::Matcher matcher(templ, testOptions());
    const auto too_small = matcher.match(cv::Mat(50, 50, CV_8U, cv::Scalar(0)));
    expect(too_small.status == orient_match::MatchStatus::template_larger_than_image,
           "small image should report template_larger_than_image");

    bool threw = false;
    try {
        const cv::Mat color(80, 80, CV_8UC3, cv::Scalar(0, 0, 0));
        (void)matcher.match(color);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    expect(threw, "multi-channel input should be rejected");

    threw = false;
    try {
        const cv::Mat flat_template(32, 32, CV_8U, cv::Scalar(42));
        const orient_match::Matcher invalid_matcher(flat_template);
        (void)invalid_matcher;
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    expect(threw, "flat template with no gradient energy should be rejected");

    const cv::Mat field = orient_match::orientationField(templ);
    expect(field.type() == CV_32FC2 && field.size() == templ.size(),
           "orientationField should return CV_32FC2 at the input size");
}

void determinismRegression() {
    const cv::Mat templ = makeTemplate();
    const orient_match::Matcher matcher(templ, testOptions());
    const cv::Mat image = placeTemplate(templ, cv::Size(240, 200), {137.5, 88.5}, 123.0);

#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const auto first = matcher.match(image);
#ifdef _OPENMP
    omp_set_num_threads(std::max(1, saved_threads));
#endif
    const auto second = matcher.match(image);
    expect(first.status == second.status && first.center == second.center &&
               first.angle_deg == second.angle_deg && first.score == second.score,
           "result should be deterministic across repeated/thread-count runs");
}

}  // namespace

int main() {
    try {
        poseRegression();
        maskRegression();
        partialAngleRangeRegression();
        statusAndValidationRegression();
        determinismRegression();
    } catch (const std::exception &error) {
        std::cerr << "Unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All OrientMatch tests passed.\n";
    return 0;
}
