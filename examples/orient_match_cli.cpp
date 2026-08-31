#include "orient_match/orient_match.hpp"

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

namespace {

void usage(const char *program) {
    std::cerr << "Usage: " << program
              << " IMAGE TEMPLATE [ANGLE_START_DEG ANGLE_EXTENT_DEG]\n";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3 && argc != 5) {
        usage(argv[0]);
        return 2;
    }

    const cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    const cv::Mat templ = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    if (image.empty() || templ.empty()) {
        std::cerr << "Failed to read IMAGE or TEMPLATE.\n";
        return 2;
    }

    try {
        orient_match::MatcherOptions options;
        if (argc == 5) {
            options.angle_start_deg = std::stod(argv[3]);
            options.angle_extent_deg = std::stod(argv[4]);
        }

        const orient_match::Matcher matcher(templ, options);
        const orient_match::MatchResult result = matcher.match(image);
        if (!result) {
            std::cerr << "Match failed: " << orient_match::statusMessage(result.status) << '\n';
            return 1;
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "{\"center_x\":" << result.center.x
                  << ",\"center_y\":" << result.center.y
                  << ",\"angle_deg\":" << result.angle_deg
                  << ",\"score\":" << result.score << "}\n";
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
