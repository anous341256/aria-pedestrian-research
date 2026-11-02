// Video-input adapter for BoT-SORT's unmasked OpenCV VideoStab GMC generator.
// Upstream: NirAharon/BoT-SORT @ 251985436d6712aaf682aaaf5f71edb4987224bd
// Source: VideoCameraCorrection/VideoCameraCorrection/main.cpp (MIT license).
// Preserved: GFTT(4000), RansacL2(MM_SIMILARITY), KeypointBasedMotionEstimator,
// BGR frames, INTER_LINEAR downscale=2, estimate(previous,current), translation
// rescale=2, previous successful/reused warp on failure, zero-based frame rows.

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videostab.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
constexpr const char* kCommit = "251985436d6712aaf682aaaf5f71edb4987224bd";
constexpr int kDownscale = 2;

std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                             << std::setfill('0') << static_cast<int>(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}

struct Estimate {
    cv::Mat warp;
    std::string status;
    bool ok = false;
};

class Cmc {
public:
    Cmc() {
        auto estimator = cv::makePtr<cv::videostab::MotionEstimatorRansacL2>(
            cv::videostab::MM_SIMILARITY);
        keypoints_ = cv::makePtr<cv::videostab::KeypointBasedMotionEstimator>(estimator);
        keypoints_->setDetector(cv::GFTTDetector::create(4000));
    }

    Estimate process(const cv::Mat& full_frame) {
        Estimate result{previous_warp_.clone(), "initial_identity", false};
        if (full_frame.empty()) {
            result.status = "fallback_decode";
            // Do not estimate across a missing decoded frame as if consecutive.
            previous_frame_.release();
            seen_frame_ = true;
            return result;
        }
        if (full_frame.cols < 2 || full_frame.rows < 2)
            throw std::runtime_error("Input dimensions must both be >= 2");
        cv::Mat frame;
        cv::resize(full_frame, frame,
                   cv::Size(full_frame.cols / kDownscale, full_frame.rows / kDownscale),
                   0, 0, cv::INTER_LINEAR);
        if (previous_frame_.empty()) {
            result.status = seen_frame_ ? "fallback_no_previous_decoded_frame"
                                        : "initial_identity";
        } else {
            try {
                bool ok = false;
                cv::Mat warp = keypoints_->estimate(previous_frame_, frame, &ok);
                if (ok && warp.rows == 3 && warp.cols == 3 && cv::checkRange(warp)) {
                    warp.convertTo(warp, CV_32F);
                    warp.at<float>(0, 2) *= kDownscale;
                    warp.at<float>(1, 2) *= kDownscale;
                    result = {warp, "estimated", true};
                } else {
                    result.status = "fallback_estimator";
                }
            } catch (const cv::Exception&) {
                result.status = "fallback_opencv_exception";
            }
        }
        frame.copyTo(previous_frame_);
        result.warp.copyTo(previous_warp_);
        seen_frame_ = true;
        return result;
    }

private:
    cv::Ptr<cv::videostab::KeypointBasedMotionEstimator> keypoints_;
    cv::Mat previous_frame_;
    cv::Mat previous_warp_ = cv::Mat::eye(3, 3, CV_32F);
    bool seen_frame_ = false;
};

void prepare_output(const fs::path& path) {
    if (fs::exists(path)) throw std::runtime_error("Refusing to overwrite: " + path.string());
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
}

long run_video(const fs::path& video, const fs::path& output,
               const fs::path& quality, long max_frames) {
    if (!fs::is_regular_file(video)) throw std::runtime_error("Video not found: " + video.string());
    cv::VideoCapture capture(video.string());
    if (!capture.isOpened()) throw std::runtime_error("Cannot open video: " + video.string());
    const double raw_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    const long declared_count = std::isfinite(raw_count) && raw_count > 0
                                    ? static_cast<long>(std::llround(raw_count)) : 0;
    const double fps = capture.get(cv::CAP_PROP_FPS);
    const long limit = max_frames > 0 && declared_count > 0
                         ? std::min(max_frames, declared_count)
                         : (max_frames > 0 ? max_frames : declared_count);
    const fs::path metadata = output.string() + ".meta.json";
    if (fs::absolute(video).lexically_normal() == fs::absolute(output).lexically_normal()
        || fs::absolute(output).lexically_normal() == fs::absolute(quality).lexically_normal())
        throw std::runtime_error("Input/output/quality paths must differ");
    prepare_output(output);
    prepare_output(quality);
    prepare_output(metadata);
    std::ofstream matrices(output), qa(quality);
    if (!matrices || !qa) throw std::runtime_error("Cannot open output files");
    matrices << std::fixed << std::setprecision(6);
    qa << "frame\tstatus\testimation_ok\telapsed_ms\n";
    Cmc cmc;
    long count = 0, failures = 0, decode_failures = 0;
    int width = 0, height = 0;
    const auto start = std::chrono::steady_clock::now();
    for (long index = 0; limit == 0 || index < limit; ++index) {
        cv::Mat frame;
        const bool decoded = capture.read(frame) && !frame.empty();
        if (!decoded && declared_count == 0) break;  // Unknown-length stream reached EOF.
        const auto frame_start = std::chrono::steady_clock::now();
        if (!decoded) {
            ++decode_failures;
            capture.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(index + 1));
        } else {
            if (width != 0 && (width != frame.cols || height != frame.rows))
                throw std::runtime_error("Video resolution changed within sequence");
            width = frame.cols;
            height = frame.rows;
        }
        const Estimate result = cmc.process(frame);
        if (result.status != "estimated" && result.status != "initial_identity") ++failures;
        matrices << index << '\t';
        for (int row = 0; row < 2; ++row)
            for (int col = 0; col < 3; ++col)
                matrices << result.warp.at<float>(row, col) << '\t';
        matrices << '\n';
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - frame_start).count();
        qa << index << '\t' << result.status << '\t' << (result.ok ? 1 : 0)
           << '\t' << std::fixed << std::setprecision(3) << ms << '\n';
        ++count;
        if (count % 300 == 0) std::cerr << "Processed " << count << " frames\n";
    }
    matrices.close();
    qa.close();
    if (!matrices || !qa) throw std::runtime_error("Output write failed");
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::ofstream meta(metadata);
    meta << "{\n  \"upstream_repository\": \"https://github.com/NirAharon/BoT-SORT\",\n"
         << "  \"upstream_commit\": " << json_string(kCommit) << ",\n"
         << "  \"upstream_file\": \"VideoCameraCorrection/VideoCameraCorrection/main.cpp\",\n"
         << "  \"opencv_version\": " << json_string(CV_VERSION) << ",\n"
         << "  \"input_video\": " << json_string(fs::absolute(video).string()) << ",\n"
         << "  \"output_matrices\": " << json_string(fs::absolute(output).string()) << ",\n"
         << "  \"quality_sidecar\": " << json_string(fs::absolute(quality).string()) << ",\n"
         << "  \"frame_index_base\": 0,\n  \"direction\": \"previous_frame_to_current_frame\",\n"
         << "  \"estimator\": \"KeypointBasedMotionEstimator(MotionEstimatorRansacL2(MM_SIMILARITY))\",\n"
         << "  \"detector\": \"GFTTDetector::create(4000)\",\n"
         << "  \"downscale\": 2,\n  \"resize\": \"INTER_LINEAR\",\n"
         << "  \"masking\": false,\n  \"estimator_failure_fallback\": \"previous_warp\",\n"
         << "  \"decode_failure_policy\": \"emit previous warp; skip nominal frame; reset frame predecessor\",\n"
         << "  \"reported_frame_count\": " << declared_count << ",\n"
         << "  \"requested_max_frames\": " << max_frames << ",\n"
         << "  \"written_frames\": " << count << ",\n  \"fallback_frames\": " << failures << ",\n"
         << "  \"decode_failures\": " << decode_failures << ",\n"
         << "  \"width\": " << width << ",\n  \"height\": " << height << ",\n"
         << "  \"reported_fps\": " << (std::isfinite(fps) ? fps : 0) << ",\n"
         << "  \"elapsed_seconds\": " << seconds << "\n}\n";
    meta.close();
    if (!meta) throw std::runtime_error("Metadata write failed");
    if (count == 0) throw std::runtime_error("No video frames processed");
    std::cout << "Wrote " << count << " rows to " << output << "; fallback=" << failures
              << ", decode_failures=" << decode_failures << ", seconds=" << seconds << '\n';
    return count;
}

void self_test(const fs::path& directory) {
    fs::create_directories(directory);
    const fs::path video = directory / "translation.avi";
    prepare_output(video);
    const cv::Size size(640, 480);
    cv::Mat base(size, CV_8UC3);
    cv::RNG rng(42);
    rng.fill(base, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(base, base, cv::Size(3, 3), 0.7);
    cv::VideoWriter writer(video.string(), cv::VideoWriter::fourcc('F', 'F', 'V', '1'), 30, size);
    if (!writer.isOpened()) throw std::runtime_error("Cannot create lossless FFV1 smoke-test video");
    for (int index = 0; index < 8; ++index) {
        cv::Mat frame;
        cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, 8 * index, 0, 1, -4 * index);
        cv::warpAffine(base, frame, transform, size, cv::INTER_LINEAR, cv::BORDER_REFLECT);
        writer.write(frame);
    }
    writer.release();
    const fs::path output = directory / "GMC-translation.txt";
    if (run_video(video, output, directory / "translation.quality.tsv", 0) != 8)
        throw std::runtime_error("Smoke test: expected exactly 8 frame rows");
    std::ifstream rows(output);
    std::string line;
    int count = 0;
    double max_error = 0;
    while (std::getline(rows, line)) {
        std::istringstream input(line);
        long index;
        double a, b, tx, c, d, ty;
        if (!(input >> index >> a >> b >> tx >> c >> d >> ty) || index != count)
            throw std::runtime_error("Smoke test: malformed frame row");
        std::string extra;
        if (input >> extra) throw std::runtime_error("Smoke test: more than 7 columns");
        const double expected_x = count == 0 ? 0 : 8;
        const double expected_y = count == 0 ? 0 : -4;
        max_error = std::max(max_error, std::max(std::abs(tx - expected_x), std::abs(ty - expected_y)));
        if (std::abs(a - 1) > .01 || std::abs(b) > .01 || std::abs(c) > .01
            || std::abs(d - 1) > .01 || max_error > .35)
            throw std::runtime_error("Smoke test: transform direction or full-resolution scale failed");
        ++count;
    }
    if (count != 8) throw std::runtime_error("Smoke test: wrong row count");
    Cmc fallback;
    fallback.process(base);
    cv::Mat shifted;
    cv::Mat translation = (cv::Mat_<double>(2, 3) << 1, 0, 8, 0, 1, -4);
    cv::warpAffine(base, shifted, translation, size, cv::INTER_LINEAR, cv::BORDER_REFLECT);
    const Estimate good = fallback.process(shifted);
    // A transition from texture to black can still yield an accepted optical-flow
    // estimate. Two consecutive black frames deterministically lack GFTT corners.
    const Estimate before_failure = fallback.process(cv::Mat::zeros(size, CV_8UC3));
    const Estimate failed = fallback.process(cv::Mat::zeros(size, CV_8UC3));
    if (!good.ok || failed.ok || cv::norm(failed.warp - before_failure.warp, cv::NORM_INF) > 1e-6)
        throw std::runtime_error("Smoke test: previous-warp fallback failed");
    const Estimate decode_failed = fallback.process(cv::Mat());
    if (decode_failed.status != "fallback_decode"
        || cv::norm(decode_failed.warp - before_failure.warp, cv::NORM_INF) > 1e-6)
        throw std::runtime_error("Smoke test: decode fallback failed");
    std::cout << "SELF_TEST_PASS: 8 rows, first identity, previous->current dx=8 dy=-4; max_translation_error="
              << max_error << " px; estimation and decode fallback preserved previous warp\n";
}

int main(int argc, char** argv) {
    try {
        fs::path video, output, quality, test_directory;
        long max_frames = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "Usage: cmc_videostab --video VIDEO --output GMC.txt [--quality QA.tsv] [--max-frames N]\n"
                          << "       cmc_videostab --self-test NEW_DIRECTORY\n";
                return 0;
            }
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            if (arg == "--video") video = argv[++i];
            else if (arg == "--output") output = argv[++i];
            else if (arg == "--quality") quality = argv[++i];
            else if (arg == "--self-test") test_directory = argv[++i];
            else if (arg == "--max-frames") {
                max_frames = std::stol(argv[++i]);
                if (max_frames < 1) throw std::runtime_error("--max-frames must be positive");
            } else throw std::runtime_error("Unknown argument: " + arg);
        }
        if (!test_directory.empty()) self_test(test_directory);
        else {
            if (video.empty() || output.empty()) throw std::runtime_error("--video and --output are required");
            if (quality.empty()) quality = output.string() + ".quality.tsv";
            run_video(video, output, quality, max_frames);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
