// 官方SDK文件取图：保存最后一组RGB、对齐深度和标定。
#include <librealsense2/rs.hpp>
#include "sensor_settings.hpp"
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

// 保存K、畸变模型和系数。
void intrinsics_json(std::ostream& out, const rs2_intrinsics& k) {
    out << "{\"width\":" << k.width << ",\"height\":" << k.height
        << ",\"K\":[[" << k.fx << ",0," << k.ppx << "],[0," << k.fy
        << ',' << k.ppy << "],[0,0,1]],\"distortion_model\":"
        << std::quoted(rs2_distortion_to_string(k.model)) << ",\"coeffs\":[";
    for (int i = 0; i < 5; ++i) out << (i ? "," : "") << k.coeffs[i];
    out << "]}";
}

int main(int argc, char** argv) {
    try {
        bool rgb_only = false;
        int count = 30;
        fs::path output = "capture";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "g1_d435i_probe [--rgb-only] [--frames N] [--output NEW_DIRECTORY]\n"
                          << "640x480 15fps; warms up 15 frames, captures N frames, saves last.\n";
                return 0;
            } else if (arg == "--rgb-only") rgb_only = true;
            else if (arg == "--output" && i + 1 < argc) output = argv[++i];
            else if (arg == "--frames" && i + 1 < argc) {
                const std::string value = argv[++i];
                size_t used = 0;
                count = std::stoi(value, &used);
                if (used != value.size() || count < 1 || count > 100000)
                    throw std::runtime_error("frames must be 1..100000");
            } else throw std::runtime_error("Unknown/incomplete argument: " + arg);
        }
        if (fs::exists(output)) throw std::runtime_error("Output directory already exists; choose a new one");
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        rs2::context context;
        const auto devices = context.query_devices();
        if (devices.size() != 1)
            throw std::runtime_error("Expected exactly one RealSense device; found " + std::to_string(devices.size()));
        const auto device = devices[0];
        const std::string serial = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
        std::cout << "Device: " << device.get_info(RS2_CAMERA_INFO_NAME)
                  << " serial=" << serial << " firmware="
                  << device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) << std::endl;
        rs2::pipeline pipeline(context);
        rs2::config config;
        config.enable_device(serial);
        config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 15);
        if (!rgb_only) config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 15);
        const auto profile = pipeline.start(config);
        const auto startup_settings = camera_settings::read_startup(profile);
        rs2::align align_to_color(RS2_STREAM_COLOR);
        rs2::frameset frames;
        const auto start = std::chrono::steady_clock::now();
        // 连续读取，最后保存一组；--frames 150 可观察约10秒的数据流。
        for (int i = 0; i < count + 15 && !interrupted; ++i) {
            frames = pipeline.wait_for_frames(5000);
            if (i >= 15 && (i - 14) % 15 == 0)
                std::cout << "Captured " << i - 14 << '/' << count << std::endl;
        }
        if (interrupted) { pipeline.stop(); return 130; }
        const double received_epoch_s = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (!rgb_only) frames = align_to_color.process(frames);
        const auto color = frames.get_color_frame();
        const auto depth = frames.get_depth_frame();
        if (!color || (!rgb_only && !depth)) throw std::runtime_error("Required frame missing");
        if (!rgb_only && (color.get_width() != depth.get_width() || color.get_height() != depth.get_height()))
            throw std::runtime_error("Aligned depth shape mismatch");
        fs::create_directories(output);
        std::ofstream rgb(output / "rgb.ppm", std::ios::binary);
        rgb << "P6\n" << color.get_width() << ' ' << color.get_height() << "\n255\n";
        const auto* bytes = static_cast<const char*>(color.get_data());
        for (int y = 0; y < color.get_height(); ++y)
            rgb.write(bytes + y * color.get_stride_in_bytes(), color.get_width() * 3);
        rgb.close();
        if (!rgb) throw std::runtime_error("RGB file write failed");
        size_t valid = 0;
        if (!rgb_only) {
            std::ofstream pgm(output / "depth_aligned_z16.pgm", std::ios::binary);
            pgm << "P5\n" << depth.get_width() << ' ' << depth.get_height() << "\n65535\n";
            const auto* data = static_cast<const uint8_t*>(depth.get_data());
            for (int y = 0; y < depth.get_height(); ++y) {
                const auto* row = reinterpret_cast<const uint16_t*>(data + y * depth.get_stride_in_bytes());
                for (int x = 0; x < depth.get_width(); ++x) {
                    const uint16_t z = row[x];
                    valid += (z != 0);
                    // PGM 16位数据按大端存储，保留原始Z16，不做彩色化。
                    pgm.put(static_cast<char>(z >> 8));
                    pgm.put(static_cast<char>(z & 255));
                }
            }
            pgm.close();
            if (!pgm) throw std::runtime_error("Depth file write failed");
        }
        std::ofstream meta(output / "metadata.json");
        meta << std::setprecision(17) << "{\n\"sdk_version\":\"" << RS2_API_VERSION_STR
             << "\",\"serial\":" << std::quoted(serial)
             << ",\"device_name\":" << std::quoted(device.get_info(RS2_CAMERA_INFO_NAME))
             << ",\"firmware\":" << std::quoted(device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION))
             << ",\"color_encoding\":\"RGB8\",\"color_frame_number\":" << color.get_frame_number()
             << ",\"color_timestamp_ms\":" << color.get_timestamp()
             << ",\"color_timestamp_domain\":" << std::quoted(rs2_timestamp_domain_to_string(color.get_frame_timestamp_domain()))
             << ",\"capture_host_epoch_s\":" << received_epoch_s
             << ",\"color_intrinsics\":";
        intrinsics_json(meta, color.get_profile().as<rs2::video_stream_profile>().get_intrinsics());
        if (!rgb_only) {
            const auto original_depth = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            const auto extrinsic = original_depth.get_extrinsics_to(profile.get_stream(RS2_STREAM_COLOR));
            const double fraction = double(valid) / (depth.get_width() * depth.get_height());
            meta << ",\"depth_aligned_to\":\"color\",\"depth_scale_m\":" << depth.get_units()
                 << ",\"valid_depth_fraction\":" << fraction
                 << ",\"depth_timestamp_ms\":" << depth.get_timestamp()
                 << ",\"depth_timestamp_domain\":" << std::quoted(rs2_timestamp_domain_to_string(depth.get_frame_timestamp_domain()))
                 << ",\"depth_frame_number\":" << depth.get_frame_number()
                 << ",\"original_depth_intrinsics\":";
            intrinsics_json(meta, original_depth.get_intrinsics());
            meta << ",\"aligned_depth_intrinsics\":";
            intrinsics_json(meta, depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics());
            meta << ",\"depth_to_color\":{\"rotation_column_major\":[";
            for (int i = 0; i < 9; ++i) meta << (i ? "," : "") << extrinsic.rotation[i];
            meta << "],\"translation_m\":[";
            for (int i = 0; i < 3; ++i) meta << (i ? "," : "") << extrinsic.translation[i];
            meta << "]}";
            std::cout << "Valid depth fraction=" << fraction << std::endl;
        }
        // SDK只能给出相机内部标定，不能代替相机到机器人机体的安装标定。
        meta << ",\"camera_to_robot_extrinsics\":null,\"startup_settings\":" << startup_settings << "}\n";
        meta.close();
        if (!meta) throw std::runtime_error("Metadata file write failed");
        pipeline.stop();
        if (!rgb_only && valid == 0) throw std::runtime_error("All depth values invalid; diagnostic files saved");
        std::cout << "Saved " << fs::absolute(output) << "; elapsed="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << "s\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Capture failed: " << error.what() << '\n';
        return interrupted ? 130 : 1;
    }
}
