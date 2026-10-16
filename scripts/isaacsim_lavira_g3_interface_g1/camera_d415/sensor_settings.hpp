#pragma once
// 两个取图程序共用的只读启动参数快照。
#include <librealsense2/rs.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace camera_settings {

// SDK错误信息可能有换行，需要完整JSON转义，不能直接拼进引号。
inline std::string quoted_json(const std::string& text) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    out << '"';
    return out.str();
}

// pipeline启动后顺序读取；自动曝光值仅代表查询时刻。
inline std::string read_startup(const rs2::pipeline_profile& profile) {
    const auto device = profile.get_device();
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\"snapshot_stage\":\"after_pipeline_start_before_warmup\","
        << "\"note\":\"Read-only sequential startup snapshot; automatic exposure can change afterwards; robot epoch clock may be incorrect\","
        << "\"robot_epoch_s\":" << std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count()
        << ",\"sdk_version\":\"" << RS2_API_VERSION_STR << "\",\"serial\":"
        << quoted_json(device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER))
        << ",\"firmware\":" << quoted_json(device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION))
        << ",\"active_streams\":[";
    bool first = true;
    for (const auto& stream : profile.get_streams()) {
        out << (first ? "" : ",") << "{\"type\":" << quoted_json(rs2_stream_to_string(stream.stream_type()))
            << ",\"format\":" << quoted_json(rs2_format_to_string(stream.format()))
            << ",\"fps\":" << stream.fps();
        if (auto video = stream.as<rs2::video_stream_profile>())
            out << ",\"width\":" << video.width() << ",\"height\":" << video.height();
        out << '}';
        first = false;
    }
    out << "],\"sensors\":[";
    struct Option { rs2_option id; const char* key; };
    const Option options[] = {
        {RS2_OPTION_ENABLE_AUTO_EXPOSURE, "enable_auto_exposure"},
        {RS2_OPTION_EXPOSURE, "exposure"},
        {RS2_OPTION_VISUAL_PRESET, "visual_preset"},
        {RS2_OPTION_EMITTER_ENABLED, "emitter_enabled"},
        {RS2_OPTION_LASER_POWER, "laser_power"},
    };
    first = true;
    for (const auto& sensor : device.query_sensors()) {
        const std::string name = sensor.get_info(RS2_CAMERA_INFO_NAME);
        out << (first ? "" : ",") << "{\"name\":" << quoted_json(name) << ",\"options\":{";
        first = false;
        bool first_option = true;
        for (const auto& option : options) {
            out << (first_option ? "" : ",") << quoted_json(option.key) << ':';
            first_option = false;
            // 单项不支持/查询失败不能伪装成数值0，也不因此阻止正常取图。
            std::string record, display;
            try {
                if (!sensor.supports(option.id)) {
                    record = "{\"status\":\"unsupported\",\"value\":null}";
                    display = "unsupported";
                } else {
                    const float value = sensor.get_option(option.id);
                    if (!std::isfinite(value)) throw std::runtime_error("Non-finite SDK option value");
                    std::ostringstream entry;
                    entry << std::setprecision(9) << "{\"status\":\"ok\",\"value\":" << value;
                    std::ostringstream shown;
                    shown << value;
                    // 枚举选项有描述时一并保留，例如High Accuracy；描述不是必需字段。
                    try {
                        if (const char* label = sensor.get_option_value_description(option.id, value)) {
                            entry << ",\"value_description\":" << quoted_json(label);
                            shown << " (" << label << ')';
                        }
                    } catch (const rs2::error&) {}
                    entry << '}';
                    record = entry.str();
                    display = shown.str();
                }
            } catch (const std::exception& error) {
                record = "{\"status\":\"read_error\",\"value\":null,\"error\":" + quoted_json(error.what()) + '}';
                display = "read_error: " + std::string(error.what());
            }
            out << record;
            std::cout << "[CAMERA SETTINGS] " << name << '.' << option.key << '=' << display << '\n';
        }
        out << "}}";
    }
    out << "]}";
    std::cout << std::flush;
    return out.str();
}

// 每次启动独立保存，以单调时间和PID命名，避免依赖机器人系统日期。
inline std::filesystem::path save_startup(const std::string& json, const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = directory / ("startup-" + std::to_string(tick) + "-" + std::to_string(getpid()) + ".json");
    std::ofstream file(path);
    file << json << '\n';
    file.close();
    if (!file) throw std::runtime_error("Cannot save camera startup settings: " + path.string());
    std::cout << "[CAMERA SETTINGS] saved " << std::filesystem::absolute(path) << std::endl;
    return path;
}
}  // namespace camera_settings
