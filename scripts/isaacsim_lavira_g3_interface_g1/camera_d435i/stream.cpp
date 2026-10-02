// 机器人端RGB-D服务：逐帧对齐深度，只保留最新观测供TCP客户端请求。
#include <librealsense2/rs.hpp>
#include "sensor_settings.hpp"
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

// 文件描述符采用 RAII，异常或 Ctrl+C 退出时也能关闭连接。
struct Socket {
    int fd;
    explicit Socket(int value) : fd(value) {
        if (fd < 0) throw std::runtime_error(std::strerror(errno));
    }
    ~Socket() { ::close(fd); }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
};

struct Packet {
    uint64_t sequence;
    Clock::time_point acquired;
    std::string metadata;
    std::vector<uint8_t> rgb, depth;
};

struct Latest {
    std::mutex mutex;
    std::condition_variable changed;
    std::shared_ptr<const Packet> packet;
    std::string error;
    std::atomic<bool> running{true};
};

void intrinsics_json(std::ostream& out, const rs2_intrinsics& k) {
    out << "{\"width\":" << k.width << ",\"height\":" << k.height
        << ",\"K\":[[" << k.fx << ",0," << k.ppx << "],[0," << k.fy
        << ',' << k.ppy << "],[0,0,1]],\"distortion_model\":"
        << std::quoted(rs2_distortion_to_string(k.model)) << ",\"coeffs\":[";
    for (int i = 0; i < 5; ++i) out << (i ? "," : "") << k.coeffs[i];
    out << "]}";
}

// 接收 NEXT 后等待随后采集的帧，避免把模型推理期间缓存的旧图当作新观测。
// SDK时间戳用于配对，新鲜度使用单调时钟。
void capture(Latest& latest, const std::string& serial, const std::string& settings_dir) {
    try {
        rs2::pipeline pipeline;
        rs2::config config;
        config.enable_device(serial);
        config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 15);
        config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 15);
        const auto profile = pipeline.start(config);
        const auto startup_settings = camera_settings::read_startup(profile);
        camera_settings::save_startup(startup_settings, settings_dir);
        const auto original_depth = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        const auto extrinsics = original_depth.get_extrinsics_to(profile.get_stream(RS2_STREAM_COLOR));
        rs2::align align(RS2_STREAM_COLOR);
        uint64_t sequence = 0;
        int warmup = 15;
        auto last_received = Clock::now();
        while (latest.running && !interrupted) {
            rs2::frameset frames;
            if (!pipeline.try_wait_for_frames(&frames, 100)) {
                if (Clock::now() - last_received > std::chrono::seconds(2))
                    throw std::runtime_error("Camera produced no frames for 2 seconds");
                continue;
            }
            last_received = Clock::now();
            if (warmup > 0) { --warmup; continue; }
            // 帧年龄从SDK取出frameset时开始计时。
            const auto acquired = last_received;
            frames = align.process(frames);
            const auto color = frames.get_color_frame();
            const auto depth = frames.get_depth_frame();
            if (!color || !depth) throw std::runtime_error("RGB or depth frame missing");
            if (color.get_width() != depth.get_width() || color.get_height() != depth.get_height())
                throw std::runtime_error("Aligned depth shape mismatch");
            auto packet = std::make_shared<Packet>();
            packet->sequence = ++sequence;
            packet->acquired = acquired;
            const int w = color.get_width(), h = color.get_height();
            packet->rgb.resize(w * h * 3);
            packet->depth.resize(w * h * 2);
            const auto* rgb = static_cast<const uint8_t*>(color.get_data());
            const auto* z16 = static_cast<const uint8_t*>(depth.get_data());
            for (int y = 0; y < h; ++y) {
                std::memcpy(packet->rgb.data() + y * w * 3, rgb + y * color.get_stride_in_bytes(), w * 3);
                const auto* row = reinterpret_cast<const uint16_t*>(z16 + y * depth.get_stride_in_bytes());
                for (int x = 0; x < w; ++x) {
                    // Z16网络字节序为大端，Python用 >u2 解码。
                    packet->depth[2 * (y * w + x)] = static_cast<uint8_t>(row[x] >> 8);
                    packet->depth[2 * (y * w + x) + 1] = static_cast<uint8_t>(row[x] & 255);
                }
            }
            std::ostringstream meta;
            meta << std::setprecision(17) << "{\"protocol\":1,\"sequence\":" << sequence
                 << ",\"serial\":" << std::quoted(serial) << ",\"sdk_version\":\"" << RS2_API_VERSION_STR
                 << "\",\"width\":" << w << ",\"height\":" << h
                 << ",\"rgb_encoding\":\"RGB8\",\"depth_encoding\":\"Z16_BE\",\"depth_aligned_to\":\"color\""
                 << ",\"rgb_bytes\":" << packet->rgb.size() << ",\"depth_bytes\":" << packet->depth.size()
                 << ",\"depth_scale_m\":" << depth.get_units()
                 << ",\"color_frame_number\":" << color.get_frame_number()
                 << ",\"depth_frame_number\":" << depth.get_frame_number()
                 << ",\"color_timestamp_ms\":" << color.get_timestamp()
                 << ",\"depth_timestamp_ms\":" << depth.get_timestamp()
                 << ",\"color_timestamp_domain\":" << std::quoted(rs2_timestamp_domain_to_string(color.get_frame_timestamp_domain()))
                 << ",\"depth_timestamp_domain\":" << std::quoted(rs2_timestamp_domain_to_string(depth.get_frame_timestamp_domain()))
                 << ",\"color_intrinsics\":";
            intrinsics_json(meta, color.get_profile().as<rs2::video_stream_profile>().get_intrinsics());
            meta << ",\"aligned_depth_intrinsics\":";
            intrinsics_json(meta, depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics());
            meta << ",\"depth_to_color\":{\"rotation_column_major\":[";
            for (int i = 0; i < 9; ++i) meta << (i ? "," : "") << extrinsics.rotation[i];
            meta << "],\"translation_m\":[";
            for (int i = 0; i < 3; ++i) meta << (i ? "," : "") << extrinsics.translation[i];
            meta << "]},\"camera_to_robot_extrinsics\":null,\"startup_settings\":" << startup_settings;
            // 末尾的 age 字段在发送时加入，因此这里暂不关闭 JSON 对象。
            packet->metadata = meta.str();
            {
                std::lock_guard<std::mutex> lock(latest.mutex);
                latest.packet = std::move(packet);
            }
            latest.changed.notify_all();
            if (sequence == 1) std::cout << "RGB-D ready: aligned 640x480 at 15fps; start laptop preview now\n" << std::flush;
        }
        pipeline.stop();
    } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(latest.mutex);
        latest.error = error.what();
    }
    latest.running = false;
    latest.changed.notify_all();
}

void send_all(int fd, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    while (size && !interrupted) {
        const auto sent = ::send(fd, bytes, size, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) throw std::runtime_error("Client disconnected or send timed out");
        bytes += sent;
        size -= sent;
    }
}

void serve(int fd, Latest& latest) {
    // 一个客户端；预览和 VLN 顺序使用。超时后释放连接，客户端可以重新运行。
    timeval timeout{2, 0};
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0)
        throw std::runtime_error("Cannot set socket timeout");
    while (latest.running && !interrupted) {
        pollfd readable{fd, POLLIN, 0};
        const int ready = ::poll(&readable, 1, 100);
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0) throw std::runtime_error("Socket poll failed");
        if (!ready) continue;
        char command[4];
        size_t used = 0;
        const auto request_started = Clock::now();
        while (used < sizeof(command) && !interrupted) {
            pollfd input{fd, POLLIN, 0};
            const int count = ::poll(&input, 1, 100);
            if (count > 0) {
                const auto received = ::recv(fd, command + used, sizeof(command) - used, 0);
                if (received <= 0) throw std::runtime_error("Client disconnected");
                used += received;
            } else if (count < 0 && errno != EINTR) throw std::runtime_error("Socket poll failed");
            if (Clock::now() - request_started > std::chrono::seconds(2))
                throw std::runtime_error("Incomplete NEXT request");
        }
        if (interrupted) break;
        if (std::memcmp(command, "NEXT", 4) != 0) throw std::runtime_error("Expected NEXT request");
        std::shared_ptr<const Packet> packet;
        {
            std::unique_lock<std::mutex> lock(latest.mutex);
            const uint64_t previous = latest.packet ? latest.packet->sequence : 0;
            if (!latest.changed.wait_for(lock, std::chrono::seconds(2), [&] {
                    return !latest.running || (latest.packet && latest.packet->sequence > previous
                        && latest.packet->acquired >= request_started);
                })) throw std::runtime_error("Timed out waiting for a new RGB-D frame");
            if (!latest.running) return;
            packet = latest.packet;
        }
        const double age = std::chrono::duration<double>(Clock::now() - packet->acquired).count();
        std::ostringstream header;
        header << packet->metadata << ",\"server_frame_age_s\":" << std::setprecision(17) << age << '}';
        const auto json = header.str();
        const uint32_t size_be = htonl(static_cast<uint32_t>(json.size()));
        send_all(fd, &size_be, sizeof(size_be));
        send_all(fd, json.data(), json.size());
        send_all(fd, packet->rgb.data(), packet->rgb.size());
        send_all(fd, packet->depth.data(), packet->depth.size());
    }
}

int main(int argc, char** argv) {
    try {
        std::string bind = "192.168.123.164", serial = "344422072128";
        std::string settings_dir = "camera_logs";
        int port = 8765;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "g1_d435i_stream [--bind IP] [--port 8765] [--serial SERIAL] [--settings-dir DIRECTORY]\n"
                          << "RGB8 + aligned Z16, 640x480 at 15fps; camera only, no robot motion.\n";
                return 0;
            }
            if (i + 1 >= argc) throw std::runtime_error("Missing argument value");
            if (arg == "--bind") bind = argv[++i];
            else if (arg == "--serial") serial = argv[++i];
            else if (arg == "--settings-dir") settings_dir = argv[++i];
            else if (arg == "--port") {
                size_t used = 0;
                const std::string value = argv[++i];
                port = std::stoi(value, &used);
                if (used != value.size() || port < 1 || port > 65535) throw std::runtime_error("Invalid port");
            } else throw std::runtime_error("Unknown argument: " + arg);
        }
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        Socket listener(::socket(AF_INET, SOCK_STREAM, 0));
        const int reuse = 1;
        setsockopt(listener.fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<uint16_t>(port));
        if (inet_pton(AF_INET, bind.c_str(), &address.sin_addr) != 1) throw std::runtime_error("Invalid bind IPv4");
        if (::bind(listener.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0
            || ::listen(listener.fd, 1) != 0) throw std::runtime_error(std::strerror(errno));
        Latest latest;
        std::thread producer(capture, std::ref(latest), serial, settings_dir);
        std::cout << "Camera stream listening on " << bind << ':' << port
                  << "; serial=" << serial << "; warming up 15 frames; Ctrl+C exits\n" << std::flush;
        // 主线程任何异常都先通知采集线程退出，再 join，避免 std::thread 析构终止进程。
        try {
            while (latest.running && !interrupted) {
                pollfd incoming{listener.fd, POLLIN, 0};
                const int ready = ::poll(&incoming, 1, 100);
                if (ready < 0 && errno == EINTR) continue;
                if (ready < 0) throw std::runtime_error("Listener poll failed");
                if (!ready) continue;
                Socket client(::accept(listener.fd, nullptr, nullptr));
                std::cout << "Client connected\n" << std::flush;
                try { serve(client.fd, latest); }
                catch (const std::exception& error) { std::cerr << "Client ended: " << error.what() << '\n'; }
            }
        } catch (...) {
            latest.running = false;
            latest.changed.notify_all();
            producer.join();
            throw;
        }
        latest.running = false;
        latest.changed.notify_all();
        producer.join();
        if (!latest.error.empty()) throw std::runtime_error(latest.error);
        return interrupted ? 130 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Camera stream failed: " << error.what() << '\n';
        return interrupted ? 130 : 1;
    }
}
