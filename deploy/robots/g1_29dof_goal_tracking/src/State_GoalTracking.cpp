#include "State_GoalTracking.h"

#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "unitree_articulation.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
std::mutex command_mutex;
std::vector<float> goal_tracking_command{0.0f, 0.0f, 0.0f};

float wrap_to_pi(float angle)
{
    while (angle > static_cast<float>(M_PI))
    {
        angle -= 2.0f * static_cast<float>(M_PI);
    }
    while (angle < -static_cast<float>(M_PI))
    {
        angle += 2.0f * static_cast<float>(M_PI);
    }
    return angle;
}

float clamp_step(float current, float target, float max_step)
{
    return current + std::clamp(target - current, -max_step, max_step);
}

float smoothstep(float value)
{
    const float t = std::clamp(value, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

std::filesystem::path resolve_policy_dir(const YAML::Node& cfg, const char* key)
{
    return param::parser_policy_dir(cfg[key].as<std::string>());
}
//看看policy频率 关节顺序 关节初始位置是否相同
void require_compatible_policies(const YAML::Node& locomotion_cfg, const YAML::Node& stand_cfg)
{
    const float locomotion_dt = locomotion_cfg["step_dt"].as<float>();
    const float stand_dt = stand_cfg["step_dt"].as<float>();
    if (std::abs(locomotion_dt - stand_dt) > 1.0e-6f)
    {
        throw std::runtime_error("Locomotion and stand policies use different step_dt values.");
    }

    const auto locomotion_map = locomotion_cfg["joint_ids_map"].as<std::vector<int>>();
    const auto stand_map = stand_cfg["joint_ids_map"].as<std::vector<int>>();
    if (locomotion_map != stand_map)
    {
        throw std::runtime_error("Locomotion and stand policies use different joint_ids_map values.");
    }

    const auto locomotion_default = locomotion_cfg["default_joint_pos"].as<std::vector<float>>();
    const auto stand_default = stand_cfg["default_joint_pos"].as<std::vector<float>>();
    if (locomotion_default.size() != stand_default.size())
    {
        throw std::runtime_error("Locomotion and stand policies use different joint counts.");
    }
    for (std::size_t i = 0; i < locomotion_default.size(); ++i)
    {
        if (std::abs(locomotion_default[i] - stand_default[i]) > 1.0e-5f)
        {
            throw std::runtime_error(
                "Locomotion and stand policies use different default_joint_pos values."
            );
        }
    }
}
}  // namespace

namespace isaaclab
{
namespace mdp
{
// 两个 deploy.yaml 都使用这个 observation 名称 从yaml文件中读取的命令会经过相同的裁剪，因此在切换阶段不会因为突然的高速度命令而失稳。
// 行走策略会按照自己的速度范围裁剪命令；站立策略的范围是零，因此自然得到零命令。
REGISTER_OBSERVATION(goal_tracking_velocity_commands)
{
    std::lock_guard<std::mutex> lock(command_mutex);
    const auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];

    std::vector<float> command = goal_tracking_command;
    command[0] = std::clamp(
        command[0],
        ranges["lin_vel_x"][0].as<float>(),
        ranges["lin_vel_x"][1].as<float>()
    );
    command[1] = std::clamp(
        command[1],
        ranges["lin_vel_y"][0].as<float>(),
        ranges["lin_vel_y"][1].as<float>()
    );
    command[2] = std::clamp(
        command[2],
        ranges["ang_vel_z"][0].as<float>(),
        ranges["ang_vel_z"][1].as<float>()
    );
    return command;
}
}  // namespace mdp
}  // namespace isaaclab
//读取FSM配置 找poicy的路径 载入policy 进行切换控制
State_GoalTracking::State_GoalTracking(int state_mode, std::string state_string)
    : FSMState(state_mode, std::move(state_string))
{
    const auto cfg = param::config["FSM"][getStateString()];
    const auto locomotion_dir = resolve_policy_dir(cfg, "locomotion_policy_dir");
    const auto stand_dir = resolve_policy_dir(cfg, "stand_policy_dir");

    const YAML::Node locomotion_cfg =
        YAML::LoadFile((locomotion_dir / "params" / "deploy.yaml").string());
    const YAML::Node stand_cfg =
        YAML::LoadFile((stand_dir / "params" / "deploy.yaml").string());
    require_compatible_policies(locomotion_cfg, stand_cfg);

    // 两个环境共享 robot_，后构造的环境会覆盖 robot_ 中的 PD 参数。
    // 切换控制始终采用 locomotion policy 的整机 PD：两套策略的腿部 PD 相同，
    // 而官方 locomotion policy 对腰部和手臂使用了更高阻尼。
    control_stiffness_ = locomotion_cfg["stiffness"].as<std::vector<float>>();
    control_damping_ = locomotion_cfg["damping"].as<std::vector<float>>();
    if (control_stiffness_.size() != control_damping_.size())
    {
        throw std::runtime_error("Locomotion stiffness and damping sizes do not match.");
    }

    robot_ = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);
    locomotion_env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(locomotion_cfg, robot_);
    locomotion_env_->alg =
        std::make_unique<isaaclab::OrtRunner>((locomotion_dir / "exported" / "policy.onnx").string());

    stand_env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(stand_cfg, robot_);
    stand_env_->alg =
        std::make_unique<isaaclab::OrtRunner>((stand_dir / "exported" / "policy.onnx").string());
    //默认初始是stand状态
    const std::string initial_policy = cfg["initial_policy"].as<std::string>("stand");
    active_mode_ = initial_policy == "locomotion" ? PolicyMode::Locomotion : PolicyMode::Stand;
    blend_destination_ = active_mode_;
    // 切换控制的参数 限制切换条件 切换过程中的混合时长 速度命令的平滑时间 以及一些限幅参数
    automatic_switching_ = cfg["automatic_switching"].as<bool>(false);
    blend_duration_ = cfg["blend_duration"].as<float>(0.6f);
    stand_blend_duration_ = cfg["stand_blend_duration"].as<float>(blend_duration_);
    hold_before_stand_duration_ = cfg["hold_before_stand_duration"].as<float>(0.25f);
    command_ramp_duration_ = cfg["command_ramp_duration"].as<float>(0.5f);
    max_joint_target_step_ = cfg["max_joint_target_step"].as<float>(0.035f);
    stand_settle_time_ = cfg["stand_settle_time"].as<float>(0.35f);
    max_settle_joint_velocity_ = cfg["max_settle_joint_velocity"].as<float>(0.8f);
    max_settle_yaw_rate_ = cfg["max_settle_yaw_rate"].as<float>(0.25f);
    stand_wait_relax_time_ = cfg["stand_wait_relax_time"].as<float>(0.5f);
    stand_wait_timeout_ = cfg["stand_wait_timeout"].as<float>(0.9f);
    relaxed_max_settle_joint_velocity_ =
        cfg["relaxed_max_settle_joint_velocity"].as<float>(1.5f);
    relaxed_max_settle_yaw_rate_ = cfg["relaxed_max_settle_yaw_rate"].as<float>(0.4f);
    stop_window_max_tilt_angle_ = cfg["stop_window_max_tilt_angle"].as<float>(0.5f);
    min_stand_wait_time_ = cfg["min_stand_wait_time"].as<float>(1.0f);
    require_leg_velocity_for_stand_ = cfg["require_leg_velocity_for_stand"].as<bool>(false);
    stand_command_threshold_ = cfg["stand_command_threshold"].as<float>(0.03f);
    locomotion_command_threshold_ = cfg["locomotion_command_threshold"].as<float>(0.06f);
    max_tilt_angle_ = cfg["max_tilt_angle"].as<float>(0.65f);
    linear_command_step_ = cfg["linear_command_step"].as<float>(0.05f);
    yaw_command_step_ = cfg["yaw_command_step"].as<float>(0.02f);
    start_goal_on_enter_ = cfg["start_goal_on_enter"].as<bool>(false);
    auto_switch_to_locomotion_ = cfg["auto_switch_to_locomotion"].as<bool>(true);
    if (cfg["goal"].IsDefined())
    {
        goal_x_ = cfg["goal"]["x"].as<float>(2.0f);
        goal_y_ = cfg["goal"]["y"].as<float>(0.0f);
        goal_yaw_ = cfg["goal"]["yaw"].as<float>(0.0f);
    }
    if (cfg["waypoints"].IsDefined())
    {
        waypoints_.clear();
        for (const auto& waypoint : cfg["waypoints"])
        {
            waypoints_.push_back(
                {
                    waypoint["x"].as<float>(),
                    waypoint["y"].as<float>(),
                    waypoint["yaw"].as<float>(0.0f),
                }
            );
        }
    }
    if (waypoints_.empty())
    {
        waypoints_.push_back({goal_x_, goal_y_, goal_yaw_});
    }
    goal_tolerance_ = cfg["goal_tolerance"].as<float>(0.15f);
    yaw_tolerance_ = cfg["yaw_tolerance"].as<float>(0.25f);
    path_lookahead_distance_ = cfg["path_lookahead_distance"].as<float>(0.7f);
    slow_radius_ = cfg["slow_radius"].as<float>(0.8f);
    xy_kp_ = cfg["xy_kp"].as<float>(0.7f);
    yaw_kp_ = cfg["yaw_kp"].as<float>(1.2f);
    max_goal_vx_ = cfg["max_goal_vx"].as<float>(0.35f);
    max_goal_vy_ = cfg["max_goal_vy"].as<float>(0.25f);
    max_goal_wz_ = cfg["max_goal_wz"].as<float>(0.35f);
    log_dt_ = cfg["log_dt"].as<float>(0.02f);
    log_path_ = cfg["log_path"].as<std::string>("outputs/goal_tracking/latest/trajectory.csv");
    if (const char* log_path_override = std::getenv("GOAL_TRACKING_LOG_PATH"))
    {
        if (log_path_override[0] != '\0')
        {
            log_path_ = log_path_override;
        }
    }

    output_targets_.assign(
        robot_->data.default_joint_pos.data(),
        robot_->data.default_joint_pos.data() + robot_->data.default_joint_pos.size()
    );
    blend_start_targets_ = output_targets_;

    // 姿态超过安全阈值时直接回到 Passive，不继续尝试策略切换。
    registered_checks.emplace_back(
        std::make_pair(
            [this]() { return current_tilt_angle() > max_tilt_angle_; },
            FSMStringMap.right.at("Passive")
        )
    );

    sport_state_ = std::make_shared<unitree::robot::go2::subscription::SportModeState>("rt/sportmodestate");
}

void State_GoalTracking::enter()
{
    // 两个策略已经在构造阶段验证了关节映射和 default pose 一致。
    // 腿部增益一致；腰部和手臂固定采用官方 locomotion policy 的整机 PD。
    for (std::size_t i = 0; i < control_stiffness_.size(); ++i)
    {
        auto& motor = lowcmd->msg_.motor_cmd()[i];
        motor.kp() = control_stiffness_[i];
        motor.kd() = control_damping_[i];
        motor.dq() = 0.0f;
        motor.tau() = 0.0f;
    }

    robot_->update();
    requested_command_ = {0.0f, 0.0f, 0.0f};
    filtered_command_ = {0.0f, 0.0f, 0.0f};
    goal_tracking_enabled_ = start_goal_on_enter_;
    goal_reached_ = false;
    waypoint_index_ = 0;
    last_path_target_ = {};
    goal_elapsed_ = 0.0f;
    log_elapsed_ = 0.0f;
    transition_mode_ = TransitionMode::None;
    settle_elapsed_ = 0.0f;
    blend_elapsed_ = 0.0f;

    {
        std::lock_guard<std::mutex> lock(command_mutex);
        goal_tracking_command = filtered_command_;
    }

    locomotion_env_->reset();
    stand_env_->reset();

    // 从当前实测姿态开始限速，而不是第一帧直接跳到 policy 目标。用机器人关节的kp kd
    output_targets_.assign(
        robot_->data.joint_pos.data(),
        robot_->data.joint_pos.data() + robot_->data.joint_pos.size()
    );

    if (!log_path_.empty())
    {
        const std::filesystem::path path(log_path_);
        if (path.has_parent_path())
        {
            std::filesystem::create_directories(path.parent_path());
        }
        log_file_.open(path);
        if (log_file_.is_open())
        {
            log_file_
                << "time,x,y,yaw,target_x,target_y,target_yaw,waypoint_index,"
                << "cmd_vx,cmd_vy,cmd_wz,dist,yaw_error,cross_track_error,path_progress,"
                << "active_policy,transition,goal_enabled,goal_reached\n";
        }
        else
        {
            std::cerr << "无法打开 goal tracking CSV: " << log_path_ << std::endl;
        }
    }

    policy_thread_running_ = true;
    policy_thread_ = std::thread(&State_GoalTracking::policy_loop, this);

    std::cout
        << "\n独立 goal tracking 双策略控制已启动（初始为站立策略）。\n"
        << "  waypoints: " << waypoints_.size()
        << ", first=(" << current_waypoint().x << ", " << current_waypoint().y
        << ", yaw=" << current_waypoint().yaw << ")"
        << ", xy_tolerance=" << goal_tolerance_
        << ", yaw_tolerance=" << yaw_tolerance_ << "\n"
        << "  1: 请求切换到站立策略\n"
        << "  2: 请求切换到行走策略\n"
        << "  3: 强制平滑切换到站立策略（跳过稳定等待，仅用于调试）\n"
        << "  g: 启动/重新启动 goal tracking\n"
        << "  x: 停止 goal tracking 并将速度命令归零\n"
        << "  w/s/a/d/q/e: 手动微调速度命令（仅在 goal tracking 停止时使用）\n"
        << "CSV: " << log_path_ << "\n"
        << std::endl;
}

void State_GoalTracking::run()
{
    std::lock_guard<std::mutex> lock(output_mutex_);
    for (std::size_t i = 0; i < robot_->data.joint_ids_map.size(); ++i)
    {
        const int motor_id = static_cast<int>(robot_->data.joint_ids_map[i]);
        lowcmd->msg_.motor_cmd()[motor_id].q() = output_targets_[i];
    }
}

void State_GoalTracking::exit()
{
    policy_thread_running_ = false;
    if (policy_thread_.joinable())
    {
        policy_thread_.join();
    }
    if (log_file_.is_open())
    {
        log_file_.flush();
        log_file_.close();
    }
}
//，它按 policy 的频率不断读取键盘命令、平滑速度、同时运行行走和站立两个 policy，然后根据切换逻辑更新最终要发给机器人的关节目标 output_targets_
void State_GoalTracking::policy_loop()
{
    using clock = std::chrono::steady_clock;
    const float dt = locomotion_env_->step_dt;
    const auto period = std::chrono::duration_cast<clock::duration>(
        std::chrono::duration<double>(dt)
    );
    auto wake_time = clock::now() + period;

    while (policy_thread_running_)
    {
        goal_elapsed_ += dt;
        handle_keyboard();
        update_goal_command(dt);
        update_command(dt);

        {
            std::lock_guard<std::mutex> lock(command_mutex);
            goal_tracking_command = filtered_command_;
        }
        robot_->data.command_vel_b =
            Eigen::Vector3f(filtered_command_[0], filtered_command_[1], filtered_command_[2]);

        // 两个环境各自维护 observation history 和 last_action。
        // 共享的 robot_ 只负责提供同一时刻的实际机器人状态。
        locomotion_env_->step();
        stand_env_->step();

        update_switch_state(dt);
        update_output_targets(dt);
        log_goal_sample(dt);

        std::this_thread::sleep_until(wake_time);
        wake_time += period;
    }
}
// 下面是一些辅助函数的实现，包括命令更新、切换状态更新、输出目标更新，以及一些状态评估函数。
void State_GoalTracking::handle_keyboard()
{
    const std::string key = FSMState::keyboard->key();
    if (key.empty())
    {
        handled_key_.clear();
        return;
    }
    if (key == handled_key_)
    {
        return;
    }
    handled_key_ = key;

    if (key == "1")
    {
        request_stand();
    }
    else if (key == "2")
    {
        request_locomotion();
    }
    else if (key == "3")
    {
        requested_command_ = {0.0f, 0.0f, 0.0f};
        filtered_command_ = {0.0f, 0.0f, 0.0f};
        goal_tracking_enabled_ = false;
        std::cout << "强制切换到站立策略：跳过稳定等待，先冻结当前目标。" << std::endl;
        begin_hold_before_stand();
    }
    else if (key == "g")
    {
        goal_tracking_enabled_ = true;
        goal_reached_ = false;
        waypoint_index_ = 0;
        last_path_target_ = {};
        goal_elapsed_ = 0.0f;
        std::cout
            << "启动 path following: "
            << waypoints_.size() << " waypoints, first=("
            << current_waypoint().x << ", " << current_waypoint().y
            << ", yaw=" << current_waypoint().yaw << ")"
            << ", lookahead=" << path_lookahead_distance_
            << std::endl;
        if (auto_switch_to_locomotion_
            && active_mode_ == PolicyMode::Stand
            && transition_mode_ == TransitionMode::None)
        {
            begin_blend(PolicyMode::Locomotion);
        }
    }
    else if (key == "w")
    {
        goal_tracking_enabled_ = false;
        requested_command_[0] += linear_command_step_;
    }
    else if (key == "s")
    {
        goal_tracking_enabled_ = false;
        requested_command_[0] -= linear_command_step_;
    }
    else if (key == "a")
    {
        goal_tracking_enabled_ = false;
        requested_command_[1] += linear_command_step_;
    }
    else if (key == "d")
    {
        goal_tracking_enabled_ = false;
        requested_command_[1] -= linear_command_step_;
    }
    else if (key == "q")
    {
        goal_tracking_enabled_ = false;
        requested_command_[2] += yaw_command_step_;
    }
    else if (key == "e")
    {
        goal_tracking_enabled_ = false;
        requested_command_[2] -= yaw_command_step_;
    }
    else if (key == "x")
    {
        goal_tracking_enabled_ = false;
        requested_command_ = {0.0f, 0.0f, 0.0f};
        std::cout << "停止 goal tracking，速度命令归零。" << std::endl;
    }

    const auto ranges = locomotion_env_->cfg["commands"]["base_velocity"]["ranges"];
    requested_command_[0] = std::clamp(
        requested_command_[0],
        ranges["lin_vel_x"][0].as<float>(),
        ranges["lin_vel_x"][1].as<float>()
    );
    requested_command_[1] = std::clamp(
        requested_command_[1],
        ranges["lin_vel_y"][0].as<float>(),
        ranges["lin_vel_y"][1].as<float>()
    );
    requested_command_[2] = std::clamp(
        requested_command_[2],
        ranges["ang_vel_z"][0].as<float>(),
        ranges["ang_vel_z"][1].as<float>()
    );

    std::cout
        << "请求速度: vx=" << requested_command_[0]
        << " vy=" << requested_command_[1]
        << " wz=" << requested_command_[2]
        << std::endl;
}

void State_GoalTracking::update_goal_command(float dt)
{
    (void)dt;
    if (!goal_tracking_enabled_ || goal_reached_)
    {
        return;
    }

    last_pose_ = current_robot_pose();
    if (!last_pose_.valid)
    {
        requested_command_ = {0.0f, 0.0f, 0.0f};
        return;
    }

    last_path_target_ = compute_path_target(last_pose_);
    if (!last_path_target_.valid)
    {
        requested_command_ = {0.0f, 0.0f, 0.0f};
        return;
    }
    waypoint_index_ = last_path_target_.segment_index;

    const Waypoint& final_target = waypoints_.back();
    const float final_dx_w = final_target.x - last_pose_.x;
    const float final_dy_w = final_target.y - last_pose_.y;
    const float final_dist = std::sqrt(final_dx_w * final_dx_w + final_dy_w * final_dy_w);
    const float yaw_error_to_final = wrap_to_pi(final_target.yaw - last_pose_.yaw);
    const bool near_path_end =
        last_path_target_.progress >= last_path_target_.total_length - goal_tolerance_;
    if (near_path_end && final_dist <= goal_tolerance_ && std::abs(yaw_error_to_final) <= yaw_tolerance_)
    {
        goal_reached_ = true;
        goal_tracking_enabled_ = false;
        requested_command_ = {0.0f, 0.0f, 0.0f};
        std::cout
            << "到达路径终点: dist=" << final_dist
            << " yaw_err=" << yaw_error_to_final
            << " cross_track=" << last_path_target_.cross_track_error
            << "，请求切回站立。"
            << std::endl;
        request_stand();
        return;
    }

    const float dx_w = last_path_target_.x - last_pose_.x;
    const float dy_w = last_path_target_.y - last_pose_.y;
    const float yaw_error_to_target = wrap_to_pi(last_path_target_.yaw - last_pose_.yaw);
    const float cy = std::cos(last_pose_.yaw);
    const float sy = std::sin(last_pose_.yaw);
    const float dx_b = cy * dx_w + sy * dy_w;
    const float dy_b = -sy * dx_w + cy * dy_w;
    // Lookahead controls path accuracy; only distance to the final goal controls slowdown.
    const float speed_scale = std::clamp(
        final_dist / std::max(slow_radius_, 1.0e-3f),
        0.15f,
        1.0f
    );

    requested_command_[0] = std::clamp(xy_kp_ * dx_b, -max_goal_vx_, max_goal_vx_) * speed_scale;
    requested_command_[1] = std::clamp(xy_kp_ * dy_b, -max_goal_vy_, max_goal_vy_) * speed_scale;
    requested_command_[2] = std::clamp(yaw_kp_ * yaw_error_to_target, -max_goal_wz_, max_goal_wz_);

    const auto ranges = locomotion_env_->cfg["commands"]["base_velocity"]["ranges"];
    requested_command_[0] = std::clamp(
        requested_command_[0],
        ranges["lin_vel_x"][0].as<float>(),
        ranges["lin_vel_x"][1].as<float>()
    );
    requested_command_[1] = std::clamp(
        requested_command_[1],
        ranges["lin_vel_y"][0].as<float>(),
        ranges["lin_vel_y"][1].as<float>()
    );
    requested_command_[2] = std::clamp(
        requested_command_[2],
        ranges["ang_vel_z"][0].as<float>(),
        ranges["ang_vel_z"][1].as<float>()
    );

    if (auto_switch_to_locomotion_
        && active_mode_ == PolicyMode::Stand
        && transition_mode_ == TransitionMode::None
        && command_norm(requested_command_) > locomotion_command_threshold_)
    {
        begin_blend(PolicyMode::Locomotion);
    }
}

void State_GoalTracking::request_stand()
{
    requested_command_ = {0.0f, 0.0f, 0.0f};
    if (active_mode_ == PolicyMode::Stand && transition_mode_ == TransitionMode::None)
    {
        std::cout << "当前已经是站立策略。" << std::endl;
        return;
    }

    transition_mode_ = TransitionMode::WaitingForStand;
    settle_elapsed_ = 0.0f;
    stand_wait_elapsed_ = 0.0f;
    stand_wait_log_elapsed_ = 0.0f;
    std::cout << "已请求站立：先将速度命令降为零并等待可接管窗口。" << std::endl;
}

void State_GoalTracking::request_locomotion()
{
    if (active_mode_ == PolicyMode::Locomotion && transition_mode_ == TransitionMode::None)
    {
        std::cout << "当前已经是行走策略。" << std::endl;
        return;
    }
    begin_blend(PolicyMode::Locomotion);
}

void State_GoalTracking::begin_hold_before_stand()
{
    if (hold_before_stand_duration_ <= 0.0f)
    {
        begin_blend(PolicyMode::Stand);
        return;
    }

    transition_mode_ = TransitionMode::HoldingBeforeStand;
    blend_destination_ = PolicyMode::Stand;
    hold_before_stand_elapsed_ = 0.0f;
    settle_elapsed_ = 0.0f;

    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        blend_start_targets_ = output_targets_;
    }

    std::cout << "冻结当前关节目标，准备切换到站立策略。" << std::endl;
}

void State_GoalTracking::begin_blend(PolicyMode destination)
{
    blend_destination_ = destination;
    blend_elapsed_ = 0.0f;
    transition_mode_ = destination == PolicyMode::Stand
        ? TransitionMode::BlendingToStand
        : TransitionMode::BlendingToLocomotion;

    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        blend_start_targets_ = output_targets_;
    }

    std::cout
        << "开始平滑切换到"
        << (destination == PolicyMode::Stand ? "站立策略。" : "行走策略。")
        << std::endl;
}

void State_GoalTracking::update_command(float dt)
{
    // 站立等待和切换到站立阶段始终将命令目标设为零。
    std::vector<float> target = requested_command_;
    if (
        transition_mode_ == TransitionMode::WaitingForStand
        || transition_mode_ == TransitionMode::HoldingBeforeStand
        || transition_mode_ == TransitionMode::BlendingToStand
        || (active_mode_ == PolicyMode::Stand && transition_mode_ == TransitionMode::None)
    )
    {
        target = {0.0f, 0.0f, 0.0f};
    }

    const float ramp = std::max(command_ramp_duration_, dt);
    const auto ranges = locomotion_env_->cfg["commands"]["base_velocity"]["ranges"];
    const std::vector<float> max_delta = {
        (ranges["lin_vel_x"][1].as<float>() - ranges["lin_vel_x"][0].as<float>()) * dt / ramp,
        (ranges["lin_vel_y"][1].as<float>() - ranges["lin_vel_y"][0].as<float>()) * dt / ramp,
        (ranges["ang_vel_z"][1].as<float>() - ranges["ang_vel_z"][0].as<float>()) * dt / ramp,
    };

    for (std::size_t i = 0; i < filtered_command_.size(); ++i)
    {
        filtered_command_[i] = clamp_step(filtered_command_[i], target[i], max_delta[i]);
    }
}

void State_GoalTracking::update_switch_state(float dt)
{
    if (automatic_switching_ && transition_mode_ == TransitionMode::None)
    {
        const float norm = command_norm(requested_command_);
        if (active_mode_ == PolicyMode::Stand && norm > locomotion_command_threshold_)
        {
            begin_blend(PolicyMode::Locomotion);
        }
        else if (active_mode_ == PolicyMode::Locomotion && norm < stand_command_threshold_)
        {
            request_stand();
        }
    }

    if (transition_mode_ == TransitionMode::WaitingForStand)
    {
        const float filtered_command_norm = command_norm(filtered_command_);
        const float leg_joint_velocity = max_leg_joint_velocity();
        const float yaw_rate = std::abs(robot_->data.root_ang_vel_b[2]);
        const float tilt_angle = current_tilt_angle();
        const bool relaxed_window = stand_wait_elapsed_ >= stand_wait_relax_time_;
        const bool command_is_zero = filtered_command_norm < stand_command_threshold_;
        const float joint_velocity_limit = relaxed_window
            ? relaxed_max_settle_joint_velocity_
            : max_settle_joint_velocity_;
        const float yaw_rate_limit = relaxed_window
            ? relaxed_max_settle_yaw_rate_
            : max_settle_yaw_rate_;
        const bool joints_are_slow =
            !require_leg_velocity_for_stand_ || leg_joint_velocity < joint_velocity_limit;
        const bool yaw_is_slow = yaw_rate < yaw_rate_limit;
        const bool tilt_is_safe = tilt_angle < stop_window_max_tilt_angle_;

        const bool waited_long_enough = stand_wait_elapsed_ >= min_stand_wait_time_;

        if (waited_long_enough && command_is_zero && joints_are_slow && yaw_is_slow && tilt_is_safe)
        {
            settle_elapsed_ += dt;
            if (settle_elapsed_ >= stand_settle_time_)
            {
                std::cout << "抓到可接管窗口，先冻结当前目标再切换到站立。" << std::endl;
                begin_hold_before_stand();
                return;
            }
        }
        else
        {
            settle_elapsed_ = 0.0f;
        }

        stand_wait_elapsed_ += dt;
        stand_wait_log_elapsed_ += dt;
        if (stand_wait_elapsed_ >= stand_wait_timeout_)
        {
            std::cout
                << "等待站立窗口超时：继续保持 locomotion 零命令等待。"
                << " 如果确认姿态安全，可按 3 手动强制切站立。"
                << std::endl;
            stand_wait_elapsed_ = stand_wait_relax_time_;
            stand_wait_log_elapsed_ = 0.0f;
        }

        if (stand_wait_log_elapsed_ >= 0.5f)
        {
            stand_wait_log_elapsed_ = 0.0f;
            print_stand_wait_status(
                filtered_command_norm,
                leg_joint_velocity,
                yaw_rate,
                tilt_angle,
                relaxed_window
            );
        }
    }
    else if (transition_mode_ == TransitionMode::HoldingBeforeStand)
    {
        hold_before_stand_elapsed_ += dt;
        if (hold_before_stand_elapsed_ >= hold_before_stand_duration_)
        {
            begin_blend(PolicyMode::Stand);
        }
    }
}

void State_GoalTracking::update_output_targets(float dt)
{
    const std::vector<float> locomotion_targets =
        locomotion_env_->action_manager->processed_actions();
    const std::vector<float> stand_targets =
        stand_env_->action_manager->processed_actions();

    std::vector<float> desired_targets;
    if (transition_mode_ == TransitionMode::BlendingToStand
        || transition_mode_ == TransitionMode::BlendingToLocomotion)
    {
        blend_elapsed_ += dt;
        const float duration = blend_destination_ == PolicyMode::Stand
            ? stand_blend_duration_
            : blend_duration_;
        const float alpha = smoothstep(blend_elapsed_ / std::max(duration, dt));
        const auto& destination_targets =
            blend_destination_ == PolicyMode::Stand ? stand_targets : locomotion_targets;

        desired_targets.resize(destination_targets.size());
        for (std::size_t i = 0; i < desired_targets.size(); ++i)
        {
            desired_targets[i] =
                (1.0f - alpha) * blend_start_targets_[i] + alpha * destination_targets[i];
        }

        if (blend_elapsed_ >= duration)
        {
            active_mode_ = blend_destination_;
            transition_mode_ = TransitionMode::None;
            std::cout
                << "切换完成，当前策略："
                << (active_mode_ == PolicyMode::Stand ? "站立" : "行走")
                << std::endl;
        }
    }
    else if (transition_mode_ == TransitionMode::HoldingBeforeStand)
    {
        desired_targets = blend_start_targets_;
    }
    else
    {
        desired_targets =
            active_mode_ == PolicyMode::Stand ? stand_targets : locomotion_targets;
    }

    std::lock_guard<std::mutex> lock(output_mutex_);

    // 正常 locomotion 必须完整执行 policy 输出。持续限速会让快速摆腿动作落后，
    // 尤其在较大速度命令下会破坏原本训练好的步态并导致摔倒。
    //
    // stand 模式继续保留目标变化限幅，用来抑制静止时的小幅高频抖动。
    // 两个 policy 的切换阶段也保留限幅，作为 smoothstep 混合之外的最后一道保护。
    const bool is_blending =
        transition_mode_ == TransitionMode::BlendingToStand
        || transition_mode_ == TransitionMode::BlendingToLocomotion;
    const bool is_holding_before_stand = transition_mode_ == TransitionMode::HoldingBeforeStand;
    const bool limit_stand_output =
        transition_mode_ == TransitionMode::None
        && active_mode_ == PolicyMode::Stand;

    if (is_blending || is_holding_before_stand || limit_stand_output)
    {
        output_targets_ = limit_target_step(output_targets_, desired_targets);
    }
    else
    {
        output_targets_ = desired_targets;
    }
}

float State_GoalTracking::command_norm(const std::vector<float>& command) const
{
    return std::sqrt(
        command[0] * command[0]
        + command[1] * command[1]
        + command[2] * command[2]
    );
}
// current_tilt_angle()
// 看机器人身体有没有歪太多，用于摔倒保护。

// max_leg_joint_velocity()
// 看腿有没有慢下来，用于行走切站立前的稳定判断。

// limit_target_step()
// 限制关节目标变化速度，避免切换或站立时目标突然跳变。
float State_GoalTracking::current_tilt_angle() const
{
    const float upright_cosine = std::clamp(-robot_->data.projected_gravity_b[2], -1.0f, 1.0f);
    return std::acos(upright_cosine);
}

const State_GoalTracking::Waypoint& State_GoalTracking::current_waypoint() const
{
    if (waypoints_.empty())
    {
        static const Waypoint fallback{3.0f, 0.0f, 0.0f};
        return fallback;
    }
    const std::size_t index = std::min(waypoint_index_, waypoints_.size() - 1);
    return waypoints_[index];
}

State_GoalTracking::PathTarget State_GoalTracking::compute_path_target(const RobotPose& pose) const
{
    PathTarget result;
    if (!pose.valid || waypoints_.empty())
    {
        return result;
    }
    if (waypoints_.size() == 1)
    {
        result.x = waypoints_[0].x;
        result.y = waypoints_[0].y;
        result.yaw = waypoints_[0].yaw;
        result.cross_track_error = std::hypot(pose.x - result.x, pose.y - result.y);
        result.valid = true;
        return result;
    }

    float best_progress = 0.0f;
    float best_distance = std::numeric_limits<float>::infinity();
    float total_length = 0.0f;
    std::size_t best_segment = 0;

    for (std::size_t i = 0; i + 1 < waypoints_.size(); ++i)
    {
        const Waypoint& start = waypoints_[i];
        const Waypoint& end = waypoints_[i + 1];
        const float vx = end.x - start.x;
        const float vy = end.y - start.y;
        const float segment_len = std::hypot(vx, vy);
        if (segment_len <= 1.0e-5f)
        {
            continue;
        }

        const float wx = pose.x - start.x;
        const float wy = pose.y - start.y;
        const float projection = std::clamp((wx * vx + wy * vy) / (segment_len * segment_len), 0.0f, 1.0f);
        const float closest_x = start.x + projection * vx;
        const float closest_y = start.y + projection * vy;
        const float distance = std::hypot(pose.x - closest_x, pose.y - closest_y);
        if (distance < best_distance)
        {
            best_distance = distance;
            best_progress = total_length + projection * segment_len;
            best_segment = i;
        }
        total_length += segment_len;
    }

    if (!std::isfinite(best_distance) || total_length <= 1.0e-5f)
    {
        return result;
    }

    const float lookahead_progress =
        std::min(best_progress + std::max(path_lookahead_distance_, 0.0f), total_length);
    float accumulated = 0.0f;
    for (std::size_t i = 0; i + 1 < waypoints_.size(); ++i)
    {
        const Waypoint& start = waypoints_[i];
        const Waypoint& end = waypoints_[i + 1];
        const float vx = end.x - start.x;
        const float vy = end.y - start.y;
        const float segment_len = std::hypot(vx, vy);
        if (segment_len <= 1.0e-5f)
        {
            continue;
        }

        if (lookahead_progress <= accumulated + segment_len || i + 2 == waypoints_.size())
        {
            const float segment_progress =
                std::clamp((lookahead_progress - accumulated) / segment_len, 0.0f, 1.0f);
            result.x = start.x + segment_progress * vx;
            result.y = start.y + segment_progress * vy;
            result.yaw = lookahead_progress >= total_length - 1.0e-4f
                ? end.yaw
                : std::atan2(vy, vx);
            result.cross_track_error = best_distance;
            result.progress = best_progress;
            result.total_length = total_length;
            result.segment_index = best_segment;
            result.valid = true;
            return result;
        }
        accumulated += segment_len;
    }

    const Waypoint& final_target = waypoints_.back();
    result.x = final_target.x;
    result.y = final_target.y;
    result.yaw = final_target.yaw;
    result.cross_track_error = best_distance;
    result.progress = best_progress;
    result.total_length = total_length;
    result.segment_index = best_segment;
    result.valid = true;
    return result;
}

State_GoalTracking::RobotPose State_GoalTracking::current_robot_pose() const
{
    RobotPose pose;
    if (sport_state_ && !sport_state_->isTimeout())
    {
        std::lock_guard<std::mutex> lock(sport_state_->mutex_);
        pose.x = sport_state_->msg_.position()[0];
        pose.y = sport_state_->msg_.position()[1];
        pose.valid = true;
    }

    robot_->update();
    const auto& q = robot_->data.root_quat_w;
    const float w = q.w();
    const float x = q.x();
    const float y = q.y();
    const float z = q.z();
    pose.yaw = std::atan2(
        2.0f * (w * z + x * y),
        1.0f - 2.0f * (y * y + z * z)
    );
    return pose;
}

void State_GoalTracking::log_goal_sample(float dt)
{
    if (!log_file_.is_open())
    {
        return;
    }

    log_elapsed_ += dt;
    if (log_elapsed_ < log_dt_)
    {
        return;
    }
    log_elapsed_ = 0.0f;

    RobotPose pose = last_pose_.valid ? last_pose_ : current_robot_pose();
    PathTarget target = last_path_target_.valid ? last_path_target_ : compute_path_target(pose);
    if (!target.valid)
    {
        const Waypoint& waypoint = current_waypoint();
        target.x = waypoint.x;
        target.y = waypoint.y;
        target.yaw = waypoint.yaw;
    }
    const float dx = target.x - pose.x;
    const float dy = target.y - pose.y;
    const float dist = pose.valid ? std::sqrt(dx * dx + dy * dy) : std::numeric_limits<float>::quiet_NaN();
    const float yaw_error = pose.valid
        ? wrap_to_pi(target.yaw - pose.yaw)
        : std::numeric_limits<float>::quiet_NaN();

    log_file_ << std::fixed << std::setprecision(6)
              << goal_elapsed_ << ","
              << pose.x << ","
              << pose.y << ","
              << pose.yaw << ","
              << target.x << ","
              << target.y << ","
              << target.yaw << ","
              << waypoint_index_ << ","
              << filtered_command_[0] << ","
              << filtered_command_[1] << ","
              << filtered_command_[2] << ","
              << dist << ","
              << yaw_error << ","
              << target.cross_track_error << ","
              << target.progress << ","
              << (active_mode_ == PolicyMode::Stand ? "stand" : "locomotion") << ","
              << static_cast<int>(transition_mode_) << ","
              << (goal_tracking_enabled_ ? 1 : 0) << ","
              << (goal_reached_ ? 1 : 0)
              << "\n";
}

float State_GoalTracking::max_leg_joint_velocity() const
{
    // G1 deploy.yaml 中腿部在 Isaac Lab joint order 下的索引。
    static const std::vector<int> leg_joint_ids = {
        0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18
    };

    float maximum = 0.0f;
    for (const int joint_id : leg_joint_ids)
    {
        maximum = std::max(maximum, std::abs(robot_->data.joint_vel[joint_id]));
    }
    return maximum;
}

void State_GoalTracking::print_stand_wait_status(
    float filtered_command_norm,
    float leg_joint_velocity,
    float yaw_rate,
    float tilt_angle,
    bool relaxed_window
) const
{
    const float joint_velocity_limit = relaxed_window
        ? relaxed_max_settle_joint_velocity_
        : max_settle_joint_velocity_;
    const float yaw_rate_limit = relaxed_window
        ? relaxed_max_settle_yaw_rate_
        : max_settle_yaw_rate_;

    std::cout
        << "等待站立窗口: cmd_norm=" << filtered_command_norm
        << " / " << stand_command_threshold_
        << ", max_leg_dq=" << leg_joint_velocity
        << " / " << joint_velocity_limit
        << (require_leg_velocity_for_stand_ ? "" : " info")
        << ", yaw_rate=" << yaw_rate
        << " / " << yaw_rate_limit
        << ", tilt=" << tilt_angle
        << " / " << stop_window_max_tilt_angle_
        << ", wait=" << stand_wait_elapsed_
        << " / " << min_stand_wait_time_
        << ", window_hold=" << settle_elapsed_
        << " / " << stand_settle_time_
        << (relaxed_window ? " relaxed" : "")
        << std::endl;
}

std::vector<float> State_GoalTracking::limit_target_step(
    const std::vector<float>& previous,
    const std::vector<float>& desired
) const
{
    if (previous.size() != desired.size())
    {
        throw std::runtime_error("Policy target sizes do not match.");
    }

    std::vector<float> limited(desired.size());
    for (std::size_t i = 0; i < desired.size(); ++i)
    {
        limited[i] = clamp_step(previous[i], desired[i], max_joint_target_step_);
    }
    return limited;
}
