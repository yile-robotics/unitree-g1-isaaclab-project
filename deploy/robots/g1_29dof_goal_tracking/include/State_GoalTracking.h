#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "unitree/dds_wrapper/robots/go2/go2_sub.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class State_GoalTracking : public FSMState
{
public:
    State_GoalTracking(int state_mode, std::string state_string);
    ~State_GoalTracking() = default;

    void enter() override;
    void run() override;
    void exit() override;

private:
    enum class PolicyMode
    {
        Stand,
        Locomotion,
    };

    enum class TransitionMode
    {
        None,
        WaitingForStand,
        HoldingBeforeStand,
        BlendingToStand,
        BlendingToLocomotion,
    };

    void policy_loop();
    void handle_keyboard();
    void request_stand();
    void request_locomotion();
    void begin_hold_before_stand();
    void begin_blend(PolicyMode destination);
    void update_command(float dt);
    void update_goal_command(float dt);
    void update_switch_state(float dt);
    void update_output_targets(float dt);
    void log_goal_sample(float dt);

    float command_norm(const std::vector<float>& command) const;
    float current_tilt_angle() const;
    float max_leg_joint_velocity() const;
    void print_stand_wait_status(
        float filtered_command_norm,
        float leg_joint_velocity,
        float yaw_rate,
        float tilt_angle,
        bool relaxed_window
    ) const;
    std::vector<float> limit_target_step(
        const std::vector<float>& previous,
        const std::vector<float>& desired
    ) const;

    struct RobotPose
    {
        float x = 0.0f;
        float y = 0.0f;
        float yaw = 0.0f;
        bool valid = false;
    };

    struct Waypoint
    {
        float x = 0.0f;
        float y = 0.0f;
        float yaw = 0.0f;
    };

    struct PathTarget
    {
        float x = 0.0f;
        float y = 0.0f;
        float yaw = 0.0f;
        float cross_track_error = 0.0f;
        float progress = 0.0f;
        float total_length = 0.0f;
        std::size_t segment_index = 0;
        bool valid = false;
    };

    const Waypoint& current_waypoint() const;
    PathTarget compute_path_target(const RobotPose& pose) const;
    RobotPose current_robot_pose() const;

    std::shared_ptr<isaaclab::Articulation> robot_;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> locomotion_env_;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> stand_env_;
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState> sport_state_;
    std::vector<float> control_stiffness_;
    std::vector<float> control_damping_;

    std::thread policy_thread_;
    std::atomic<bool> policy_thread_running_{false};

    mutable std::mutex output_mutex_;
    std::vector<float> output_targets_;
    std::vector<float> blend_start_targets_;

    PolicyMode active_mode_ = PolicyMode::Stand;
    PolicyMode blend_destination_ = PolicyMode::Stand;
    TransitionMode transition_mode_ = TransitionMode::None;

    std::vector<float> requested_command_{0.0f, 0.0f, 0.0f};
    std::vector<float> filtered_command_{0.0f, 0.0f, 0.0f};
    std::string handled_key_;
    RobotPose last_pose_;

    float blend_elapsed_ = 0.0f;
    float settle_elapsed_ = 0.0f;
    float hold_before_stand_elapsed_ = 0.0f;
    float stand_wait_elapsed_ = 0.0f;
    float stand_wait_log_elapsed_ = 0.0f;
    float goal_elapsed_ = 0.0f;
    float log_elapsed_ = 0.0f;

    float blend_duration_ = 0.6f;
    float stand_blend_duration_ = 1.2f;
    float hold_before_stand_duration_ = 0.25f;
    float command_ramp_duration_ = 0.5f;
    float max_joint_target_step_ = 0.035f;
    float stand_settle_time_ = 0.35f;
    float max_settle_joint_velocity_ = 0.8f;
    float max_settle_yaw_rate_ = 0.25f;
    float stand_wait_relax_time_ = 0.5f;
    float stand_wait_timeout_ = 0.9f;
    float relaxed_max_settle_joint_velocity_ = 1.5f;
    float relaxed_max_settle_yaw_rate_ = 0.4f;
    float stop_window_max_tilt_angle_ = 0.5f;
    float min_stand_wait_time_ = 1.0f;
    bool require_leg_velocity_for_stand_ = false;
    float stand_command_threshold_ = 0.03f;
    float locomotion_command_threshold_ = 0.06f;
    float max_tilt_angle_ = 0.65f;
    float linear_command_step_ = 0.05f;
    float yaw_command_step_ = 0.02f;
    bool automatic_switching_ = false;

    bool goal_tracking_enabled_ = false;
    bool goal_reached_ = false;
    bool start_goal_on_enter_ = false;
    bool auto_switch_to_locomotion_ = true;
    std::vector<Waypoint> waypoints_;
    std::size_t waypoint_index_ = 0;
    PathTarget last_path_target_;
    float goal_x_ = 3.0f;
    float goal_y_ = 0.0f;
    float goal_yaw_ = 0.0f;
    float goal_tolerance_ = 0.15f;
    float yaw_tolerance_ = 0.25f;
    float path_lookahead_distance_ = 0.7f;
    float slow_radius_ = 0.8f;
    float xy_kp_ = 0.7f;
    float yaw_kp_ = 1.2f;
    float max_goal_vx_ = 0.35f;
    float max_goal_vy_ = 0.25f;
    float max_goal_wz_ = 0.35f;
    float log_dt_ = 0.02f;
    std::string log_path_;
    std::ofstream log_file_;
};

REGISTER_FSM(State_GoalTracking)
