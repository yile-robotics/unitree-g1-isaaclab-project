#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class State_PolicySwitch : public FSMState
{
public:
    State_PolicySwitch(int state_mode, std::string state_string);
    ~State_PolicySwitch() = default;

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
    void update_switch_state(float dt);
    void update_output_targets(float dt);

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

    std::shared_ptr<isaaclab::Articulation> robot_;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> locomotion_env_;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> stand_env_;
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

    float blend_elapsed_ = 0.0f;
    float settle_elapsed_ = 0.0f;
    float hold_before_stand_elapsed_ = 0.0f;
    float stand_wait_elapsed_ = 0.0f;
    float stand_wait_log_elapsed_ = 0.0f;

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
};

REGISTER_FSM(State_PolicySwitch)
