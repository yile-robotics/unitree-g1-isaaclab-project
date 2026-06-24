#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace isaaclab
{
namespace
{
using TrackingClock = std::chrono::steady_clock;

struct TrackingCase
{
    std::string name;
    float vx;
    float vy;
    float wz;
};

std::string tracking_case_name(const std::string& prefix, int centi_units)
{
    char suffix[16];
    std::snprintf(suffix, sizeof(suffix), "%03d", std::abs(centi_units));
    return prefix + "_" + suffix;
}

//测试mujoco速度跟随 用不同的速度测试并且看跟随能力
const std::vector<TrackingCase>& mujoco_tracking_cases()
{
    static const std::vector<TrackingCase> cases = [] {
        std::vector<TrackingCase> items;
        items.push_back({"stand", 0.00f, 0.00f, 0.00f});

        for (int centi = 5; centi <= 60; centi += 5)
        {
            items.push_back({tracking_case_name("forward", centi), centi / 100.0f, 0.00f, 0.00f});
        }
        for (int centi = 5; centi <= 30; centi += 5)
        {
            items.push_back({tracking_case_name("backward", centi), -centi / 100.0f, 0.00f, 0.00f});
        }

        for (int centi = 5; centi <= 50; centi += 5)
        {
            items.push_back({tracking_case_name("left", centi), 0.00f, centi / 100.0f, 0.00f});
        }
        for (int centi = 5; centi <= 50; centi += 5)
        {
            items.push_back({tracking_case_name("right", centi), 0.00f, -centi / 100.0f, 0.00f});
        }

        for (int centi = 5; centi <= 40; centi += 5)
        {
            items.push_back({tracking_case_name("yaw_left", centi), 0.00f, 0.00f, centi / 100.0f});
        }
        for (int centi = 5; centi <= 40; centi += 5)
        {
            items.push_back({tracking_case_name("yaw_right", centi), 0.00f, 0.00f, -centi / 100.0f});
        }

        items.push_back({"diag_small", 0.05f, 0.05f, 0.05f});
        items.push_back({"diag_medium", 0.20f, 0.15f, 0.10f});
        items.push_back({"diag_fast", 0.45f, 0.30f, 0.20f});
        items.push_back({"diag_limit_pos", 0.60f, 0.50f, 0.40f});
        items.push_back({"diag_limit_neg", -0.30f, -0.50f, -0.40f});
        return items;
    }();
    return cases;
}

bool mujoco_tracking_eval_enabled()
{
    return std::getenv("MUJOCO_TRACKING_EVAL") != nullptr;
}

double mujoco_tracking_case_duration_s()
{
    const char* duration_env = std::getenv("MUJOCO_TRACKING_CASE_DURATION_S");
    if (!duration_env)
    {
        return 8.0;
    }
    const double duration_s = std::atof(duration_env);
    return duration_s > 0.0 ? duration_s : 8.0;
}

bool mujoco_tracking_exit_on_done()
{
    return std::getenv("MUJOCO_TRACKING_EXIT_ON_DONE") != nullptr;
}

TrackingClock::time_point mujoco_tracking_start_time;
bool mujoco_tracking_timer_started = false;

//将mujoco测试结果写进表格
void write_mujoco_tracking_command(const TrackingCase& item, double case_time_s)
{
    const char* path_env = std::getenv("MUJOCO_TRACKING_CMD_FILE");
    const std::string path = path_env ? path_env : "/tmp/g1_mujoco_tracking_cmd.txt";
    const std::string tmp_path = path + ".tmp";

    std::ofstream f(tmp_path);
    f << item.name << " " << item.vx << " " << item.vy << " " << item.wz << " " << case_time_s << "\n";
    f.close();
    std::rename(tmp_path.c_str(), path.c_str());
}
}

void reset_mujoco_tracking_eval_timer()
{
    if (mujoco_tracking_eval_enabled())
    {
        mujoco_tracking_start_time = TrackingClock::now();
        mujoco_tracking_timer_started = true;
    }
}

// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
//给 policy 生成 velocity command observation
 //如果开启了 mujoco_tracking_eval_enabled()
   //→ 不用键盘，自动按预设 case 发送 vx, vy, wz

//2. 如果没开启 tracking eval
  // → 读取键盘，手动控制 vx, vy, wz
  //每个 case 跑 8 秒，case 包含不同的 vx, vy, wz 组合，覆盖前后左右移动和旋转
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    if (mujoco_tracking_eval_enabled())
    {
        static int last_case_index = -1;
        const double case_duration_s = mujoco_tracking_case_duration_s();

        if (!mujoco_tracking_timer_started)
        {
            reset_mujoco_tracking_eval_timer();
        }

        const auto& cases = mujoco_tracking_cases();
        const double elapsed_s =
            std::chrono::duration<double>(TrackingClock::now() - mujoco_tracking_start_time).count();
        const double total_duration_s = case_duration_s * static_cast<double>(cases.size());
        if (elapsed_s >= total_duration_s)
        {
            static bool printed_done = false;
            const TrackingCase done_case = {"done", 0.0f, 0.0f, 0.0f};
            if (!printed_done)
            {
                std::cout << "mujoco tracking evaluation done: "
                          << cases.size() << " cases x " << case_duration_s
                          << "s = " << total_duration_s << "s" << std::endl;
                printed_done = true;
            }
            env->robot->data.command_vel_b = Eigen::Vector3f::Zero();
            write_mujoco_tracking_command(done_case, case_duration_s);
            if (mujoco_tracking_exit_on_done())
            {
                std::exit(0);
            }
            return std::vector<float>{0.0f, 0.0f, 0.0f};
        }

        int case_index = static_cast<int>(elapsed_s / case_duration_s);
        if (case_index >= static_cast<int>(cases.size()))
        {
            case_index = static_cast<int>(cases.size()) - 1;
        }
        const double case_time_s = elapsed_s - case_index * case_duration_s;
        const auto& item = cases[case_index];

        if (case_index != last_case_index)
        {
            std::cout << "mujoco tracking case " << item.name
                      << " vx=" << item.vx << " vy=" << item.vy << " wz=" << item.wz << std::endl;
            last_case_index = case_index;
        }

        env->robot->data.command_vel_b = Eigen::Vector3f(item.vx, item.vy, item.wz);
        write_mujoco_tracking_command(item, case_time_s);
        return std::vector<float>{item.vx, item.vy, item.wz};
    }

    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    static std::string handled_key;
    constexpr float lin_step = 0.05f;
    constexpr float yaw_step = 0.02f;
//键盘输入指令进行测试
    if (key.empty())
    {
        handled_key.clear();
    }
    else if (key != handled_key)
    {
        if (key == "w") {
            cmd[0] += lin_step;
        } else if (key == "s") {
            cmd[0] -= lin_step;
        } else if (key == "a") {
            cmd[1] += lin_step;
        } else if (key == "d") {
            cmd[1] -= lin_step;
        } else if (key == "q") {
            cmd[2] += yaw_step;
        } else if (key == "e") {
            cmd[2] -= yaw_step;
        } else if (key == "x") {
            cmd = {0.0f, 0.0f, 0.0f};
        }
        handled_key = key;
    }
    cmd[0] = std::clamp(cmd[0], cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    cmd[1] = std::clamp(cmd[1], cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    cmd[2] = std::clamp(cmd[2], cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
    static std::vector<float> last_cmd = {999.0f, 999.0f, 999.0f};
    if (cmd != last_cmd)
    {
        std::cout << "keyboard cmd vx=" << cmd[0] << " vy=" << cmd[1] << " wz=" << cmd[2] << std::endl;
        last_cmd = cmd;
    }
    env->robot->data.command_vel_b = Eigen::Vector3f(cmd[0], cmd[1], cmd[2]);
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    static int print_count = 0;
    if (++print_count >= 200)
    {
        print_count = 0;
        env->robot->update();
        const auto cmd = env->robot->data.command_vel_b;
        const auto gyro = env->robot->data.root_ang_vel_b;
        std::cout
            << "track cmd=(" << cmd[0] << ", " << cmd[1] << ", " << cmd[2] << ")"
            << " actual_wz=" << gyro[2]
            << " wz_err=" << (gyro[2] - cmd[2])
            << " gyro=(" << gyro[0] << ", " << gyro[1] << ", " << gyro[2] << ")"
            << std::endl;
    }
}
