#include "FSM/CtrlFSM.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_Passive.h"
#include "State_PolicySwitch.h"

#include <cstdlib>
#include <thread>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

namespace
{
void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if (!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
    }

    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to policy-switch MuJoCo/robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected.");
}
}  // namespace

int main(int argc, char** argv)
{
    const auto vm = param::helper(argc, argv);
    const int domain_id = param::config["domain_id"].as<int>(10);
    const std::string network = vm["network"].as<std::string>();

    std::cout
        << " --- Unitree Robotics ---\n"
        << " G1 Independent Dual-Policy Switch Controller\n"
        << " DDS domain: " << domain_id << "\n";

    unitree::robot::ChannelFactory::Instance()->Init(domain_id, network);
    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5;
    if (!FSMState::lowcmd->check_mode_machine(FSMState::lowstate))
    {
        spdlog::critical("Unmatched robot type.");
        return -1;
    }

    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    if (vm.count("auto_start") || vm.count("auto_fixstand_then_velocity"))
    {
        std::thread([fsm_ptr = fsm.get()] {
            sleep(1);
            spdlog::info("Auto start: Passive -> FixStand");
            fsm_ptr->requestState("FixStand");
            sleep(5);
            spdlog::info("Auto start: FixStand -> PolicySwitch");
            fsm_ptr->requestState("PolicySwitch");
        }).detach();
    }
    else
    {
        std::cout
            << "手动启动：先按手柄 [L2 + Up] 进入 FixStand，"
            << "再按 [R1 + X] 进入 PolicySwitch。\n";
    }

    while (true)
    {
        sleep(1);
    }
}

