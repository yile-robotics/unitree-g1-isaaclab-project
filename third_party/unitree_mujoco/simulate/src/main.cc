// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// !!! hack code: make glfw_adapter.window_ public
#define private public
#include "glfw_adapter.h"
#undef private

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <limits>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>
#include "simulate.h"
#include "array_safety.h"
#include "unitree_sdk2_bridge.h"
#include "param.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"
#define NUM_MOTOR_IDL_GO 20

extern "C"
{
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}
//ElasticBand 类：仿真里的“弹力绳” 最开始拉着这个机器人
class ElasticBand
{
public:
  ElasticBand(){};
  void Advance(std::vector<double> x, std::vector<double> dx)
  {
    std::vector<double> delta_x = {0.0, 0.0, 0.0};
    delta_x[0] = point_[0] - x[0];
    delta_x[1] = point_[1] - x[1];
    delta_x[2] = point_[2] - x[2];
    double distance = sqrt(delta_x[0] * delta_x[0] + delta_x[1] * delta_x[1] + delta_x[2] * delta_x[2]);

    std::vector<double> direction = {0.0, 0.0, 0.0};
    direction[0] = delta_x[0] / distance;
    direction[1] = delta_x[1] / distance;
    direction[2] = delta_x[2] / distance;

    double v = dx[0] * direction[0] + dx[1] * direction[1] + dx[2] * direction[2];

    f_[0] = (stiffness_ * (distance - length_) - damping_ * v) * direction[0];
    f_[1] = (stiffness_ * (distance - length_) - damping_ * v) * direction[1];
    f_[2] = (stiffness_ * (distance - length_) - damping_ * v) * direction[2];
  }


  double stiffness_ = 200;
  double damping_ = 100;
  std::vector<double> point_ = {0, 0, 3};
  double length_ = 0.0;
  bool enable_ = true;
  std::vector<double> f_ = {0, 0, 0};
};
inline ElasticBand elastic_band;


namespace
{
  namespace mj = ::mujoco;
  namespace mju = ::mujoco::sample_util;

  // constants
  const double syncMisalign = 0.1;       // maximum mis-alignment before re-sync (simulation seconds)
  const double simRefreshFraction = 0.7; // fraction of refresh available for simulation
  const int kErrorLength = 1024;         // load error string length

  // model and data
  mjModel *m = nullptr;
  mjData *d = nullptr;

  // control noise variables
  mjtNum *ctrlnoise = nullptr;

  using Seconds = std::chrono::duration<double>;

  struct TrackingCommand
  {
    std::string name = "none";
    double vx = 0.0;
    double vy = 0.0;
    double wz = 0.0;
    double case_time_s = 0.0;
  };

  bool ReadTrackingCommand(TrackingCommand &cmd)
  {
    const char *path_env = std::getenv("MUJOCO_TRACKING_CMD_FILE");
    const std::string path = path_env ? path_env : "/tmp/g1_mujoco_tracking_cmd.txt";
    std::ifstream f(path);
    if (!f.good())
    {
      return false;
    }
    std::string line;
    std::getline(f, line);
    if (line.empty())
    {
      return false;
    }

    static std::string last_line;
    static auto last_line_change = std::chrono::steady_clock::now();
    if (line != last_line)
    {
      last_line = line;
      last_line_change = std::chrono::steady_clock::now();
    }
    const auto stale_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_line_change).count();
    if (stale_s > 0.5)
    {
      return false;
    }

    std::istringstream parser(line);
    TrackingCommand parsed;
    if (!(parser >> parsed.name >> parsed.vx >> parsed.vy >> parsed.wz >> parsed.case_time_s))
    {
      return false;
    }
    cmd = parsed;
    return true;
  }

  void RotateWorldVelocityToLocal(const double quat[4], const double world[3], double local[3])
  {
    const double w = quat[0];
    const double x = quat[1];
    const double y = quat[2];
    const double z = quat[3];

    const double r00 = 1.0 - 2.0 * (y * y + z * z);
    const double r01 = 2.0 * (x * y - z * w);
    const double r02 = 2.0 * (x * z + y * w);
    const double r10 = 2.0 * (x * y + z * w);
    const double r11 = 1.0 - 2.0 * (x * x + z * z);
    const double r12 = 2.0 * (y * z - x * w);
    const double r20 = 2.0 * (x * z - y * w);
    const double r21 = 2.0 * (y * z + x * w);
    const double r22 = 1.0 - 2.0 * (x * x + y * y);

    local[0] = r00 * world[0] + r10 * world[1] + r20 * world[2];
    local[1] = r01 * world[0] + r11 * world[1] + r21 * world[2];
    local[2] = r02 * world[0] + r12 * world[1] + r22 * world[2];
  }

  void MaybeLogTrackingSample()
  {
    if (!m || !d)
    {
      return;
    }

    TrackingCommand cmd;
    if (!ReadTrackingCommand(cmd) || cmd.name == "none")
    {
      return;
    }

    static double last_log_time = -1.0;
    if (last_log_time >= 0.0 && d->time - last_log_time < 0.02)
    {
      return;
    }
    last_log_time = d->time;

    static int root_body_id = -2;
    if (root_body_id == -2)
    {
      root_body_id = mj_name2id(m, mjOBJ_BODY, "pelvis");
      if (root_body_id < 0)
      {
        root_body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
      }
      if (root_body_id < 0)
      {
        root_body_id = mj_name2id(m, mjOBJ_BODY, "torso_link");
      }
    }

    if (root_body_id < 0 || m->nq < 7 || m->nv < 6)
    {
      return;
    }

    const double root_pos[3] = {d->qpos[0], d->qpos[1], d->qpos[2]};
    const double root_quat[4] = {d->qpos[3], d->qpos[4], d->qpos[5], d->qpos[6]};
    const double root_origin_world_vel[3] = {d->qvel[0], d->qvel[1], d->qvel[2]};
    const double root_world_ang_vel[3] = {d->qvel[3], d->qvel[4], d->qvel[5]};
    const double* subtree_com = d->subtree_com + 3 * root_body_id;
    const double root_to_com[3] = {
      subtree_com[0] - root_pos[0],
      subtree_com[1] - root_pos[1],
      subtree_com[2] - root_pos[2],
    };
    const double root_com_world_vel[3] = {
      root_origin_world_vel[0] + root_world_ang_vel[1] * root_to_com[2] - root_world_ang_vel[2] * root_to_com[1],
      root_origin_world_vel[1] + root_world_ang_vel[2] * root_to_com[0] - root_world_ang_vel[0] * root_to_com[2],
      root_origin_world_vel[2] + root_world_ang_vel[0] * root_to_com[1] - root_world_ang_vel[1] * root_to_com[0],
    };

    double root_local_vel[3] = {0.0, 0.0, 0.0};
    double root_local_ang_vel[3] = {0.0, 0.0, 0.0};
    RotateWorldVelocityToLocal(root_quat, root_com_world_vel, root_local_vel);
    RotateWorldVelocityToLocal(root_quat, root_world_ang_vel, root_local_ang_vel);

    const double world_vel[3] = {
      root_com_world_vel[0],
      root_com_world_vel[1],
      root_com_world_vel[2],
    };

    const double actual_vx = root_local_vel[0];
    const double actual_vy = root_local_vel[1];
    const double actual_wz = root_local_ang_vel[2];
    const double err_vx = actual_vx - cmd.vx;
    const double err_vy = actual_vy - cmd.vy;
    const double err_wz = actual_wz - cmd.wz;

    const char *csv_env = std::getenv("MUJOCO_TRACKING_CSV");
    if (!csv_env)
    {
      csv_env = std::getenv("MUJOCO_TRACKING_OUT");
    }
    const std::string csv_path = csv_env ? csv_env : "/tmp/g1_mujoco_tracking.csv";
    static bool wrote_header = false;
    std::ofstream out;
    if (!wrote_header)
    {
      out.open(csv_path, std::ios::out | std::ios::trunc);
      out << "time_s,case,case_time_s,cmd_vx,cmd_vy,cmd_wz,actual_vx,actual_vy,actual_wz,"
          << "world_vx,world_vy,world_vz,err_vx,err_vy,err_wz,abs_err_vx,abs_err_vy,abs_err_wz\n";
      wrote_header = true;
    }
    else
    {
      out.open(csv_path, std::ios::out | std::ios::app);
    }

    out << d->time << ',' << cmd.name << ',' << cmd.case_time_s << ','
        << cmd.vx << ',' << cmd.vy << ',' << cmd.wz << ','
        << actual_vx << ',' << actual_vy << ',' << actual_wz << ','
        << world_vel[0] << ',' << world_vel[1] << ',' << world_vel[2] << ','
        << err_vx << ',' << err_vy << ',' << err_wz << ','
        << std::abs(err_vx) << ',' << std::abs(err_vy) << ',' << std::abs(err_wz) << '\n';
  }

  void RotationMatrixToRpy(const mjtNum *mat, double &roll, double &pitch, double &yaw)
  {
    pitch = std::asin(std::clamp(-static_cast<double>(mat[6]), -1.0, 1.0));
    const double cos_pitch = std::cos(pitch);
    if (std::abs(cos_pitch) > 1.0e-6)
    {
      roll = std::atan2(mat[7], mat[8]);
      yaw = std::atan2(mat[3], mat[0]);
    }
    else
    {
      roll = 0.0;
      yaw = std::atan2(-mat[1], mat[4]);
    }
  }

  void MaybeLogCameraVibrationSample()
  {
    const char *site_name = std::getenv("MUJOCO_CAMERA_VIBRATION_SITE");
    const char *csv_path = std::getenv("MUJOCO_CAMERA_VIBRATION_CSV");
    if (!m || !d || !site_name || !site_name[0] || !csv_path || !csv_path[0])
    {
      return;
    }

    static const mjModel *cached_model = nullptr;
    static int site_id = -1;
    static double last_log_time = -1.0;
    static double previous_time = -1.0;
    static double previous_linear_velocity[3] = {0.0, 0.0, 0.0};
    static bool wrote_header = false;

    if (cached_model != m)
    {
      cached_model = m;
      site_id = mj_name2id(m, mjOBJ_SITE, site_name);
      last_log_time = -1.0;
      previous_time = -1.0;
      wrote_header = false;
      if (site_id < 0)
      {
        std::cerr << "Camera vibration site not found: " << site_name << '\n';
        return;
      }
    }

    if (site_id < 0 || (last_log_time >= 0.0 && d->time - last_log_time < 0.02))
    {
      return;
    }
    last_log_time = d->time;

    const mjtNum *position = d->site_xpos + 3 * site_id;
    const mjtNum *rotation = d->site_xmat + 9 * site_id;
    mjtNum velocity[6] = {};
    mj_objectVelocity(m, d, mjOBJ_SITE, site_id, velocity, 0);

    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    RotationMatrixToRpy(rotation, roll, pitch, yaw);

    const double dt = previous_time >= 0.0 ? d->time - previous_time : 0.0;
    double acceleration[3] = {
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
    };
    if (dt > 1.0e-9)
    {
      for (int i = 0; i < 3; ++i)
      {
        acceleration[i] = (velocity[3 + i] - previous_linear_velocity[i]) / dt;
      }
    }

    std::ofstream out;
    if (!wrote_header)
    {
      out.open(csv_path, std::ios::out | std::ios::trunc);
      out << "time,x,y,z,vx,vy,vz,roll,pitch,yaw,wx,wy,wz,ax,ay,az\n";
      wrote_header = true;
    }
    else
    {
      out.open(csv_path, std::ios::out | std::ios::app);
    }

    out << d->time << ','
        << position[0] << ',' << position[1] << ',' << position[2] << ','
        << velocity[3] << ',' << velocity[4] << ',' << velocity[5] << ','
        << roll << ',' << pitch << ',' << yaw << ','
        << velocity[0] << ',' << velocity[1] << ',' << velocity[2] << ','
        << acceleration[0] << ',' << acceleration[1] << ',' << acceleration[2] << '\n';

    previous_time = d->time;
    for (int i = 0; i < 3; ++i)
    {
      previous_linear_velocity[i] = velocity[3 + i];
    }
  }

  //---------------------------------------- plugin handling -----------------------------------------

  // return the path to the directory containing the current executable
  // used to determine the location of auto-loaded plugin libraries
  std::string getExecutableDir()
  {
#if defined(_WIN32) || defined(__CYGWIN__)
    constexpr char kPathSep = '\\';
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      DWORD buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
        if (written < buf_size)
        {
          success = true;
        }
        else if (written == buf_size)
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
        else
        {
          std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
          return "";
        }
      }
      return realpath.get();
    }();
#else
    constexpr char kPathSep = '/';
#if defined(__APPLE__)
    std::unique_ptr<char[]> buf(nullptr);
    {
      std::uint32_t buf_size = 0;
      _NSGetExecutablePath(nullptr, &buf_size);
      buf.reset(new char[buf_size]);
      if (!buf)
      {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }
      if (_NSGetExecutablePath(buf.get(), &buf_size))
      {
        std::cerr << "unexpected error from _NSGetExecutablePath\n";
      }
    }
    const char *path = buf.get();
#else
    const char *path = "/proc/self/exe";
#endif
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      std::uint32_t buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        std::size_t written = readlink(path, realpath.get(), buf_size);
        if (written < buf_size)
        {
          realpath.get()[written] = '\0';
          success = true;
        }
        else if (written == -1)
        {
          if (errno == EINVAL)
          {
            // path is already not a symlink, just use it
            return path;
          }

          std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
          return "";
        }
        else
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
      }
      return realpath.get();
    }();
#endif

    if (realpath.empty())
    {
      return "";
    }

    for (std::size_t i = realpath.size() - 1; i > 0; --i)
    {
      if (realpath.c_str()[i] == kPathSep)
      {
        return realpath.substr(0, i);
      }
    }

    // don't scan through the entire file system's root
    return "";
  }

  // scan for libraries in the plugin directory to load additional plugins
  void scanPluginLibraries()
  {
    // check and print plugins that are linked directly into the executable
    int nplugin = mjp_pluginCount();
    if (nplugin)
    {
      std::printf("Built-in plugins:\n");
      for (int i = 0; i < nplugin; ++i)
      {
        std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
      }
    }

    // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
    const std::string sep = "\\";
#else
    const std::string sep = "/";
#endif

    // try to open the ${EXECDIR}/plugin directory
    // ${EXECDIR} is the directory containing the simulate binary itself
    const std::string executable_dir = getExecutableDir();
    if (executable_dir.empty())
    {
      return;
    }

    const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
    mj_loadAllPluginLibraries(
        plugin_dir.c_str(), +[](const char *filename, int first, int count)
                            {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        } });
  }

  //------------------------------------------- simulation -------------------------------------------

  mjModel *LoadModel(const char *file, mj::Simulate &sim)
  {
    // this copy is needed so that the mju::strlen call below compiles
    char filename[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(filename, file);

    // make sure filename is not empty
    if (!filename[0])
    {
      return nullptr;
    }

    // load and compile
    char loadError[kErrorLength] = "";
    mjModel *mnew = 0;
    if (mju::strlen_arr(filename) > 4 &&
        !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                      mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4))
    {
      mnew = mj_loadModel(filename, nullptr);
      if (!mnew)
      {
        mju::strcpy_arr(loadError, "could not load binary model");
      }
    }
    else
    {
      mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
      // remove trailing newline character from loadError
      if (loadError[0])
      {
        int error_length = mju::strlen_arr(loadError);
        if (loadError[error_length - 1] == '\n')
        {
          loadError[error_length - 1] = '\0';
        }
      }
    }

    mju::strcpy_arr(sim.load_error, loadError);

    if (!mnew)
    {
      std::printf("%s\n", loadError);
      return nullptr;
    }

    // compiler warning: print and pause
    if (loadError[0])
    {
      // mj_forward() below will print the warning message
      std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
      sim.run = 0;
    }

    return mnew;
  }

  // simulate in background thread (while rendering in main thread)
  void PhysicsLoop(mj::Simulate &sim)
  {
    // cpu-sim syncronization point
    std::chrono::time_point<mj::Simulate::Clock> syncCPU;
    mjtNum syncSim = 0;

    // ChannelFactory::Instance()->Init(0);
    // UnitreeDds ud(d);

    // run until asked to exit
    while (!sim.exitrequest.load())
    {
      if (sim.droploadrequest.load())
      {
        sim.LoadMessage(sim.dropfilename);
        mjModel *mnew = LoadModel(sim.dropfilename, sim);
        sim.droploadrequest.store(false);

        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.dropfilename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = (mjtNum *)malloc(sizeof(mjtNum) * m->nu);
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      if (sim.uiloadrequest.load())
      {
        sim.uiloadrequest.fetch_sub(1);
        sim.LoadMessage(sim.filename);
        mjModel *mnew = LoadModel(sim.filename, sim);
        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.filename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      // sleep for 1 ms or yield, to let main thread run
      //  yield results in busy wait - which has better timing but kills battery life
      if (sim.run && sim.busywait)
      {
        std::this_thread::yield();
      }
      else
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      {
        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        // run only if model is present
        if (m)
        {
          // running
          if (sim.run)
          {
            bool stepped = false;

            // record cpu time at start of iteration
            const auto startCPU = mj::Simulate::Clock::now();

            // elapsed CPU and simulation time since last sync
            const auto elapsedCPU = startCPU - syncCPU;
            double elapsedSim = d->time - syncSim;

            // inject noise
            if (sim.ctrl_noise_std)
            {
              // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
              mjtNum rate = mju_exp(-m->opt.timestep / mju_max(sim.ctrl_noise_rate, mjMINVAL));
              mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

              for (int i = 0; i < m->nu; i++)
              {
                // update noise
                ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

                // apply noise
                d->ctrl[i] = ctrlnoise[i];
              }
            }

            // requested slow-down factor
            double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

            // misalignment condition: distance from target sim time is bigger than syncmisalign
            bool misaligned =
                mju_abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

            // out-of-sync (for any reason): reset sync times, step
            if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                misaligned || sim.speed_changed)
            {
              // re-sync
              syncCPU = startCPU;
              syncSim = d->time;
              sim.speed_changed = false;

              // run single step, let next iteration deal with timing
              mj_step(m, d);
              MaybeLogTrackingSample();
              MaybeLogCameraVibrationSample();
              stepped = true;
            }

            // in-sync: step until ahead of cpu
            else
            {
              bool measured = false;
              mjtNum prevSim = d->time;

              double refreshTime = simRefreshFraction / sim.refresh_rate;

              // step while sim lags behind cpu and within refreshTime
              while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                     mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime))
              {
                // measure slowdown before first step
                if (!measured && elapsedSim)
                {
                  sim.measured_slowdown =
                      std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                  measured = true;
                }

                // elastic band on base link
                if (param::config.enable_elastic_band == 1)
                {
                  if (elastic_band.enable_)
                  {
                    std::vector<double> x = {d->qpos[0], d->qpos[1], d->qpos[2]};
                    std::vector<double> dx = {d->qvel[0], d->qvel[1], d->qvel[2]};

                    elastic_band.Advance(x, dx);

                    d->xfrc_applied[param::config.band_attached_link] = elastic_band.f_[0];
                    d->xfrc_applied[param::config.band_attached_link + 1] = elastic_band.f_[1];
                    d->xfrc_applied[param::config.band_attached_link + 2] = elastic_band.f_[2];
                  }
                }

                // call mj_step
                mj_step(m, d);
                MaybeLogTrackingSample();
                MaybeLogCameraVibrationSample();
                stepped = true;

                // break if reset
                if (d->time < prevSim)
                {
                  break;
                }
              }
            }

            // save current state to history buffer
            if (stepped)
            {
              sim.AddToHistory();
            }
          }

          // paused
          else
          {
            // run mj_forward, to update rendering and joint sliders
            mj_forward(m, d);
            sim.speed_changed = true;
          }
        }
      } // release std::lock_guard<std::mutex>
    }
  }
} // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate *sim, const char *filename)
{
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr)
  {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m)
      d = mj_makeData(m);
    if (d)
    {
      sim->Load(m, d, filename);
      mj_forward(m, d);

      // allocate ctrlnoise
      free(ctrlnoise);
      ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
      mju_zero(ctrlnoise, m->nu);
    }
    else
    {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);

  exit(0);
}

void *UnitreeSdk2BridgeThread(void *arg)
{
  // Wait for mujoco data
  while (true)
  {
    if (d)
    {
      std::cout << "Mujoco data is prepared" << std::endl;
      break;
    }
    usleep(500000);
  }

  unitree::robot::ChannelFactory::Instance()->Init(param::config.domain_id, param::config.interface);


  int body_id = mj_name2id(m, mjOBJ_BODY, "torso_link");
  if (body_id < 0) {
    body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
  }
  param::config.band_attached_link = 6 * body_id;
  
  std::unique_ptr<UnitreeSDK2BridgeBase> interface = nullptr;
  if (m->nu > NUM_MOTOR_IDL_GO) {
    interface = std::make_unique<G1Bridge>(m, d);
  } else {
    interface = std::make_unique<Go2Bridge>(m, d);
  }
  interface->start();
  
  while (true)
  {
    sleep(1);
  }
}
//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char *title, const char *msg);
static const char *rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char *msg)
{
  rosetta_error_msg = msg;
}
#endif

// user keyboard callback
void user_key_cb(GLFWwindow* window, int key, int scancode, int act, int mods) {
  if (act==GLFW_PRESS)
  {
    if(param::config.enable_elastic_band == 1) {
      if (key==GLFW_KEY_9) {
        elastic_band.enable_ = !elastic_band.enable_;
      } else if (key==GLFW_KEY_7 || key==GLFW_KEY_UP) {
        elastic_band.length_ -= 0.1;
      } else if (key==GLFW_KEY_8 || key==GLFW_KEY_DOWN) {
        elastic_band.length_ += 0.1;
      }
    }
    if(key==GLFW_KEY_BACKSPACE) {
      mj_resetData(m, d);
      mj_forward(m, d);
    }
  }
}

// run event loop
int main(int argc, char **argv)
{

  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg)
  {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version())
  {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // Load simulation configuration
  std::filesystem::path proj_dir = std::filesystem::path(getExecutableDir()).parent_path();
  param::config.load_from_yaml(proj_dir / "config.yaml");
  param::helper(argc, argv);
  if(param::config.robot_scene.is_relative()) {
    param::config.robot_scene = proj_dir.parent_path() / "unitree_robots" / param::config.robot / param::config.robot_scene;
  }

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mj::Simulate>(
    std::make_unique<mj::GlfwAdapter>(),
    &cam, &opt, &pert, /* is_passive = */ false);

  std::thread unitree_thread(UnitreeSdk2BridgeThread, nullptr);

  // start physics thread
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), param::config.robot_scene.c_str());
  // start simulation UI loop (blocking call)
  glfwSetKeyCallback(static_cast<mj::GlfwAdapter*>(sim->platform_ui.get())->window_,user_key_cb);
  sim->RenderLoop();
  physicsthreadhandle.join();

  pthread_exit(NULL);
  return 0;
}
