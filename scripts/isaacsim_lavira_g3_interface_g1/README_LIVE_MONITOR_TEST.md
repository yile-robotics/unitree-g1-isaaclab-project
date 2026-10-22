# D415 实时画面与单次旋转决策测试

先准备相机、模型连接和iPlanner，再打开监控窗口。这些准备命令不控制机器人。
机器人旋转命令单独列在最后，由用户手动执行；监控窗口不会触发它，也不会执行模型给出的行走动作。

## 终端1：相机

已有服务时保持运行，不重复启动。D415通过USB接笔记本，输出RGB和对齐深度1280×720、15fps：

```bash
/home/yile/projects/g1_camera_sdk/local/build-probe-x86_64/g1_d435i_stream \
  --bind 127.0.0.1 --port 8765 --serial 211222063932 \
  --resolution 720p \
  --settings-dir /home/yile/projects/g1_camera_sdk/d415_camera_logs
```

## 终端2：iPlanner

已有8888服务时保持运行。这里只启动规划服务，不启动仿真或运动控制：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
IPLANNER_PYTHON=/home/yile/miniconda3/envs/isaacsim/bin/python \
  bash scripts/isaacsim_lavira_g3_interface_g1/run_iplanner_local.sh
```

## 终端3：模型隧道

远端G3服务已启动时，只需要SSH转发。18765已有可用转发时不重复运行：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18765:127.0.0.1:8765 wangchu@131.159.60.188
```

## 终端4：检查并打开监控

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source /home/yile/projects/.venvs/unitree_g1/bin/activate
python scripts/isaacsim_lavira_g3_interface_g1/check_live_monitor.py
```

应看到 `status: PASS`、RGB `[720,1280,3]`、深度 `[720,1280]`、iPlanner `model_loaded: true`。
检查只读取相机新帧和两个HTTP健康接口，不上传模型决策，不导入DDS。

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_iplanner_viewer.py
```

窗口始终更新RGB与深度；等待新模型结果时没有历史框。每次新决策自动替换同一实时RGB中的框和标题。
模型框底边中心和当前深度用于iPlanner路径预览。开始下一轮旋转前，旧框、目标与路径清除，
RGB/深度继续显示；停车且新响应返回后恢复显示。没有目标跟踪，不在运动中请求模型。
`H`隐藏旧决策（其他遥控运动前使用），`S`保存当前预览，`Q`/Esc退出。

框属于模型选中的全景方向。若结果为left/behind/right，当前前视画面只提示方向，不把其他视图的像素框错画在前视RGB上；
本单次决策探针不会为了显示框自动转向。STOP/BACKTRACK/SAFE_STOP/无框结果清除旧框。
即使前方框也不补偿全景回正角度偏差或相机移动，因此只用于检查决策显示。

路径是名义相机局部坐标预览，尚未应用torso/map外参，不等于可直接执行的机器人路线。
监控窗口本身不发送机器人命令。

## 终端5：用户手动运行旋转采图和一次决策

这一步会让真机旋转。先按[旋转采集说明](README_ROTATION_CAPTURE.md#2-先开rviz再启动重定位)准备SLAM重定位，
确认D415固定在胸前并随机器人转动。保持监控窗口打开，用户在新终端手动执行：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source /home/yile/projects/.venvs/unitree_g1/bin/activate
python scripts/isaacsim_lavira_g3_interface_g1/g1_rotation_model_probe.py \
  --network-interface enxf8e43bbea486 \
  --instruction "向前走到一个黑色的行李箱前" \
  --execute
```

四段左转，角速度命令0.8rad/s，IMU相对75°提前停车，等待0.5秒后采图；实际每段停稳角度不是严格90°。
完整采集后只请求一次决策，不执行模型动作。每次运行自动生成新的 `outputs/d415_rotation_model/run_.../`，
所以无需手改输出目录。下一轮仍手动运行同一命令；窗口持续监听各轮新结果。
Ctrl+C复用原入口停车清理；旋转失败或采集不完整时不请求模型，也不恢复上一轮框。

原始图像位于 `panorama/current_forward.png` 等，响应位于 `model/response.json`，
离线核对框为 `model/decision_bbox.png`，流程报告为 `flow_report.json`。
监控输出在 `outputs/iplanner_viewer/run_.../`，`latest_target.json`与`latest_plan.json`记录框和路径来源。
