# D415 → VLN 模型静态联调

入口：`g1_camera_model_probe.py`。在笔记本手动采集四方向RGB-D，使用正式`CombinedModelClient`和G3 Session协议请求一次决策。程序不导入Unitree运控后端，不发送速度、站立或停车命令，不调用iPlanner。

相机可以暂时放在桌子上。这个阶段不需要胸前安装外参或SLAM位姿，也不验证真实运动、世界目标位置或完整监督闭环。

机器人实际转四段、每段停车后自动采集的步骤见 [真机旋转采集](README_ROTATION_CAPTURE.md)。相机仍在桌上时可以验证时序，不能把图像当作真实四方向视野。

## 1. 笔记本启动相机服务

D415通过USB连接笔记本，确认没有其他程序占用。以下命令在笔记本终端执行，不是机器人SSH终端：

```bash
/home/yile/projects/g1_camera_sdk/local/build-probe-x86_64/g1_d435i_stream \
  --bind 127.0.0.1 --port 8765 --serial 211222063932 \
  --settings-dir /home/yile/projects/g1_camera_sdk/d415_camera_logs
```

看到`RGB-D ready`后，保持服务运行。同一时间只能连接一个接收程序，先关闭旧的`network_camera.py`预览。

## 2. 先采集四方向

另开笔记本终端：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source /home/yile/projects/.venvs/unitree_g1/bin/activate
python scripts/isaacsim_lavira_g3_interface_g1/g1_camera_model_probe.py \
  --capture-only --output /home/yile/projects/g1_camera_sdk/d415_panorama_01
```

窗口显示RGB和对齐深度。以开始的朝向为前方：前方→向左约90°→后方→右方，保持相机位置尽量不变。每个方向停稳后，在预览窗口按空格或Enter保存并进入下一方向。`q`、Esc、关闭窗口或Ctrl+C退出。`--headless`取消窗口，改用终端Enter确认。

输出目录必须不存在。每张保存`current_方向.png`和`方向_depth_m.npy`，清单`panorama.json`记录K、帧号、时间及原始相机元数据；`probe_report.json`记录本次测试范围。中断后的部分数据会保留，但不能作为完整四方向重放。不要复制一张图冒充四个方向。

## 3. 上传已保存的真实四方向图像

先启动你现有的G3模型服务或SSH隧道，使完整决策地址可访问。默认地址是`http://127.0.0.1:18765/v1/lavira/decision`；机器人连接不等于模型服务器已连接。

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_camera_model_probe.py \
  --panorama-dir /home/yile/projects/g1_camera_sdk/d415_panorama_01 \
  --instruction "Go to the door." \
  --model-url http://127.0.0.1:18765/v1/lavira/decision \
  --output /home/yile/projects/g1_camera_sdk/d415_model_test_01
```

重放不连接相机，因此采集成功后可关闭相机服务。重复上传时换一个新的输出目录和符合画面的指令。

默认流程为`health → start_session → decision → end_session`。指令在会话开始时提交，决策上传四张真实RGB；深度和内参保留本地。终端打印完整响应，保存`request_metadata.json`、`response.json`、会话日志和`probe_report.json`；模型返回bbox时保存`decision_bbox.png`。

不发送`motion_window`或`action_complete`。静态决策验证完成后以`CANCELLED/static_probe_no_robot_execution`结束会话，不能把它记成导航任务成功。服务不可达或协议不匹配会报错，已采集图像不会删除。失败阶段记录在报告里。

若需要采集后立即上传，可不加`--capture-only`或`--panorama-dir`，直接传`--instruction`和`--model-url`。`--legacy-model`仅用于旧版无Session服务，正常G3使用默认模式。

## 验证范围

新增入口已通过6项离线测试：图像颜色/深度/内参保存重放、缺失方向拒绝、非法深度拒绝、Ctrl+C释放相机、测试HTTP服务四图上传与bbox输出、G3结束会话且不提交虚假执行报告。测试HTTP服务不是远端模型，现场接口联调结果单独记录如下。

## 2026-10-06 现场静态联调

D415通过USB连接笔记本，四方向手动采集完成：
`/home/yile/projects/g1_camera_sdk/d415_panorama_01`，帧号3821、3966、4037、4153。
RGB、米制对齐深度和K的保存重放检查通过。

通过SSH隧道连接现有远端G3服务，将这四张真实RGB提交给
`http://127.0.0.1:18765/v1/lavira/decision`，指令 `Go to the door.`。
结果目录：`/home/yile/projects/g1_camera_sdk/d415_model_test_01`。
`probe_report.json` 为 `PASS/static_model_decision`，服务返回
`NAVIGATE / left / door`，bbox为 `[490,156,640,468]`；完整响应与bbox图已保存。
响应的 `stage_progress.model_backend` 为 `local_navigator_shadow`，作为服务返回字段记录。

health、start、decision、end接口均完成。会话以
`CANCELLED/static_probe_no_robot_execution`结束，未发送机器人命令、motion_window
或action_complete。bbox图包含前景椅子/物品及部分背景，尚未验证bbox深度投影、
iPlanner轨迹、机器人执行和最终任务成功。
