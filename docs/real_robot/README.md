# 真机相机、标定与规划工作记录

更新：2026-10-22，Europe/Berlin。10月20日完整真机测试与Physical、Semantic分层结果已补充；历史实验仍保留。

| 工作 | 已确认 | 尚未确认 |
|---|---|---|
| [10月20日：完整两阶段VLN与分层审查](2026-10-20/README.md) | 前进、真实左转、第二目标接近、Physical与Semantic记录、STOP Gate及Session SUCCESS | 精确停靠、避障、持续异常与恢复 |
| [10月13日：板几何审查与外参重算](2026-10-13/README.md) | 从原图重检测、原样保留关节与验证分组；两套候选分别求解 | 指尖真实偏移、原装RGB到torso参考及最终外参绝对精度 |
| [10月15日：旋转、模型、iPlanner与障碍分析](2026-10-15/README.md) | 四段旋转后拍摄、一次G3决策、真实iPlanner请求与实时显示 | 目标/路径净空、实际避障、轨迹跟踪与完整任务成功 |

**当前结论：已完成一次真机两阶段VLN任务，含iPlanner路径跟随和语义停止；实际避障与精确停靠仍待验证。**

代码入口：

| 入口 | 作用 |
|---|---|
| [camera_d415](../../scripts/isaacsim_lavira_g3_interface_g1/camera_d415/README.md) | 官方SDK取图、自定义TCP传输、RGB-D检查 |
| [camera_extrinsics](../../scripts/camera_extrinsics/README.md) | 手触碰板＋实际关节角＋URDF正运动学 |
| [camera_pair_extrinsics](../../scripts/camera_pair_extrinsics/README.md) | 原装D435i与新D415的相对外参 |
| [g1_rotation_model_probe.py](../../scripts/isaacsim_lavira_g3_interface_g1/g1_rotation_model_probe.py) | 真机四段旋转采图，完整完成后只请求一次模型决策 |
| [g1_iplanner_viewer.py](../../scripts/isaacsim_lavira_g3_interface_g1/g1_iplanner_viewer.py) | 实时人工选点规划；或读取已保存模型框重放规划 |

上述10月13/15日实验使用胸前D415、USB接笔记本、640×480/15fps。2026-10-16起默认切到1280×720/15fps，实测与启动说明见[当前D415记录](../../scripts/isaacsim_lavira_g3_interface_g1/camera_d415/README_D415_LOCAL.md)。当前接收代码目录统一为
`camera_d415`。已有可执行文件仍名为 `g1_d435i_stream`，由序列号决定实际设备。
新旧相机均通过自己的TCP服务传图，不经ROS图像话题。TCP配对检查不等于硬件曝光同步。

历史提交包含约1.9MB精选证据；本次新增10月20日的JSON、事件日志和目标图，未加入完整原始采集目录或模型权重。
[evidence_manifest.json](evidence_manifest.json)记录所选文件的本机来源和SHA256；原始采集数据仍在
`/home/yile/projects/g1_camera_sdk/` 和项目 `outputs/` 下。JSON中的原始本机路径保留用于追溯，
不会因上传GitHub而变成可在其他电脑上直接读取的数据。

本轮离线回归：真机/G3目录200项、手触碰32项、双相机12项，合计244项通过。
离线测试使用替身，不代表实物避障或导航成功。相机SDK、固件、显卡驱动和仿真控制流程未改。
