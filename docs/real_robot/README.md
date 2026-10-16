# 真机相机、标定与规划工作记录

更新：2026-10-15，Europe/Berlin。最近两次工作日期为10月13日与10月15日；10月8～9日的标定采集历史保留在各工具README中。

| 工作 | 已确认 | 尚未确认 |
|---|---|---|
| [10月13日：板几何审查与外参重算](2026-10-13/README.md) | 从原图重检测、原样保留关节与验证分组；两套候选分别求解 | 指尖真实偏移、原装RGB到torso参考及最终外参绝对精度 |
| [10月15日：旋转、模型、iPlanner与障碍分析](2026-10-15/README.md) | 四段旋转后拍摄、一次G3决策、真实iPlanner请求与实时显示 | 目标/路径净空、实际避障、轨迹跟踪与完整任务成功 |

**当前结论：实时规划链路已打通，实际避障能力待验证，尚未进行本次 iPlanner 路径的行走执行。**

代码入口：

| 入口 | 作用 |
|---|---|
| [camera_d415](../../scripts/isaacsim_lavira_g3_interface_g1/camera_d415/README.md) | 官方SDK取图、自定义TCP传输、RGB-D检查 |
| [camera_extrinsics](../../scripts/camera_extrinsics/README.md) | 手触碰板＋实际关节角＋URDF正运动学 |
| [camera_pair_extrinsics](../../scripts/camera_pair_extrinsics/README.md) | 原装D435i与新D415的相对外参 |
| [g1_rotation_model_probe.py](../../scripts/isaacsim_lavira_g3_interface_g1/g1_rotation_model_probe.py) | 真机四段旋转采图，完整完成后只请求一次模型决策 |
| [g1_iplanner_viewer.py](../../scripts/isaacsim_lavira_g3_interface_g1/g1_iplanner_viewer.py) | 实时人工选点规划；或读取已保存模型框重放规划 |

相机目前是胸前D415，USB接笔记本，默认640×480、15fps；当前接收代码目录统一为
`camera_d415`。已有可执行文件仍名为 `g1_d435i_stream`，由序列号决定实际设备。
新旧相机均通过自己的TCP服务传图，不经ROS图像话题。TCP配对检查不等于硬件曝光同步。

本次提交包含代码、操作文档和约1.9MB精选结果证据，没有加入完整原始采集目录或模型权重。
[evidence_manifest.json](evidence_manifest.json)记录所选文件的本机来源和SHA256；原始采集数据仍在
`/home/yile/projects/g1_camera_sdk/` 和项目 `outputs/` 下。JSON中的原始本机路径保留用于追溯，
不会因上传GitHub而变成可在其他电脑上直接读取的数据。

本轮离线回归：真机/G3目录200项、手触碰32项、双相机12项，合计244项通过。
离线测试使用替身，不代表实物避障或导航成功。相机SDK、固件、显卡驱动和仿真控制流程未改。
