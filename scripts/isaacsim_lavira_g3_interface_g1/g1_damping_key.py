#!/usr/bin/env python3
"""兼容入口：同终端空格+Enter阻尼，或显式清除本机锁存；不再创建GUI窗口。"""
import argparse
from unified_vln.damping_safety import latch_path
from unified_vln.terminal_damping import main as terminal_main


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network-interface',default='enxf8e43bbea486')
    parser.add_argument('--clear-latch',action='store_true',help='仅清除本机导航锁存，不切换机器人模式')
    args=parser.parse_args()
    if args.clear_latch:
        print('仅在所有导航/旋转程序已停止后清除。不会恢复机器人模式。')
        if input('输入 CLEAR 再按Enter确认：').strip()!='CLEAR':return 1
        latch_path(args.network_interface).unlink(missing_ok=True)
        print('本机锁存已清除；已锁存的旧进程仍需退出后重启。');return 0
    return terminal_main(['--network-interface',args.network_interface])

if __name__=='__main__':raise SystemExit(main())
