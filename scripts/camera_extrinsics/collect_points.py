#!/usr/bin/env python3
"""Visible-terminal assistant for nine manual contacts; no robot commands."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def points_to_collect(samples):
    order=[(r,c,'fit' if r<2 else 'check') for r in range(3) for c in range(3)]
    saved={(p['row'],p['col'],p['role']) for p in samples}
    return [p for p in order if p not in saved]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--network-interface',default='enxf8e43bbea486')
    p.add_argument('--reference-fixed',action='store_true',required=True)
    args=p.parse_args()
    def load():return json.loads((args.session/'session.json').read_text())
    s=load();board=s['reference']['board']
    if (board['cols'],board['rows'])!=(3,3):raise ValueError('This contact sequence expects the selected local 3x3 grid')
    print('只读触碰采样：不会发送机器人命令。相机和棋盘的相对位置必须保持不变。',flush=True)
    print('使用左手中指上与模型候选点对应的同一接触点；模型偏移尚未实测。',flush=True)
    for row,col,role in points_to_collect(s['samples']):
        while True:
            print(f'\n===== 当前点 ({row},{col})，用途 {role} =====',flush=True)
            print('移动手臂轻触该交点，停稳后按 Enter；随后继续保持约1秒。',flush=True)
            cmd=[sys.executable,str(Path(__file__).with_name('calibrate.py')),'touch',
                 '--session',str(args.session),'--network-interface',args.network_interface,
                 '--row',str(row),'--col',str(col),'--role',role,'--reference-fixed']
            result=subprocess.run(cmd)
            if result.returncode==0:
                saved=load()['samples']
                if any((x['row'],x['col'],x['role'])==(row,col,role) for x in saved):
                    print(f'点 ({row},{col}) 保存成功，当前 {len(saved)} 条。',flush=True);break
            if result.returncode==130:return 130
            error=args.session/'last_contact_error.json'
            if error.exists():print('本次失败原因：'+json.loads(error.read_text())['error'],flush=True)
            if input('Enter 重试当前点；输入 q 后 Enter 结束：').strip().lower()=='q':return 1
    print('\n9个接触点全部保存完成。暂不自动应用到VLN；可以检查记录并离线求解。',flush=True)
    input('按 Enter 关闭此窗口：')
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:print('\n已取消；保存的接触记录保留。');raise SystemExit(130)
    except Exception as e:
        print('[ERROR]',e)
        input('按 Enter 关闭此窗口：')
        raise SystemExit(1)
