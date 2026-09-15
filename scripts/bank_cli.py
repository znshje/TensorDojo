#!/usr/bin/env python3
"""HTTP client for the local bank, using only Python's standard library."""
import argparse
import json
from pathlib import Path
import sys
from urllib.error import HTTPError,URLError
from urllib.parse import quote
from urllib.request import Request,urlopen


def main():
    parser=argparse.ArgumentParser(description='Tensor Dojo 题库管理（先启动服务）')
    parser.add_argument('--url',default='http://127.0.0.1:8765')
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('list')
    for command in ('get','delete','restore'):
        p=commands.add_parser(command);p.add_argument('id')
    for command in ('create','update','validate','import'):
        p=commands.add_parser(command);p.add_argument('file',type=Path)
        if command=='import':p.add_argument('--overwrite',action='store_true');p.add_argument('--dry-run',action='store_true')
    for command in ('export','example'):
        p=commands.add_parser(command);p.add_argument('--output','-o',type=Path);p.add_argument('--force',action='store_true')
        if command=='export':p.add_argument('--ids',help='逗号分隔题目 ID；省略则导出全部有效题目')
    args=parser.parse_args();token=''
    def api(path,method='GET',body=None):
        req=Request(args.url.rstrip('/')+path,method=method,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','X-Dojo-Token':token})
        try:
            with urlopen(req,timeout=60) as response:return json.load(response)
        except HTTPError as exc:
            try:message=json.load(exc).get('error',str(exc))
            except ValueError:message=str(exc)
            raise RuntimeError(f'HTTP {exc.code}: {message}') from exc
    token=api('/api/health')['token']
    command=args.command
    if command=='list':result=api('/api/bank')
    elif command=='get':result=api('/api/bank/problems/'+quote(args.id,safe=''))
    elif command in ('export','example'):
        suffix='/api/bank/'+command
        if command=='export' and args.ids:suffix+='?ids='+quote(args.ids,safe=',')
        result=api(suffix)
    else:
        revision=api('/api/bank')['revision']
        if command in ('delete','restore'):
            suffix='/api/bank/problems/'+quote(args.id,safe='')+('/restore' if command=='restore' else '')
            result=api(suffix,'POST' if command=='restore' else 'DELETE',{'expected_revision':revision})
        else:
            content=json.loads(args.file.read_text())
            if command in ('create','update'):
                if isinstance(content,dict) and content.get('format')=='tensor-dojo-bank':
                    if len(content.get('problems',[]))!=1:raise ValueError('create/update 文件只能含一道题')
                    content=content['problems'][0]
                if isinstance(content,dict) and 'problem' in content:content=content['problem']
                suffix='/api/bank/problems'+('/'+quote(content['id'],safe='') if command=='update' else '')
                result=api(suffix,'PUT' if command=='update' else 'POST',{'problem':content,'expected_revision':revision})
            else:
                body={'bundle':content,'expected_revision':revision}
                if command=='import':body.update(overwrite=args.overwrite,dry_run=args.dry_run)
                result=api('/api/bank/'+command,'POST',body)
    output=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if getattr(args,'output',None):
        with args.output.open('w' if args.force else 'x') as file:file.write(output)
        print(f'已保存：{args.output}')
    else:print(output,end='')

if __name__=='__main__':
    try:main()
    except (RuntimeError,URLError,OSError,ValueError,KeyError,TypeError) as exc:
        print('错误：'+str(exc),file=sys.stderr);sys.exit(1)
