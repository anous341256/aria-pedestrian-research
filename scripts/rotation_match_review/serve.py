"""Loopback-only review UI, append-only SQLite audit and reproducible exports."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import sqlite3
import threading
from urllib.parse import urlparse, parse_qs
import webbrowser
import cv2
from core import METHODS, NAMES, VALUES, pair_key, answers_for, question_state, summarize

HERE=Path(__file__).resolve().parent


class Review:
    def __init__(self,directory):
        self.directory=Path(directory).resolve()
        blob=(self.directory/'dataset.json').read_bytes()
        self.digest=hashlib.sha256(blob).hexdigest()
        self.data=json.loads(blob)
        manifest=json.loads((self.directory/'manifest.json').read_text(encoding='utf-8'))
        if manifest['dataset_sha256']!=self.digest:raise ValueError('Dataset checksum mismatch')
        self.queries={q['id']:q for q in self.data['queries']}
        self.frames={i for q in self.queries.values() for i in (q['frame'],q['previous_frame'])}
        self.db=self.directory/'review.sqlite3'
        self.lock=threading.RLock();self.token=secrets.token_urlsafe(32)
        with self.connect() as c:
            c.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS labels(query_id TEXT,candidate INTEGER,value TEXT NOT NULL,PRIMARY KEY(query_id,candidate));
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,utc TEXT NOT NULL,query_id TEXT NOT NULL,candidate INTEGER NOT NULL,old_value TEXT,new_value TEXT,undo_of INTEGER UNIQUE);''')
            prior=c.execute("SELECT value FROM meta WHERE key='dataset_sha256'").fetchone()
            if prior and prior[0]!=self.digest:raise ValueError('Labels belong to a different dataset')
            c.execute("INSERT OR IGNORE INTO meta VALUES('dataset_sha256',?)",(self.digest,))

    def connect(self):
        return sqlite3.connect(self.db,timeout=10)

    def snapshot(self):
        with self.connect() as c:
            labels={pair_key(q,i):v for q,i,v in c.execute('SELECT query_id,candidate,value FROM labels')}
            revision=c.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
            undo=c.execute('SELECT id FROM events WHERE undo_of IS NULL AND id NOT IN (SELECT undo_of FROM events WHERE undo_of IS NOT NULL) ORDER BY id DESC LIMIT 1').fetchone()
        return labels,revision,bool(undo)

    def state(self):
        with self.lock:
            labels,revision,can_undo=self.snapshot()
            return {'dataset':self.data,'labels':labels,'revision':revision,'can_undo':can_undo,
                    'token':self.token,'summary':summarize(self.data['queries'],labels),'directory':str(self.directory)}

    def change(self,body,undo=False):
        with self.lock, self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            revision=c.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
            if body.get('revision')!=revision:raise RuntimeError('审核记录已变化，请刷新后重试。')
            undo_of=None
            if undo:
                row=c.execute('SELECT id,query_id,candidate,old_value FROM events WHERE undo_of IS NULL AND id NOT IN (SELECT undo_of FROM events WHERE undo_of IS NOT NULL) ORDER BY id DESC LIMIT 1').fetchone()
                if row is None:raise ValueError('没有可撤销的判断')
                undo_of,qid,i,value=row
            else:
                qid=body.get('query_id');i=body.get('candidate');value=body.get('value')
                if qid not in self.queries or type(i) is not int or not 0<=i<len(self.queries[qid]['candidates']):raise ValueError('未知候选')
                if value not in VALUES:raise ValueError('未知判断')
            old=c.execute('SELECT value FROM labels WHERE query_id=? AND candidate=?',(qid,i)).fetchone()
            old=old[0] if old else None
            if value is None:c.execute('DELETE FROM labels WHERE query_id=? AND candidate=?',(qid,i))
            else:c.execute('INSERT OR REPLACE INTO labels VALUES(?,?,?)',(qid,i,value))
            c.execute('INSERT INTO events(utc,query_id,candidate,old_value,new_value,undo_of) VALUES(?,?,?,?,?,?)',
                      (datetime.now(timezone.utc).isoformat(),qid,i,old,value,undo_of))
        # Persist exports after SQLite transaction; DB is always the authoritative store.
        self.export()
        return {'query_id':qid,'candidate':i,'value':value}

    def export(self):
        with self.lock:
            labels,revision,_=self.snapshot()
            summary=summarize(self.data['queries'],labels)
            dest=self.directory/'exports';dest.mkdir(exist_ok=True)
            pairs=[];rows=[]
            for q in self.data['queries']:
                a=answers_for(q,labels);pos=[i for i,v in a.items() if v=='yes']
                for i,v in a.items():
                    if v is not None:pairs.append(dict(query_id=q['id'],frame=q['frame'],source_index=q['source_index'],candidate=i,decision=v,cohort=q['cohort']))
                for m in METHODS:
                    d=q['methods'][m];i=d['top1']
                    neg=[j for j,v in a.items() if v=='no']
                    rows.append(dict(query_id=q['id'],cohort=q['cohort'],frame=q['frame'],seconds=q['seconds'],
                        angle_deg=q['angle_deg'],source_diagonal_px=((q['source_box'][2]-q['source_box'][0])**2+(q['source_box'][3]-q['source_box'][1])**2)**.5,
                        candidates=len(q['candidates']),rgb_status=q['rgb_status'],method=m,top1=i,top1_decision=a[i],
                        top1_tie=d['tie_top1'],confirmed_candidates=';'.join(map(str,pos)),multiple_positive=len(pos)>1,
                        confirmed_best_rank=min(d['order'].index(j)+1 for j in pos) if pos else None,
                        confirmed_best_iou_rank=min(d['iou_order'].index(j)+1 for j in pos) if pos else None,
                        confirmed_center_error_px=min(d['distances_px'][j] for j in pos) if pos else None,
                        margin_to_reviewed_negative=(min(d['normalized_distances'][j] for j in neg)-min(d['normalized_distances'][j] for j in pos)) if pos and neg else None))
            def csv_text(items,fields):
                buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=fields);w.writeheader();w.writerows(items);return buf.getvalue()
            grouped=[]
            for cohort in ('random','disagreement'):
                for grouping in ('rotation','scale','candidate_count','rgb_status'):
                    def group_key(row):
                        if grouping=='rotation':return 'low_le_3.316deg' if row['angle_deg']<=3.316432277567482 else ('mid_le_5.323deg' if row['angle_deg']<=5.322703585296192 else 'high_gt_5.323deg')
                        if grouping=='scale':return 'small_diag_lt160px' if row['source_diagonal_px']<160 else ('medium_diag_lt400px' if row['source_diagonal_px']<400 else 'large_diag_ge400px')
                        if grouping=='candidate_count':return 'le5' if row['candidates']<=5 else ('6to10' if row['candidates']<=10 else 'gt10')
                        return row['rgb_status']
                    for group in sorted({group_key(r) for r in rows if r['cohort']==cohort}):
                        for method in METHODS:
                            selected=[r for r in rows if r['cohort']==cohort and r['method']==method and group_key(r)==group]
                            values=[r['top1_decision'] for r in selected];yes=values.count('yes');no=values.count('no')
                            grouped.append(dict(cohort=cohort,grouping=grouping,group=group,method=method,n=len(values),yes=yes,no=no,
                                unsure=values.count('unsure'),pending=values.count(None),judged_accuracy=yes/(yes+no) if yes+no else None))
            outputs={'pair_labels.csv':csv_text(pairs,['query_id','frame','source_index','candidate','decision','cohort']),
                     'query_method_results.csv':csv_text(rows,list(rows[0]) if rows else ['query_id']),
                     'grouped_results.csv':csv_text(grouped,list(grouped[0]) if grouped else ['cohort','group']),
                     'summary.json':json.dumps({'dataset_sha256':self.digest,'revision':revision,'summary':summary},ensure_ascii=False,indent=2),
                     'labels.json':json.dumps({'dataset_sha256':self.digest,'revision':revision,'labels':labels},ensure_ascii=False,indent=2)}
            report=['# 三条件单步关联：人工审核进度','',f'数据摘要：`{self.digest}`；审核版本：{revision}。',
                    '', '没有人工判断的候选不是错误，也不是正确。部分审核结果不代表全样本。',
                    '“同人/非同人”直接评价各方法首选；确认有正例的子集才支持两帧均有检测的条件分析。',
                    '排名只相对于已确认正例，重复框可能产生多个正例；未全部核验的候选不能当作负例。','']
            for cohort,title in [('random','独立随机复核'),('disagreement','方法分歧诊断（不能并入总体准确率）')]:
                s=summary[cohort];report += [f'## {title}','',f"样本 {s['queries']}，已判断框对 {s['reviewed_pairs']}，确认有正例 {s['confirmed_positive_queries']}。",'',
                    '| 方法 | 是 | 否 | 不确定 | 未审核 | 明确判断子集准确率 | 全样本准确率界限 |','|---|---:|---:|---:|---:|---:|---:|']
                for m in METHODS:
                    x=s['methods'][m];percent=lambda v:'—' if v is None else f'{v:.1%}'
                    report.append(f"| {NAMES[m]} | {x['yes']} | {x['no']} | {x['unsure']} | {x['pending']} | {percent(x['judged_accuracy'])} | {percent(x['all_sample_lower'])}–{percent(x['all_sample_upper'])} |")
                report += ['', '配对改善与按 5 秒块重采样的区间见 summary.json；不足四个已审核时间块时不计算区间。',
                           '按旋转量、框尺度、候选数量、RGB 回退状态分组见 grouped_results.csv。','']
            report += ['## 限制','', 'RGB 是现有 VideoStab 缓存（472/600 帧回退），不是 RGB 方法的最优上界。',
                'Aria 使用离线姿态插值，不是严格实时输入实验；只有一条录制，没有泛化结论。',
                '当前不训练任何模型，不做强化学习；反馈是可审计的成对监督数据，未自动传播为身份标注。',
                '随机抽样只覆盖具有有效姿态、可投影源框及当前候选的查询；不能代表漏检、完全遮挡及姿态失败片段。']
            outputs['REPORT_ZH.md']='\n'.join(report)+'\n'
            # Replace only our derived snapshots atomically; append-only events remain in SQLite.
            for name,content in outputs.items():
                target=dest/name;temp=dest/(name+'.tmp');temp.write_text(content,encoding='utf-8-sig' if name.endswith('.csv') else 'utf-8');temp.replace(target)
            return str(dest)


def handler_for(review):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args):
            print(fmt%args,flush=True)

        def send(self,status,data,content_type='application/json; charset=utf-8'):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(data)

        def do_GET(self):
            try:
                path=urlparse(self.path);args=parse_qs(path.query)
                if path.path=='/':return self.send(200,(HERE/'web.html').read_bytes(),'text/html; charset=utf-8')
                if path.path=='/api/state':return self.send(200,review.state())
                if path.path=='/health':return self.send(200,{'app':'aria-rotation-match-review','dataset':review.digest})
                if path.path=='/frame':
                    frame=int(args['id'][0])
                    if frame not in review.frames:raise ValueError('Frame not in review set')
                    return self.send(200,(review.directory/f'frames/{frame:04d}.jpg').read_bytes(),'image/jpeg')
                if path.path=='/crop':
                    q=review.queries[args['query'][0]];which=args['which'][0]
                    if which=='source':frame=q['previous_frame'];box=q['source_box']
                    else:
                        i=int(which)
                        if not 0<=i<len(q['candidates']):raise ValueError('Bad candidate')
                        frame=q['frame'];box=q['candidates'][i]['box']
                    im=cv2.imread(str(review.directory/f'frames/{frame:04d}.jpg'))
                    x1,y1,x2,y2=box;padx=max(12,(x2-x1)*.35);pady=max(12,(y2-y1)*.15)
                    left=max(0,int(x1-padx));top=max(0,int(y1-pady))
                    crop=im[top:min(im.shape[0],int(y2+pady)+1),left:min(im.shape[1],int(x2+padx)+1)].copy()
                    if not crop.size:raise ValueError('Empty crop')
                    color=(255,207,0) if which=='source' else (32,186,255)
                    cv2.rectangle(crop,(round(x1-left),round(y1-top)),(round(x2-left),round(y2-top)),color,2)
                    ok,jpg=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95])
                    if not ok:raise RuntimeError('Encode error')
                    return self.send(200,jpg.tobytes(),'image/jpeg')
                return self.send(404,{'error':'Not found'})
            except (ValueError,KeyError,IndexError) as e:self.send(400,{'error':str(e)})
            except Exception as e:self.send(500,{'error':str(e)})

        def do_POST(self):
            try:
                host=self.headers.get('Host','')
                if host not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):raise PermissionError('Invalid host')
                origin=self.headers.get('Origin')
                if origin and origin!=f'http://{host}':raise PermissionError('Cross-origin write rejected')
                if self.headers.get('X-Review-Token')!=review.token:raise PermissionError('Invalid review token')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<4096:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(length))
                if self.path=='/api/label':result=review.change(body)
                elif self.path=='/api/undo':result=review.change(body,undo=True)
                elif self.path=='/api/export':result={'directory':review.export()}
                else:return self.send(404,{'error':'Not found'})
                return self.send(200,result)
            except PermissionError as e:self.send(403,{'error':str(e)})
            except RuntimeError as e:self.send(409,{'error':str(e)})
            except (ValueError,KeyError,TypeError) as e:self.send(400,{'error':str(e)})
            except Exception as e:self.send(500,{'error':str(e)})
    return Handler


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--port',type=int,default=8766);p.add_argument('--open-browser',action='store_true')
    args=p.parse_args();review=Review(args.data)
    try:server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(review))
    except OSError:
        import urllib.request
        try:
            health=json.load(urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health',timeout=2))
            if health.get('app')!='aria-rotation-match-review' or health.get('dataset')!=review.digest:raise ValueError('Port occupied by another app')
        except Exception:raise RuntimeError('Port occupied. Do not close unrelated applications; choose another --port.')
        if args.open_browser:webbrowser.open(f'http://127.0.0.1:{args.port}')
        print('Matching review server already running.');return
    review.export()
    print(f'Review: http://127.0.0.1:{args.port} | data: {review.directory}',flush=True)
    if args.open_browser:webbrowser.open(f'http://127.0.0.1:{args.port}')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
