"""Local continuous-video editor: durable corrections change identity components."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from collections import OrderedDict
from datetime import datetime,timezone
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import sqlite3
import threading
from urllib.parse import urlparse,parse_qs
import webbrowser
import cv2
from engine import Engine,Conflict,pair
from exporter import export
from segments import SegmentService

HERE=Path(__file__).resolve().parent


class Project:
    def __init__(self,directory,engine=None):
        self.directory=Path(directory).resolve();self.input_hash=hashlib.sha256((self.directory/'input.json').read_bytes()).hexdigest()
        manifest=json.loads((self.directory/'manifest.json').read_text(encoding='utf-8'))
        if self.input_hash!=manifest['input_sha256']:raise ValueError('Input manifest mismatch')
        if engine is None:
            if hashlib.sha256((self.directory/'features.npz').read_bytes()).hexdigest()!=manifest['features_sha256']:raise ValueError('Feature checksum mismatch')
            engine=Engine.load(directory)
        self.engine=engine;self.db=self.directory/'corrections.sqlite3';self.lock=threading.RLock();self.token=secrets.token_urlsafe(32)
        self.frame_lock=threading.RLock();self.image_cache=OrderedDict();self.cap=None
        # SQLite online backup before the additive batch-review schema migration.
        if self.db.exists():
            with self.connect() as source:
                columns={r[1] for r in source.execute('PRAGMA table_info(events)')}
                if 'payload' not in columns:
                    backup=self.directory/('corrections_before_segments_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.sqlite3')
                    with sqlite3.connect(backup) as target:source.backup(target)
        with self.connect() as c:
            c.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS labels(pair TEXT PRIMARY KEY,value TEXT NOT NULL,origin TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,utc TEXT NOT NULL,pair TEXT NOT NULL,old_value TEXT,new_value TEXT,old_origin TEXT,new_origin TEXT,undo_of INTEGER UNIQUE);''')
            if 'payload' not in {r[1] for r in c.execute('PRAGMA table_info(events)')}:c.execute('ALTER TABLE events ADD COLUMN payload TEXT')
            previous=c.execute("SELECT value FROM meta WHERE key='input_hash'").fetchone()
            if previous and previous[0]!=self.input_hash:raise ValueError('Correction project input changed')
            if not previous:
                c.execute("INSERT INTO meta VALUES('input_hash',?)",(self.input_hash,))
                c.executemany('INSERT INTO labels VALUES(?,?,?)',[(k,v,'imported_review')for k,v in engine.input['initial_labels'].items()])
        self.labels,self.revision,self.origins=self.read_labels();self.result=self.engine.solve(self.labels)
        self.export_path=None;self.export_error=None
        self.segments=SegmentService(self)

    def connect(self):return sqlite3.connect(self.db,timeout=20)
    def read_labels(self):
        with self.connect() as c:
            rows=list(c.execute('SELECT pair,value,origin FROM labels'));rev=c.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
        return {k:v for k,v,o in rows},rev,{k:o for k,v,o in rows}

    def state(self):
        with self.lock:
            with self.connect() as c:undo=c.execute('SELECT id FROM events WHERE undo_of IS NULL AND id NOT IN (SELECT undo_of FROM events WHERE undo_of IS NOT NULL) ORDER BY id DESC LIMIT 1').fetchone()
            return {'token':self.token,'revision':self.revision,'can_undo':bool(undo),'labels':self.labels,'origins':self.origins,
                    'nodes':list(self.engine.nodes.values()),'result':self.result,'fps':self.engine.input['fps'],'frames':self.engine.frame_count,
                    'width':self.engine.input['width'],'height':self.engine.input['height'],'export_path':self.export_path,'export_error':self.export_error,
                    'imported_count':len(self.engine.input['initial_labels']),'input_hash':self.input_hash,'directory':str(self.directory)}

    def apply(self,body,undo=False,segment=False):
        with self.lock:
            if body.get('revision')!=self.revision:raise Conflict('页面版本已过期，请刷新后重试。')
            with self.connect() as c:
                c.execute('BEGIN IMMEDIATE')
                current_rev=c.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
                if current_rev!=self.revision:raise Conflict('其他窗口已更新项目，请重启/刷新服务。')
                undo_of=None;review=None;updates={};origins={}
                if undo:
                    e=c.execute('SELECT id,pair,old_value,old_origin,payload FROM events WHERE undo_of IS NULL AND id NOT IN (SELECT undo_of FROM events WHERE undo_of IS NOT NULL) ORDER BY id DESC LIMIT 1').fetchone()
                    if e is None:raise ValueError('没有本编辑器内可撤销的操作；导入记录可单独改为不确定以解除约束。')
                    undo_of,token,value,origin,payload=e
                    if payload:
                        changes=json.loads(payload)['changes'];updates={k:v['old'] for k,v in changes.items()};origins={k:v['old_origin'] for k,v in changes.items()}
                    else:updates[token]=value;origins[token]=origin
                elif segment:
                    updates,review=self.segments.prepare_review(body);origins={k:'segment_'+review['action'] for k in updates}
                    token='segment:'+review['task_id'];value=review['action'];origin='segment_review'
                else:
                    a,b=body.get('a'),body.get('b');value=body.get('value');origin='local_human'
                    if a not in self.engine.nodes or b not in self.engine.nodes or self.engine.nodes[a]['frame']>=self.engine.nodes[b]['frame']:raise ValueError('请选择时间先后的两个不同检测框')
                    if value not in ('yes','no','unsure'):raise ValueError('未知判断')
                    token=pair(a,b)
                    updates[token]=value;origins[token]=origin
                proposed=dict(self.labels)
                for k,v in updates.items():
                    if v is None:proposed.pop(k,None)
                    else:proposed[k]=v
                # Compute and verify every identity constraint before committing any correction.
                solved=self.engine.solve(proposed)
                old=self.labels.get(token);old_origin=self.origins.get(token)
                changes={k:{'old':self.labels.get(k),'new':v,'old_origin':self.origins.get(k),'new_origin':origins[k]} for k,v in updates.items()}
                for k,v in updates.items():
                    if v is None:c.execute('DELETE FROM labels WHERE pair=?',(k,))
                    else:c.execute('INSERT OR REPLACE INTO labels VALUES(?,?,?)',(k,v,origins[k]))
                if review:
                    confirmed=set(review['nodes']) if review['action'] in ('approve','candidate') else set()
                    review['resolved_risk_keys']=[k for k in review['risk_keys'] if (all(n in confirmed for n in k.split('|')) or k in updates)]
                payload=json.dumps({'changes':changes,'review':review},ensure_ascii=False)
                cur=c.execute('INSERT INTO events(utc,pair,old_value,new_value,old_origin,new_origin,undo_of,payload) VALUES(?,?,?,?,?,?,?,?)',
                    (datetime.now(timezone.utc).isoformat(),token,old,value,old_origin,origin,undo_of,payload));revision=cur.lastrowid
                changed=sum(self.result['track_ids'][n]!=solved['track_ids'][n] for n in self.engine.nodes)
            self.labels,self.revision,self.origins=self.read_labels();self.result=solved
            self.save_export()
            return {'revision':revision,'pair':token,'value':value,'changed_box_ids':changed,'stats':solved['stats'],'export_path':self.export_path,'export_error':self.export_error}

    def save_export(self):
        with self.lock:
            try:self.export_path=export(self.directory,self.engine,self.result,self.labels,self.revision,reviews=self.segments.records());self.export_error=None
            except Exception as e:self.export_error=str(e)
            return {'directory':self.export_path,'error':self.export_error,'revision':self.revision}

    def frame(self,index):
        if type(index)is not int or not 0<=index<self.engine.frame_count:raise ValueError('Invalid frame')
        with self.frame_lock:
            if index in self.image_cache:self.image_cache.move_to_end(index);return self.image_cache[index]
            if self.cap is None:self.cap=cv2.VideoCapture(self.engine.input['video'])
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=self.cap.read()
            if not ok:raise ValueError('视频帧读取失败')
            self.image_cache[index]=im
            while len(self.image_cache)>4:self.image_cache.popitem(last=False)
            return im

    def crop(self,node):
        n=self.engine.nodes[node];im=self.frame(n['frame']);x1,y1,x2,y2=n['box']
        dx=max(12,(x2-x1)*.3);dy=max(12,(y2-y1)*.15);left=max(0,int(x1-dx));top=max(0,int(y1-dy))
        crop=im[top:min(im.shape[0],int(y2+dy)+1),left:min(im.shape[1],int(x2+dx)+1)].copy()
        if not crop.size:raise ValueError('Empty crop')
        cv2.rectangle(crop,(round(x1-left),round(y1-top)),(round(x2-left),round(y2-top)),(30,225,245),2)
        return cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tobytes()


def handler_for(project):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,f,*args):print(f%args,flush=True)
        def send(self,status,data,kind='application/json; charset=utf-8'):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(data)
        def video(self):
            p=Path(project.engine.input['video']);length=p.stat().st_size;start=0;end=length-1;status=200
            requested=self.headers.get('Range')
            if requested:
                m=re.fullmatch(r'bytes=(\d*)-(\d*)',requested)
                if not m or not any(m.groups()):return self.send(416,{'error':'Invalid range'})
                if m[1]:start=int(m[1]);end=min(int(m[2]),end) if m[2] else end
                else:start=max(0,length-int(m[2]))
                if start>=length or start>end:return self.send(416,{'error':'Range outside file'})
                status=206
            self.send_response(status);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes')
            self.send_header('Content-Length',str(end-start+1))
            if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{length}')
            self.end_headers()
            try:
                with p.open('rb') as f:
                    f.seek(start);remaining=end-start+1
                    while remaining:
                        block=f.read(min(1024*1024,remaining))
                        if not block:break
                        self.wfile.write(block);remaining-=len(block)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        def do_GET(self):
            try:
                url=urlparse(self.path);args=parse_qs(url.query)
                if url.path=='/':return self.send(200,(HERE/'segments.html').read_bytes(),'text/html; charset=utf-8')
                if url.path=='/pairs':return self.send(200,(HERE/'web.html').read_bytes(),'text/html; charset=utf-8')
                if url.path=='/health':return self.send(200,{'app':'aria-identity-editor','input_hash':project.input_hash,'ui_version':2})
                if url.path=='/api/segments/meta':return self.send(200,project.segments.metadata())
                if url.path=='/api/segments':return self.send(200,project.segments.page(args.get('after',[None])[0],args.get('deferred',['0'])[0]=='1'))
                if url.path=='/api/segment':return self.send(200,project.segments.detail(args['id'][0]))
                if url.path=='/api/segment/reopen':return self.send(200,project.segments.reopen(args['id'][0]))
                if url.path=='/api/segment/window':return self.send(200,project.segments.window(int(args['start'][0]),int(args['end'][0]),args['anchor'][0]))
                if url.path=='/api/segment/candidates':return self.send(200,project.segments.candidates(args['id'][0],args['anchor'][0]))
                if url.path=='/api/state':return self.send(200,project.state())
                if url.path=='/video':return self.video()
                if url.path=='/crop':return self.send(200,project.crop(args['node'][0]),'image/jpeg')
                if url.path=='/frame':return self.send(200,cv2.imencode('.jpg',project.frame(int(args['index'][0])))[1].tobytes(),'image/jpeg')
                if url.path=='/api/candidates':
                    a=args['node'][0]
                    if a not in project.engine.nodes:raise ValueError('Unknown node')
                    return self.send(200,project.engine.candidates(a,limit=30,aliases=project.result['aliases']))
                return self.send(404,{'error':'Not found'})
            except (ValueError,KeyError,IndexError)as e:self.send(400,{'error':str(e)})
            except Exception as e:self.send(500,{'error':str(e)})
        def do_POST(self):
            try:
                host=self.headers.get('Host','');origin=self.headers.get('Origin')
                if host not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}') or (origin and origin!=f'http://{host}') or self.headers.get('X-Review-Token')!=project.token:raise PermissionError('请求来源无效')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<16384:raise ValueError('Request too large')
                body=json.loads(self.rfile.read(length))
                if self.path=='/api/correct':result=project.apply(body)
                elif self.path=='/api/segment/review':result=project.apply(body,segment=True)
                elif self.path=='/api/undo':result=project.apply(body,True)
                elif self.path=='/api/export':result=project.save_export()
                else:return self.send(404,{'error':'Not found'})
                return self.send(200,result)
            except PermissionError as e:self.send(403,{'error':str(e)})
            except Conflict as e:self.send(409,{'error':str(e)})
            except (ValueError,KeyError,TypeError)as e:self.send(400,{'error':str(e)})
            except Exception as e:self.send(500,{'error':str(e)})
    return Handler


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--port',type=int,default=8768);p.add_argument('--open-browser',action='store_true');a=p.parse_args()
    # Reuse an already-running editor only if its immutable input digest matches.
    import urllib.request
    expected=hashlib.sha256((a.data/'input.json').read_bytes()).hexdigest()
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health',timeout=1)as r:health=json.load(r)
    except Exception:health=None
    if health:
        if health.get('app')!='aria-identity-editor' or health.get('input_hash')!=expected:raise RuntimeError('Port occupied by a different project')
        if a.open_browser:webbrowser.open(f'http://127.0.0.1:{a.port}')
        print('Editor already running.');return
    project=Project(a.data);project.save_export();server=ThreadingHTTPServer(('127.0.0.1',a.port),handler_for(project))
    print(json.dumps({'url':f'http://127.0.0.1:{a.port}','stats':project.result['stats'],'export':project.export_path,'export_error':project.export_error},ensure_ascii=False),flush=True)
    if a.open_browser:webbrowser.open(f'http://127.0.0.1:{a.port}')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
