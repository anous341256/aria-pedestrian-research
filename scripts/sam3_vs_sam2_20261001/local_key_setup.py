"""Loopback-only secret entry and Windows-account encryption; never log secrets."""
import argparse
import ctypes
from ctypes import wintypes
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import getpass
import os
import re
from pathlib import Path
import secrets
import subprocess
import threading
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/sam3_vs_sam2_20261001'
KEY = Path(os.environ['LOCALAPPDATA']) / 'CodexResearch/roboflow_api_key.dpapi'


class Blob(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]


def protect(data, decrypt=False):
    buf = ctypes.create_string_buffer(data)
    incoming = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    outgoing = Blob()
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    fn = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    fn.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    fn.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if not fn(ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(outgoing)):
        raise RuntimeError('Windows encryption failed')
    try:
        return ctypes.string_at(outgoing.data, outgoing.size)
    finally:
        kernel.LocalFree(outgoing.data)


def write_status(value):
    (OUT/'credential_status.json').write_text(json.dumps(value), encoding='utf-8')


def serve():
    token = secrets.token_urlsafe(32)
    server = HTTPServer(('127.0.0.1', 0), Handler)
    server.token = token
    server.origin = f'http://127.0.0.1:{server.server_port}'
    server.saved = False
    write_status(dict(status='waiting_for_local_browser_input',url=server.origin+'/?token='+token))
    server.serve_forever(poll_interval=.2)
    server.server_close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def page(self, text, code=200):
        data = text.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        supplied = query.get('token',[''])[0]
        if not secrets.compare_digest(supplied, self.server.token):
            self.page('This setup link is invalid.', 403)
            return
        self.page('''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Roboflow 本机配置</title>
<style>body{background:#f2f4f7;color:#17202d;font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;padding:50px 24px}
main{max-width:570px;margin:auto;padding:36px;background:white;border-radius:16px;border:1px solid #dce1e8}h1{font-size:25px;margin-top:0}p{line-height:1.8;color:#4b5668}
label{display:block;margin:28px 0 10px;font-weight:600}input[type=password]{box-sizing:border-box;width:100%;font-size:20px;padding:12px;border:1px solid #9aa6b6;border-radius:8px}
button{margin-top:22px;border:0;border-radius:8px;background:#1767d5;color:white;padding:13px 24px;font-size:16px;cursor:pointer}.small{font-size:13px}</style>
<main><h1>Roboflow 本机密钥配置</h1><p>粘贴你刚找到的 Private API Key。保存后，我会检查 SAM 3 视频模型的获取权限。</p>
<form method="post" action="/save"><input type="hidden" name="token" value="TOKEN">
<label for="key">Private API Key</label><input id="key" name="key" type="password" autocomplete="off" required autofocus>
<button type="submit">加密保存到本机</button></form>
<p class="small">密钥使用当前 Windows 账号加密保存。本页面仅在你的电脑上运行，密钥不会显示在聊天中。</p></main></html>'''.replace('TOKEN', self.server.token))

    def do_POST(self):
        if self.path != '/save' or self.headers.get('Origin') != self.server.origin:
            self.page('Invalid request.',403)
            return
        size = int(self.headers.get('Content-Length',0))
        if not 0 < size < 8192:
            self.page('Invalid input length.',400)
            return
        fields = urllib.parse.parse_qs(self.rfile.read(size).decode('utf-8'))
        if not secrets.compare_digest(fields.get('token',[''])[0], self.server.token):
            self.page('Invalid setup token.',403)
            return
        value = fields.get('key',[''])[0].strip()
        if not value or any(c.isspace() for c in value):
            self.page('API Key 格式不正确，请返回并重新粘贴。',400)
            return
        try:
            KEY.parent.mkdir(parents=True,exist_ok=True)
            KEY.write_bytes(protect(value.encode('utf-8')))
            write_status(dict(status='saved_locally_encrypted',format='windows_dpapi_binary_v1'))
            self.server.saved = True
            self.page('<html lang="zh"><meta charset="utf-8"><body style="font:20px system-ui;padding:60px"><h2>密钥已加密保存</h2><p>接下来会检查 SAM 3 模型权限。你可以回到对话。</p></body></html>')
            threading.Thread(target=self.server.shutdown,daemon=True).start()
        except Exception:
            self.page('保存失败，请回到对话报告“本机保存失败”。',500)


def run_check(download=False):
    key = protect(KEY.read_bytes(),decrypt=True).decode('utf-8')
    script = '/mnt/e/New_research/scripts/sam3_vs_sam2_20261001/' + ('download_roboflow_model.py' if download else 'check_roboflow_access.py')
    if download:
        child = subprocess.Popen(['wsl.exe','-e','/opt/sam3/bin/python','-u',script],
                                 stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 encoding='utf-8',bufsize=1)
        child.stdin.write(key+'\n');child.stdin.close()
        for line in child.stdout:
            print(re.sub(r'https?://\S+','[URL]',line.replace(key,'[REDACTED]')).rstrip(),flush=True)
        print('DOWNLOAD_EXIT',child.wait(),flush=True)
        return
    child = subprocess.run(['wsl.exe','-e','/opt/sam3/bin/python',script],input=key+'\n',
                           encoding='utf-8',capture_output=True,timeout=150)
    # The child emits only a sanitised, whitelisted result object.
    print(child.stdout)
    if child.returncode:
        print('ACCESS_CHECK_EXIT',child.returncode)
        cleaned = child.stderr.replace(key,'[REDACTED]')
        print(re.sub(r'https?://\S+','[URL]',cleaned)[-1500:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['serve','check','store-check','download'])
    args=parser.parse_args()
    if args.action == 'serve':
        serve()
    elif args.action == 'store-check':
        key_value = getpass.getpass('Roboflow private key (hidden input): ').strip()
        if not key_value:
            raise SystemExit('No key supplied')
        KEY.parent.mkdir(parents=True,exist_ok=True)
        KEY.write_bytes(protect(key_value.encode('utf-8')))
        key_value = None
        write_status(dict(status='saved_locally_encrypted',format='windows_dpapi_binary_v1'))
        print('KEY_SAVED_ENCRYPTED')
        run_check()
    else:
        run_check(download=args.action=='download')
