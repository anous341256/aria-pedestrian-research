"""Download the SAM3 package granted to this Roboflow key; no video is uploaded."""
import base64
import datetime
import hashlib
import json
import re
from pathlib import Path
import sys
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'models/sam3_roboflow_hf'


def file_hashes(path):
    md5,sha=hashlib.md5(),hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            md5.update(block);sha.update(block)
    return md5,sha.hexdigest()


def main():
    key=sys.stdin.readline().strip()
    if not key:raise RuntimeError('Missing key')
    request=urllib.request.Request('https://api.roboflow.com/models/v1/external/weights?modelId=sam3video',
                                   headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(request,timeout=45) as response:
        metadata=json.load(response)['modelMetadata']
    print('METADATA_READY',metadata['modelId'],flush=True)
    package=next(p for p in metadata['modelPackages'] if p['packageId']=='45c999b9eaec5eeee18616bd494a5bc9')
    DEST.mkdir(parents=True,exist_ok=True)
    records=[]
    # Download configs and license first, then the large checkpoint.
    for item in sorted(package['packageFiles'],key=lambda x:x['fileHandle'].endswith('.safetensors')):
        name=item['fileHandle']
        assert Path(name).name==name and name not in ('','..')
        url=item['downloadUrl']
        assert urllib.parse.urlsplit(url).scheme=='https'
        parts=urllib.parse.urlsplit(url)
        print('FILE_SOURCE',name,'host',parts.hostname,'query_fields',list(urllib.parse.parse_qs(parts.query)),flush=True)
        path=DEST/name
        expected=item.get('md5Hash')
        if path.exists():
            md5,sha=file_hashes(path)
            if expected and expected in (md5.hexdigest(),base64.b64encode(md5.digest()).decode()):
                records.append(dict(file=name,size=path.stat().st_size,sha256=sha,md5_verified=True))
                print('VERIFIED_EXISTING',name,flush=True)
                continue
            raise RuntimeError('Existing file is not verified; preserving it')
        partial=path.with_suffix(path.suffix+'.partial')
        done=partial.stat().st_size if partial.exists() else 0
        headers={'Range':f'bytes={done}-'} if done else {}
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as source:
            if done and source.status!=206:
                raise RuntimeError('Server did not support resume; preserving partial')
            total=done+int(source.headers.get('Content-Length','0'))
            last=done
            print('DOWNLOADING',name,'total_bytes',total,flush=True)
            with partial.open('ab' if done else 'wb') as target:
                for block in iter(lambda:source.read(8*1024*1024),b''):
                    target.write(block);done+=len(block)
                    if done-last>=128*1024*1024:
                        print('PROGRESS_MB',round(done/2**20),'/',round(total/2**20),flush=True)
                        last=done
        if done!=total:raise RuntimeError('Incomplete download')
        md5,sha=file_hashes(partial)
        if expected and expected not in (md5.hexdigest(),base64.b64encode(md5.digest()).decode()):
            raise RuntimeError('Downloaded checksum mismatch')
        partial.replace(path)
        records.append(dict(file=name,size=done,sha256=sha,md5_verified=bool(expected)))
        print('FILE_READY',name,done,flush=True)
    manifest=dict(provider='Roboflow authenticated weights API',model_id=metadata['modelId'],
                  package_id=package['packageId'],package_manifest=package['packageManifest'],
                  downloaded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=records)
    (DEST/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('MODEL_DOWNLOAD_COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('DOWNLOAD_FAILED',type(error).__name__,getattr(error,'code',''),flush=True)
        if isinstance(error,urllib.error.HTTPError):
            raw_body=error.read(4096).decode('utf-8',errors='replace')
            print('HTTP_RESPONSE_TYPE',error.headers.get('Content-Type'),error.headers.get('Server'),flush=True)
            try:
                body=ET.fromstring(raw_body)
                print('SERVER_ERROR_CODE',body.findtext('Code'),flush=True)
                msg=body.findtext('Message','')
                if 'http' not in msg and 'token' not in msg.lower():
                    print('SERVER_ERROR_MESSAGE',msg[:250],flush=True)
            except Exception:
                safe=re.sub(r'https?://\S+','[URL]',re.sub(r'<[^>]*>',' ',raw_body))
                print('SERVER_ERROR_TEXT',' '.join(safe.split())[:400],flush=True)
        sys.exit(1)
