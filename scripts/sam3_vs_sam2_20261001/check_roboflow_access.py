"""Check model download metadata with a credential received through stdin."""
import datetime
import json
from pathlib import Path
import re
import sys
import urllib.request
import urllib.parse
import urllib.error


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

OUT = Path(__file__).resolve().parents[2] / 'outputs/sam3_vs_sam2_20261001'
key = sys.stdin.readline().strip()
if not key:
    raise SystemExit('No credential received')
results = []
for model_id in ['sam3video', 'sam3trackervideo']:
    row = dict(model=model_id)
    try:
        url = 'https://api.roboflow.com/models/v1/external/weights?' + urllib.parse.urlencode({'modelId':model_id})
        request = urllib.request.Request(url,headers={'Authorization':'Bearer '+key})
        try:
            response = urllib.request.build_opener(NoRedirect()).open(request,timeout=45)
        except urllib.error.HTTPError as error:
            response = error
        row['http_status'] = response.code
        payload = json.load(response)
        metadata = payload.get('modelMetadata',{})
        if response.code == 200 and metadata:
            row['status'] = 'model_metadata_accessible'
            row['resolved_model_id'] = metadata.get('modelId')
            row['architecture'] = metadata.get('modelArchitecture')
            row['packages'] = [dict(package_id=p.get('packageId'),
                                   manifest_keys=list(p.get('packageManifest',{})),
                                   md5_hashes={f.get('fileHandle'):f.get('md5Hash') for f in p.get('packageFiles',[])},
                                   files=[f.get('fileHandle') for f in p.get('packageFiles',[])])
                               for p in metadata.get('modelPackages',[])]
            if model_id == 'sam3video':
                row['provider_download_links'] = {
                    f['fileHandle']:f['downloadUrl']
                    for p in metadata.get('modelPackages',[]) for f in p.get('packageFiles',[])
                    if urllib.parse.urlsplit(f['downloadUrl']).hostname=='core-models.roboflow.com'
                    and not urllib.parse.urlsplit(f['downloadUrl']).query and key not in f['downloadUrl']
                }
        else:
            row['status'] = 'access_not_confirmed'
            # Never expose request headers, key, signed download URLs or full API payloads.
            messages = [payload.get(k) for k in ('message','error','description')]
            raw = ' '.join(x for x in messages if isinstance(x,str)).replace(key,'[REDACTED]')
            row['message'] = re.sub(r'https?://\S+', '[URL]',raw)[:800]
    except Exception as error:
        row['status'] = 'request_failed'
        row['error_type'] = type(error).__name__
    results.append(row)
result=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            operation='read_model_download_metadata_only',results=results)
(OUT/'roboflow_access.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
