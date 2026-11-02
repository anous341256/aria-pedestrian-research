"""Export every RGB record in five fixed recordings, preserving source mapping."""
import argparse
import csv
import sys
from pathlib import Path
import cv2
import numpy as np
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.sensor_data import TimeDomain

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import encoder,finish_encoder,write_json,digest
OUT=ROOT/'outputs/sam2_fullvideos_20260927'
IDS=['162538','164557','151153','193709','201129']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    for sid in IDS:
        dest=OUT/sid;dest.mkdir(parents=True,exist_ok=True)
        if not args.preview and (dest/'manifest.json').exists():
            print('REUSE',sid,flush=True);continue
        src=ROOT/'data/Aria/raw_vrs'/f'unknown_20260803_{sid}.vrs'
        stat=src.stat();provider=data_provider.create_vrs_data_provider(str(src))
        stream=provider.get_stream_id_from_label('camera-rgb')
        calib=provider.get_device_calibration().get_camera_calib('camera-rgb')
        w,h=map(int,calib.get_image_size())
        target=calibration.get_linear_camera_calibration(w,h,float(calib.get_focal_lengths()[0]),'pinhole',calib.get_transform_device_camera())
        n=provider.get_num_data(stream);fps=float(provider.get_image_configuration(stream).nominal_rate_hz)
        times=provider.get_timestamps_ns(stream,TimeDomain.DEVICE_TIME)
        assert len(times)==n and fps==10
        selected=sorted(set([0,10,30]+list(range(100,n,100))+[n-1]))
        if args.preview:
            d=dest/'preview';d.mkdir(exist_ok=True);tiles=[];page=0
            for f in selected:
                image,_=provider.get_image_data_by_index(stream,f)
                rgb=calibration.distort_by_calibration(image.to_numpy_array(),target,calib)
                bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                small=cv2.resize(bgr,(1280,960))
                cv2.imwrite(str(d/f'{f:05d}.jpg'),small,[cv2.IMWRITE_JPEG_QUALITY,94])
                tile=cv2.resize(small,(640,480));cv2.rectangle(tile,(0,0),(640,32),(0,0,0),-1)
                cv2.putText(tile,f'{sid} f={f} t={f/10:.1f}s',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
                tiles.append(tile)
                if len(tiles)==8 or f==selected[-1]:
                    while len(tiles)<8:tiles.append(np.zeros_like(tile))
                    cv2.imwrite(str(d/f'sheet_{page:02d}.jpg'),np.vstack([np.hstack(tiles[j:j+2]) for j in range(0,8,2)]))
                    tiles=[];page+=1
            write_json(dest/'preview_manifest.json',{'recording':src.stem,'frames':n,'fps':fps,'preview_frames':selected,'prompt_coordinate_size':[1280,960]})
            print('PREVIEW COMPLETE',sid,n,flush=True)
        else:
            proc,log=encoder(dest/'rgb.mp4',w,h,fps,18)
            for f in range(n):
                image,_=provider.get_image_data_by_index(stream,f)
                rgb=calibration.distort_by_calibration(image.to_numpy_array(),target,calib)
                proc.stdin.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR).tobytes())
                if f%100==0:print('EXPORT',sid,f,'/',n,flush=True)
            finish_encoder(proc,log)
            with (dest/'frame_timestamps.csv').open('w',encoding='utf-8',newline='') as handle:
                wr=csv.writer(handle);wr.writerow(['frame','source_rgb_index','device_timestamp_ns','relative_recording_s'])
                wr.writerows((f,f,int(times[f]),(times[f]-times[0])/1e9) for f in range(n))
            assert (stat.st_size,stat.st_mtime_ns)==(src.stat().st_size,src.stat().st_mtime_ns)
            write_json(dest/'manifest.json',{'recording':src.stem,'source_vrs':str(src),'source_size':stat.st_size,'source_mtime_ns':stat.st_mtime_ns,'source_indices':[0,n-1],'frames':n,'fps':fps,'width':w,'height':h,'duration_s':n/fps,'source_timestamp_span_s':(times[-1]-times[0])/1e9,'image_domain':'factory-calibrated linear RGB, no temporal stabilization','source_calibration':calib.get_projection_params().tolist(),'target_calibration':target.get_projection_params().tolist(),'video_sha256':digest(dest/'rgb.mp4'),'script_sha256':digest(Path(__file__))})
            print('EXPORT COMPLETE',sid,n,flush=True)

if __name__=='__main__':main()
