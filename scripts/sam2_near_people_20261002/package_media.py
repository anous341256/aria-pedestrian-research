"""Assemble the complete 53-second review and decode every delivered video."""
from render import *
import subprocess

def main():
    p=OUT/'presentation/reviewed';ffmpeg=Path('E:/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe')
    for kind in ['near_masks','before_after','motion_prompts']:
        paths=[p/f'{i:02d}_{kind}.mp4' for i in range(1,11)];assert all(x.exists() for x in paths)
        listing=p/f'concat_{kind}.txt';listing.write_text(''.join(f"file '{x.as_posix()}'\n" for x in paths),encoding='utf8')
        target=p/f'all_10_{kind}.mp4'
        if not target.exists():subprocess.run([str(ffmpeg),'-v','error','-n','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(target)],check=True)
    results=[]
    for file in sorted(p.glob('*.mp4')):
        cap=cv2.VideoCapture(str(file));reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));decoded=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape[:2]==(height,width);decoded+=1
        cap.release();expected=530 if file.name.startswith('all_10') else read_json(p/f'{file.name[:2]}_media.json')['frames'];assert expected==decoded==reported,(file,expected,decoded,reported);assert abs(fps-10)<.01
        results.append(dict(file=file.name,frames=decoded,seconds=decoded/fps,width=width,height=height,decode_pass=True))
    write_json(p/'media_verification.json',dict(passed=True,videos=results));print('Verified',len(results),'videos; each complete anthology is 530 frames / 53 seconds.')
if __name__=='__main__':main()
