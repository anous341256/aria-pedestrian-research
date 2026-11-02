"""Extract exact comparison frames for independently recorded visual findings."""
import argparse
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('recording');p.add_argument('name');p.add_argument('frames',nargs='+',type=int);a=p.parse_args()
    base=ROOT/'outputs/sam2_fullvideos_20260927'/a.recording/'sam2_raw_v1';dest=base/'evidence';dest.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(base/'comparison.mp4'));tiles=[]
    for f in a.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok,f
        cv2.imwrite(str(dest/f'{a.name}_{f:05d}.jpg'),img,[cv2.IMWRITE_JPEG_QUALITY,94])
        tiles.append(cv2.resize(img,(1280,480)))
    cap.release()
    for start in range(0,len(tiles),4):cv2.imwrite(str(dest/f'{a.name}_sheet_{start//4:02d}.jpg'),np.vstack(tiles[start:start+4]),[cv2.IMWRITE_JPEG_QUALITY,94])

if __name__=='__main__':main()
