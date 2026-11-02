"""Fixed recording-spread mid-frame visual inspection sheets."""
import cv2
import numpy as np
from selection import OUT

root = OUT/'presentation'
dest = root/'inspection'
dest.mkdir(exist_ok=True)
ids = list(range(1, 22))
for start in range(0, len(ids), 4):
    sheet = np.full((1056, 1280, 3), 255, np.uint8)
    for j, i in enumerate(ids[start:start+4]):
        cap = cv2.VideoCapture(str(root/'raw'/f'{i:03d}.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
        ok, image = cap.read()
        cap.release()
        assert ok
        sheet[(j//2)*528:(j//2+1)*528, (j%2)*640:(j%2+1)*640] = cv2.resize(image, (640, 528))
    cv2.imwrite(str(dest/f'fixed21_sheet_{start//4+1:02d}.jpg'), sheet)
print('Six fixed review sheets saved.')
