from render import *
cid='10';cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));sheet=np.zeros((960,1920,3),np.uint8)
for j in range(6):
    ok,im=cap.read();assert ok;im=cv2.resize(im,(640,480));cv2.putText(im,str(j),(20,40),0,1,(0,255,255),2)
    sheet[j//3*480:(j//3+1)*480,j%3*640:(j%3+1)*640]=im
cap.release();cv2.imwrite(str(OUT/cid/'first_six_review.jpg'),sheet)
