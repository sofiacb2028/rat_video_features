import cv2

video_path = "data/top_left.mp4"   # <-- change if your file is named differently
cap = cv2.VideoCapture(video_path)

print("Opened?", cap.isOpened())
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()
