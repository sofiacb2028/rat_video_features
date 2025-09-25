import cv2

cap = cv2.VideoCapture("top_left.mp4")  # Change to another file if needed

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Done reading video.")
        break

    cv2.imshow("Video Frame", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Video playback stopped.")
