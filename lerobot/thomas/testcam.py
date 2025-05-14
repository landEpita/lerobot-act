import cv2

print("Scanning available camera indices...")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera found at index {i}")
        else:
            print(f"⚠️ Opened index {i}, but can't read frame.")
        cap.release()
    else:
        print(f"❌ No camera at index {i}")