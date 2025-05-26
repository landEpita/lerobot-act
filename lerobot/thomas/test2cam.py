import cv2

index = 1  # change ce chiffre entre 0 et 5 selon ce que tu trouves

cap = cv2.VideoCapture(index)

if not cap.isOpened():
    print(f"Camera {index} not available")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow(f"Camera {index}", frame)
    if cv2.waitKey(1) == 27:  # ESC pour quitter
        break

cap.release()
cv2.destroyAllWindows()
