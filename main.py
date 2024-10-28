import cv2
from ultralytics import YOLO

# Modelni yuklash
model = YOLO('yolov8n.pt')  # yoki kerakli modelni tanlang

# Videoni ochish
cap = cv2.VideoCapture(r'D:\\HUB\\test\\D48_20240923042700.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ob'ektlarni aniqlash
    results = model(frame)

    # Natijalarni videoga chizish
    annotated_frame = results[0].plot()  # Birinchi natijani chizish

    # # # Natijalarni ko'rsatish
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # 'q' tugmasi bosilganda chiqish
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
