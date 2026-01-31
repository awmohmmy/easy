from ultralytics import YOLO
import cv2
from pathlib import Path
import os
import random

# ----------------- CONFIG -----------------
# ใช้โมเดลตัวใหญ่ yolo11l หรือ yolo11x เพื่อความแม่น
MODEL_PATH = "yolo11l.pt"   # <---- ใช้ตัว L ตามเดิม

LANE_A_DIR = "a"
LANE_B_DIR = "b"

OUTPUT_A_DIR = "output_a"
OUTPUT_B_DIR = "output_b"

CAR_CLASS_IDS = [2, 3, 5, 7]

# ปรับให้จับรถได้เยอะขึ้น
CONF_THRESHOLD = 0.25        # ลดลงจาก 0.30
SCALE_FACTOR = 1.8           # ขยายภาพให้ใหญ่ขึ้นกว่าเดิม
# ------------------------------------------


def load_model():
    print(f"กำลังโหลดโมเดล: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


def count_cars_in_image(model, image_path, lane_name="A"):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] อ่านรูปไม่ได้: {image_path}")
        return None, 0

    # ขยายภาพก่อน detect
    if SCALE_FACTOR != 1.0:
        img = cv2.resize(img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)

    results = model.predict(
        source=img,
        classes=CAR_CLASS_IDS,
        conf=CONF_THRESHOLD,
        imgsz=1280,      # เพิ่มขนาดภาพ input ให้โมเดล
        verbose=False
    )

    car_count = 0

    for r in results:
        for box in r.boxes:
            car_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, f"car {conf:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    cv2.putText(
        img, f"Lane {lane_name}: {car_count} cars", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
    )

    return img, car_count


def get_image_files(folder):
    folder = Path(folder)
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    return [f for f in folder.iterdir() if f.suffix.lower() in exts]


def main():
    os.makedirs(OUTPUT_A_DIR, exist_ok=True)
    os.makedirs(OUTPUT_B_DIR, exist_ok=True)

    model = load_model()

    files_a = get_image_files(LANE_A_DIR)
    files_b = get_image_files(LANE_B_DIR)

    if not files_a or not files_b:
        print("ไม่พบรูปในโฟลเดอร์ a หรือ b")
        return

    # สุ่มรูปจากเลน A และ B
    img_a_path = random.choice(files_a)
    img_b_path = random.choice(files_b)

    print(f"\nสุ่มรูปเลน A: {img_a_path.name}")
    print(f"สุ่มรูปเลน B: {img_b_path.name}")

    annotated_a, count_a = count_cars_in_image(model, img_a_path, "A")
    annotated_b, count_b = count_cars_in_image(model, img_b_path, "B")

    if count_a > count_b:
        more = "A"
    elif count_b > count_a:
        more = "B"
    else:
        more = "เท่ากัน"

    print(f"จำนวนรถเลน A: {count_a}")
    print(f"จำนวนรถเลน B: {count_b}")
    print(f"👉 เลนที่มีรถมากกว่า: {more}")

    cv2.imwrite(f"{OUTPUT_A_DIR}/result_{img_a_path.name}", annotated_a)
    cv2.imwrite(f"{OUTPUT_B_DIR}/result_{img_b_path.name}", annotated_b)

    print("\nเสร็จแล้ว! ดูผลลัพธ์ที่ output_a/ และ output_b/")


if __name__ == "__main__":
    main()
