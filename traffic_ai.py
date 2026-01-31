from ultralytics import YOLO
import cv2
from pathlib import Path
import os
import random
import serial
import time

# ----------------- CONFIG -----------------
MODEL_PATH = "yolo11l.pt"      # โมเดล YOLO
LANE_A_DIR = "a"               # โฟลเดอร์รูปเลน A
LANE_B_DIR = "b"               # โฟลเดอร์รูปเลน B

OUTPUT_A_DIR = "output_a"
OUTPUT_B_DIR = "output_b"

CAR_CLASS_IDS = [2, 3, 5, 7]   # car, motorcycle, bus, truck

CONF_THRESHOLD = 0.25
SCALE_FACTOR = 1.8
IMG_SIZE = 1280

SERIAL_PORT = "COM3"           # <-- แก้ให้ตรงกับเครื่องนาย
BAUDRATE = 115200
SEND_INTERVAL_S = 3            # ส่งคำสั่งไป ESP32 ทุกกี่วินาที (กันสแปมเกิน)
# ------------------------------------------


def open_serial():
    print(f"[INFO] เปิด Serial ที่ {SERIAL_PORT} {BAUDRATE}")
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    time.sleep(2)  # รอให้บอร์ดรีเซ็ต
    return ser


def load_model():
    print(f"[INFO] กำลังโหลดโมเดล: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


def count_cars_in_image(model, image_path, lane_name="A"):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] อ่านรูปไม่ได้: {image_path}")
        return None, 0

    # ขยายภาพ
    if SCALE_FACTOR != 1.0:
        img = cv2.resize(img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)

    results = model.predict(
        source=img,
        classes=CAR_CLASS_IDS,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
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


def send_request(ser, lane):
    """
    lane = 'A' หรือ 'B'
    Python จะไม่สั่งไฟโดยตรง แค่บอก ESP32 ว่าอยากให้ lane ไหนได้สิทธิ์
    """
    if lane not in ["A", "B"]:
        return

    cmd = f"REQUEST_{lane}\n"
    ser.write(cmd.encode("utf-8"))
    print(f"[Serial] ส่งคำสั่งไป ESP32: {cmd.strip()}")


def main():
    os.makedirs(OUTPUT_A_DIR, exist_ok=True)
    os.makedirs(OUTPUT_B_DIR, exist_ok=True)

    ser = open_serial()
    model = load_model()

    files_a = get_image_files(LANE_A_DIR)
    files_b = get_image_files(LANE_B_DIR)

    if not files_a or not files_b:
        print("[ERROR] ไม่พบรูปในโฟลเดอร์ a หรือ b")
        return

    last_send_time = 0

    while True:
        # สุ่มรูปจากเลน A และ B
        img_a_path = random.choice(files_a)
        img_b_path = random.choice(files_b)

        print(f"\nสุ่มรูปเลน A: {img_a_path.name}")
        print(f"สุ่มรูปเลน B: {img_b_path.name}")

        annotated_a, count_a = count_cars_in_image(model, img_a_path, "A")
        annotated_b, count_b = count_cars_in_image(model, img_b_path, "B")

        # ตัดสินใจเลนที่ควรได้สิทธิ์
        if count_a > count_b:
            preferred = "A"
        elif count_b > count_a:
            preferred = "B"
        else:
            preferred = None  # รถเท่ากัน ไม่ขออะไรเพิ่ม

        print(f"จำนวนรถเลน A: {count_a}")
        print(f"จำนวนรถเลน B: {count_b}")
        print(f"เลนที่รถมากกว่า (จาก AI): {preferred if preferred else 'เท่ากัน'}")

        # ส่งคำขอไป ESP32 เป็นระยะ ๆ
        now = time.time()
        if preferred is not None and now - last_send_time >= SEND_INTERVAL_S:
            send_request(ser, preferred)
            last_send_time = now

        # เซฟรูปไว้ดู
        cv2.imwrite(f"{OUTPUT_A_DIR}/result_{img_a_path.name}", annotated_a)
        cv2.imwrite(f"{OUTPUT_B_DIR}/result_{img_b_path.name}", annotated_b)

        # หน่วงนิดนึงกัน loop ไวไป
        time.sleep(10)


if __name__ == "__main__":
    main()
