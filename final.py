import time
from pathlib import Path
import cv2
import serial
from ultralytics import YOLO

# =================== CONFIG ===================
PORT = "COM3"          # <- แก้ให้ตรง ESP32 ของคุณ (เช่น COM5)
BAUD = 115200

MODEL_PATH = "yolo11l.pt"  # แนะนำสำหรับความแม่นขึ้นกว่า n
LANE_A_DIR = "a"
LANE_B_DIR = "b"

# COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
CAR_CLASS_IDS = [2]

CONF_THRESHOLD = 0.15
IMG_SIZE = 1280

STEP_DELAY_SEC = 10         # หน่วงรอบละ 10 วิ
SWITCH_MARGIN = 1           # ต้องต่างกันอย่างน้อยกี่คันถึงสลับไฟ (กันแกว่ง)
SHOW_WINDOWS = True         # โชว์ภาพตีกรอบไหม
# =============================================

def list_images(folder: str):
    p = Path(folder)
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for e in exts:
        files += list(p.glob(e))
    return sorted(files)

def send_cmd(ser: serial.Serial, cmd: str):
    cmd = cmd.strip().upper()
    ser.write((cmd + "\n").encode("utf-8"))
    # อ่านตอบกลับ (ถ้ามี)
    time.sleep(0.05)
    while ser.in_waiting:
        print("[ESP32]", ser.readline().decode(errors="ignore").strip())

def count_and_annotate(model: YOLO, img_path: Path, lane_name: str):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] อ่านรูปไม่ได้: {img_path}")
        return 0, None

    results = model.predict(
        source=img,
        classes=CAR_CLASS_IDS,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False
    )

    count = len(results[0].boxes) if results else 0
    annotated = results[0].plot() if results else img

    cv2.putText(
        annotated,
        f"Lane {lane_name}: {count} vehicles",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )
    return count, annotated

def decide(count_a: int, count_b: int):
    # กันแกว่ง: ต้องต่างกันมากพอถึงสลับ
    if count_a >= count_b + SWITCH_MARGIN:
        return "A"
    if count_b >= count_a + SWITCH_MARGIN:
        return "B"
    return None

def main():
    files_a = list_images(LANE_A_DIR)
    files_b = list_images(LANE_B_DIR)

    if not files_a or not files_b:
        print("[ERROR] ไม่พบรูปใน lane_a หรือ lane_b")
        return

    print("[INFO] โหลดโมเดล YOLO:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    print(f"[INFO] เปิด Serial: {PORT} @ {BAUD}")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # รอ ESP32 รีเซ็ต

    idx = 0
    last_sent = None

    try:
        while True:
            img_a = files_a[idx % len(files_a)]
            img_b = files_b[idx % len(files_b)]
            idx += 1

            print(f"\n[FRAME] A={img_a.name} | B={img_b.name}")

            count_a, vis_a = count_and_annotate(model, img_a, "A")
            count_b, vis_b = count_and_annotate(model, img_b, "B")

            pref = decide(count_a, count_b)

            if pref == "A":
                cmd = "A_GREEN"
            elif pref == "B":
                cmd = "B_GREEN"
            else:
                cmd = "OFF"   # รถใกล้เคียงกัน ไม่เปลี่ยน/หรือปิด (ปรับได้)

            print(f"[DECIDE] A={count_a}, B={count_b} -> {cmd}")

            # ส่งเฉพาะตอนคำสั่งเปลี่ยน (กันสแปม)
            if cmd != last_sent:
                send_cmd(ser, cmd)
                last_sent = cmd

            if SHOW_WINDOWS:
                if vis_a is not None:
                    cv2.imshow("Lane A", vis_a)
                if vis_b is not None:
                    cv2.imshow("Lane B", vis_b)

                # ESC เพื่อออก
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

            time.sleep(STEP_DELAY_SEC)

    finally:
        try:
            send_cmd(ser, "OFF")
        except Exception:
            pass
        ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
