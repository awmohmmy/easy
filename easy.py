import time
from pathlib import Path
import cv2
import serial
from ultralytics import YOLO

# ===================== CONFIG =====================
PORT = "COM3"          # <-- แก้ให้ตรง ESP32
BAUD = 115200

MODEL_PATH = "yolo11l.pt"
LANE_A_DIR = "a"
LANE_B_DIR = "b"

VEHICLE_CLASS_IDS = [2]   # car อย่างเดียว
CONF = 0.15
IMGSZ = 1280

STEP_DELAY_SEC = 10
SWITCH_MARGIN = 1
SHOW_WINDOWS = True
SEND_ONLY_ON_CHANGE = True
# ==================================================

def list_images(folder: str):
    p = Path(folder)
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for e in exts:
        files += list(p.glob(e))
    return sorted(files)

def open_serial():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser

def send_cmd(ser: serial.Serial, cmd: str):
    cmd = cmd.strip().upper()
    ser.write((cmd + "\n").encode("utf-8"))
    time.sleep(0.03)
    while ser.in_waiting:
        print("[ESP32]", ser.readline().decode(errors="ignore").strip())

def count_and_plot(model: YOLO, img_path: Path, lane_name: str):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] อ่านรูปไม่ได้: {img_path}")
        return 0, None

    results = model.predict(
        source=img,
        classes=VEHICLE_CLASS_IDS,
        conf=CONF,
        imgsz=IMGSZ,
        verbose=False
    )

    count = len(results[0].boxes) if results else 0
    vis = results[0].plot() if results else img

    cv2.putText(
        vis,
        f"Lane {lane_name}: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2
    )
    return count, vis

def decide(count_a: int, count_b: int):
    if count_a >= count_b + SWITCH_MARGIN:
        return "A"
    if count_b >= count_a + SWITCH_MARGIN:
        return "B"
    return None

def main():
    files_a = list_images(LANE_A_DIR)
    files_b = list_images(LANE_B_DIR)

    if not files_a or not files_b:
        print("[ERROR] ไม่พบรูปในโฟลเดอร์ a หรือ b")
        return

    print("[INFO] Loading YOLO model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    print(f"[INFO] Opening Serial: {PORT} @ {BAUD}")
    ser = open_serial()

    idx = 0
    last_cmd = None

    try:
        while True:
            img_a = files_a[idx % len(files_a)]
            img_b = files_b[idx % len(files_b)]
            idx += 1

            print(f"\n[FRAME] A={img_a.name} | B={img_b.name}")

            count_a, vis_a = count_and_plot(model, img_a, "A")
            count_b, vis_b = count_and_plot(model, img_b, "B")

            pref = decide(count_a, count_b)

            if pref == "A":
                cmd = "A_GREEN"
            elif pref == "B":
                cmd = "B_GREEN"
            else:
                cmd = "OFF"

            print(f"[DECIDE] A={count_a}, B={count_b} -> {cmd}")

            if (not SEND_ONLY_ON_CHANGE) or (cmd != last_cmd):
                send_cmd(ser, cmd)
                last_cmd = cmd

            if SHOW_WINDOWS:
                if vis_a is not None:
                    cv2.imshow("Lane A", vis_a)
                if vis_b is not None:
                    cv2.imshow("Lane B", vis_b)
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
        print("[INFO] Exit.")

if __name__ == "__main__":
    main()
