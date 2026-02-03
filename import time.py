import time
import cv2
import serial
from ultralytics import YOLO

# ===================== CONFIG =====================
PORT = "COM3"
BAUD = 115200

MODEL_PATH = "yolo11l.pt"
VIDEO_PATH = "C://Users//TUF F15//Downloads//4K Road traffic video for object detection and tracking - free download now!.mp4"

VEHICLE_CLASS_IDS = [2]   # car อย่างเดียว
CONF = 0.15
IMGSZ = 640

STEP_DELAY_SEC = 2
SHOW_WINDOWS = True
SEND_ONLY_ON_CHANGE = True

SKIP_FRAMES = 15
RESIZE_WIDTH = 640

GREEN_LOCK_SECONDS = 20   # <<< รถ > 15 ให้เขียวค้าง 20 วินาที
# ==================================================

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

def resize_keep_ratio(img, width):
    h, w = img.shape[:2]
    scale = width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h))

def count_and_plot(model: YOLO, img):
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
        f"Cars: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2
    )
    return count, vis

def main():
    print("[INFO] Loading YOLO model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    print("[INFO] Opening video:", VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] ไม่สามารถเปิดไฟล์วิดีโอได้")
        return

    print(f"[INFO] Opening Serial: {PORT} @ {BAUD}")
    ser = open_serial()

    last_cmd = None
    last_time = 0
    frame_count = 0

    last_vis = None
    last_count = 0

    lock_until = 0   # เวลาสิ้นสุดการล็อกไฟเขียว

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] วิดีโอจบแล้ว")
                break

            frame_count += 1
            now = time.time()

            # ===== Skip Frame + Detect =====
            if frame_count % SKIP_FRAMES == 0:
                small = resize_keep_ratio(frame, RESIZE_WIDTH)
                count, vis = count_and_plot(model, small)
                last_count = count
                last_vis = vis
            else:
                count = last_count
                vis = last_vis if last_vis is not None else frame

            # ===== ตรรกะควบคุมไฟ =====
            if now < lock_until:
                # อยู่ในช่วงล็อก 20 วินาที → เขียวอย่างเดียว
                cmd = "GREEN"
            else:
                # ไม่ได้ล็อก → ตัดสินใจตามจำนวนรถ
                if count > 15:
                    cmd = "GREEN"
                    lock_until = now + GREEN_LOCK_SECONDS
                    print("[LOCK] Cars > 15 -> GREEN for 20 seconds")
                elif count >= 10:
                    cmd = "GREEN"
                else:
                    cmd = "RED"

            print(f"[FRAME {frame_count}] Cars={count} -> {cmd}")

            # ===== ส่งคำสั่งไป ESP32 =====
            if now - last_time >= STEP_DELAY_SEC:
                if (not SEND_ONLY_ON_CHANGE) or (cmd != last_cmd):
                    send_cmd(ser, cmd)
                    last_cmd = cmd
                last_time = now

            if SHOW_WINDOWS:
                cv2.imshow("Traffic", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    finally:
        try:
            send_cmd(ser, "OFF")
        except Exception:
            pass
        cap.release()
        ser.close()
        cv2.destroyAllWindows()
        print("[INFO] Exit.")

if __name__ == "__main__":
    main()
