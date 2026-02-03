import time
import cv2
import serial
from ultralytics import YOLO

# ===================== CONFIG =====================
PORT = "COM3"          # <<< เปลี่ยนเป็น COM5 ถ้าใช้ COM5
BAUD = 115200

MODEL_PATH = "yolo11l.pt"

VIDEO_PATH_A = "C:/Users/anaph/Downloads/videolanea.mp4"
VIDEO_PATH_B = "C:/Users/anaph/Downloads/videolaneb.mp4"

VEHICLE_CLASS_IDS = [2]   # car
CONF = 0.15
IMGSZ = 640

RESIZE_WIDTH = 640
DETECT_INTERVAL = 1.5     # ตรวจจับทุก 1.5 วิ
MIN_GREEN_TIME = 6.0      # เขียวขั้นต่ำต่อรอบ

ROI_A = (175, 225, 600, 550)
ROI_B = (175, 225, 600, 550)
# ==================================================

def open_serial():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    return ser

def send_cmd(ser, cmd):
    try:
        ser.write((cmd + "\n").encode())
        print("[SEND]", cmd)
    except:
        pass

def resize_keep_ratio(img, width):
    h, w = img.shape[:2]
    s = width / w
    return cv2.resize(img, (int(w * s), int(h * s)))

def count_and_plot(model, img, roi, lane, countdown):
    x1, y1, x2, y2 = roi
    res = model.predict(
        img,
        classes=VEHICLE_CLASS_IDS,
        conf=CONF,
        imgsz=IMGSZ,
        verbose=False
    )

    count = 0
    vis = img.copy()

    if res and res[0].boxes:
        for b in res[0].boxes:
            bx1, by1, bx2, by2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            cx, cy = (bx1 + bx2)//2, (by1 + by2)//2

            if x1 <= cx <= x2 and y1 <= cy <= y2:
                count += 1
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0,255,0), 2)
                cv2.putText(
                    vis,
                    f"{conf:.2f}",
                    (bx1, by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0,0,255), 3)
    cv2.putText(vis, f"Lane {lane}: {count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(vis, f"Change in: {countdown:.1f}s", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    return count, vis

def main():
    model = YOLO(MODEL_PATH)

    capA = cv2.VideoCapture(VIDEO_PATH_A)
    capB = cv2.VideoCapture(VIDEO_PATH_B)

    if not capA.isOpened() or not capB.isOpened():
        print("[ERROR] เปิดวิดีโอไม่ได้")
        return

    ser = open_serial()

    current_state = "GREEN_A"
    state_start_time = time.time()
    last_detect_time = 0

    countA = countB = 0
    visA = visB = None

    send_cmd(ser, current_state)
    print("[START]", current_state)

    try:
        while True:
            retA, frameA = capA.read()
            retB, frameB = capB.read()
            if not retA or not retB:
                print("[INFO] Video ended")
                break

            now = time.time()
            elapsed = now - state_start_time
            countdown = max(0.0, MIN_GREEN_TIME - elapsed)

            # ===== detect ตามเวลา =====
            if now - last_detect_time >= DETECT_INTERVAL:
                smallA = resize_keep_ratio(frameA, RESIZE_WIDTH)
                smallB = resize_keep_ratio(frameB, RESIZE_WIDTH)

                countA, visA = count_and_plot(model, smallA, ROI_A, "A", countdown)
                countB, visB = count_and_plot(model, smallB, ROI_B, "B", countdown)

                last_detect_time = now

            # ===== STATE MACHINE (จุดแก้หลัก) =====
            if elapsed >= MIN_GREEN_TIME:
                if countA > countB:
                    next_state = "GREEN_A"
                elif countB > countA:
                    next_state = "GREEN_B"
                else:
                    # รถเท่ากัน → สลับเลน
                    next_state = "GREEN_B" if current_state == "GREEN_A" else "GREEN_A"

                if next_state != current_state:
                    current_state = next_state
                    send_cmd(ser, current_state)
                    print(f"[STATE CHANGE] {current_state} | A={countA} B={countB}")
                else:
                    print(f"[STATE CONTINUE] {current_state} | A={countA} B={countB}")

                # 🔥 สำคัญที่สุด: reset เวลาเสมอ
                state_start_time = now

            # ===== show =====
            if visA is not None:
                cv2.imshow("Lane A", visA)
            if visB is not None:
                cv2.imshow("Lane B", visB)

            if cv2.waitKey(15) & 0xFF == 27:
                break

    finally:
        send_cmd(ser, "OFF")
        capA.release()
        capB.release()
        ser.close()
        cv2.destroyAllWindows()
        print("[EXIT]")

if __name__ == "__main__":
    main()
