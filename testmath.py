import time
import cv2
import serial
from ultralytics import YOLO

# ===================== CONFIG =====================
PORT = "COM3"
BAUD = 115200

MODEL_PATH = "yolo11l.pt"

VIDEO_PATH_A = "C:/Users/TUF F15/Downloads/videolanea.mp4"
VIDEO_PATH_B = "C:/Users/TUF F15/Downloads/videolaneb.mp4"

VEHICLE_CLASS_IDS = [2]   # car
CONF = 0.15
IMGSZ = 640

RESIZE_WIDTH = 640
DETECT_INTERVAL = 1.5

# ==== เวลาไฟเขียว ====
TIME_PER_CAR = 2.5
MIN_GREEN_TIME = 6.0
MAX_GREEN_TIME = 25.0

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


def calc_green_time(cars):
    t = cars * TIME_PER_CAR
    return max(MIN_GREEN_TIME, min(t, MAX_GREEN_TIME))


def count_and_plot(model, img, roi, lane, remain):
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
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2

            if x1 <= cx <= x2 and y1 <= cy <= y2:
                count += 1
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(vis, f"Lane {lane}: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(vis, f"Remain: {remain:.1f}s", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

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
    current_green_time = MIN_GREEN_TIME
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
                break

            now = time.time()
            elapsed = now - state_start_time
            remain = max(0.0, current_green_time - elapsed)

            # ===== DETECT =====
            if now - last_detect_time >= DETECT_INTERVAL:
                smallA = resize_keep_ratio(frameA, RESIZE_WIDTH)
                smallB = resize_keep_ratio(frameB, RESIZE_WIDTH)

                countA, visA = count_and_plot(model, smallA, ROI_A, "A", remain)
                countB, visB = count_and_plot(model, smallB, ROI_B, "B", remain)

                print(f"[DETECT] A={countA} B={countB}")
                last_detect_time = now

            # ===== STATE MACHINE =====
            if elapsed >= current_green_time:

                if countA > countB:
                    current_state = "GREEN_A"
                    current_green_time = calc_green_time(countA)
                elif countB > countA:
                    current_state = "RED_A"
                    current_green_time = calc_green_time(countB)
                else:
                    if current_state == "GREEN_A":
                        current_state = "RED_A"
                        current_green_time = calc_green_time(countB)
                    else:
                        current_state = "GREEN_A"
                        current_green_time = calc_green_time(countA)

                send_cmd(ser, current_state)
                print(f"[STATE] {current_state} | green={current_green_time:.1f}s")

                state_start_time = now

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
