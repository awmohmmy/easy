import time
import cv2
import serial
from ultralytics import YOLO

# ===================== CONFIG =====================
PORT = "COM3"
BAUD = 115200
MODEL_PATH = "yolo11l.pt"

VEHICLE_CLASS_IDS = [2]   # car
CONF = 0.25
IMGSZ = 640

DETECT_INTERVAL = 1.5

# ==== เวลาไฟเขียว ====
TIME_PER_CAR = 2.5
MIN_GREEN_TIME = 6.0
MAX_GREEN_TIME = 25.0

# ==== กรอบเลน (ปรับตามกล้องคุณ) ====
LANE_A = (100, 200, 500, 600)   # (x1,y1,x2,y2)
LANE_B = (600, 200, 1000, 600)
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


def calc_green_time(cars):
    t = cars * TIME_PER_CAR
    return max(MIN_GREEN_TIME, min(t, MAX_GREEN_TIME))


def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 < x < x2 and y1 < y < y2


def detect_and_count(model, frame):
    res = model.predict(
        frame,
        classes=VEHICLE_CLASS_IDS,
        conf=CONF,
        imgsz=IMGSZ,
        verbose=False
    )

    countA = 0
    countB = 0
    vis = frame.copy()

    # วาดกรอบเลน
    cv2.rectangle(vis, LANE_A[:2], LANE_A[2:], (255, 0, 0), 2)
    cv2.putText(vis, "Lane A", (LANE_A[0], LANE_A[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.rectangle(vis, LANE_B[:2], LANE_B[2:], (0, 0, 255), 2)
    cv2.putText(vis, "Lane B", (LANE_B[0], LANE_B[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if res and res[0].boxes:
        for b in res[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if point_in_box(cx, cy, LANE_A):
                countA += 1
                color = (255, 0, 0)

            elif point_in_box(cx, cy, LANE_B):
                countB += 1
                color = (0, 0, 255)

            else:
                color = (0, 255, 0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis, (cx, cy), 4, color, -1)

    return countA, countB, vis


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] เปิดกล้องไม่ได้")
        return

    ser = open_serial()

    current_state = "GREEN_A"
    current_green_time = MIN_GREEN_TIME
    state_start_time = time.time()
    last_detect_time = 0

    countA = countB = 0
    vis = None

    send_cmd(ser, current_state)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            elapsed = now - state_start_time
            remain = max(0.0, current_green_time - elapsed)

            # ===== DETECT =====
            if now - last_detect_time >= DETECT_INTERVAL:

                countA, countB, vis = detect_and_count(model, frame)

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
                    current_state = "RED_A" if current_state == "GREEN_A" else "GREEN_A"
                    current_green_time = MIN_GREEN_TIME

                send_cmd(ser, current_state)
                print(f"[STATE] {current_state} | {current_green_time:.1f}s")

                state_start_time = now

            if vis is not None:
                cv2.putText(vis, f"Remain: {remain:.1f}s", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Traffic AI", vis)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        send_cmd(ser, "OFF")
        cap.release()
        ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()