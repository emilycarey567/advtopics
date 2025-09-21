
# from vehicle import Driver
# from controller import Keyboard

# driver = Driver()
# timestep = int(driver.getBasicTimeStep())


# cam = driver.getDevice("camera")
# cam.enable(timestep)

# kb = Keyboard()
# kb.enable(timestep)


# STEER_STEP = 0.02
# STEER_MAX  = 0.6
# SPEED_STEP = 0.025
# BRAKE_STEP = 0.1

# target_speed = 0.0
# steer = 0.0
# brake = 0.0

# print("✅ manual_cam running — open Tools → Camera Window and select 'camera'")
# width, height = cam.getWidth(), cam.getHeight()
# print(f"camera enabled: {width}x{height}")

# while driver.step() != -1:

    # pressed = set()
    # key = kb.getKey()
    # while key != -1:
        # pressed.add(key)
        # key = kb.getKey()

    # if Keyboard.LEFT in pressed:
        # steer -= STEER_STEP
    # elif Keyboard.RIGHT in pressed:
        # steer += STEER_STEP
    # else:
        # steer *= 0.9  # self-center

    # steer = max(-STEER_MAX, min(STEER_MAX, steer))
    # driver.setSteeringAngle(steer)


    # if Keyboard.UP in pressed:
        # target_speed += SPEED_STEP
        # brake = max(0.0, brake - BRAKE_STEP)
    # elif Keyboard.DOWN in pressed:
        # if driver.getCurrentSpeed() > 0.5:
            # brake = min(1.0, brake + BRAKE_STEP)
        # else:
            # target_speed = max(0.0, target_speed - SPEED_STEP)
    # else:
        # target_speed *= 0.995
        # brake *= 0.9

    # driver.setBrakeIntensity(brake)
    # driver.setCruisingSpeed(max(0.0, target_speed))

# manual_cam_logger.py — manual driving + camera + 10 Hz logging for steering-angle regression
# manual_cam_logger.py — manual driving + camera + 10 Hz logging for steering regression
# manual_cam_logger.py — manual driving + camera + 10 Hz logging for steering regression
from vehicle import Driver
from controller import Keyboard
import os, csv, time

# ========= CONFIG =========
RUN_NAME    = time.strftime("clear_%Y%m%d_%H%M%S")
DATA_ROOT   = "dataset"
DATA_DIR    = os.path.join(DATA_ROOT, RUN_NAME)
IMG_DIR     = os.path.join(DATA_DIR, "images")
CSV_PATH    = os.path.join(DATA_DIR, "labels.csv")
TARGET_HZ   = 10
SAVE_QUALITY = 100  # PNG: often ignored; use .jpg if you want quality to matter

# Control params
STEER_STEP = 0.02
STEER_MAX  = 0.6
SPEED_STEP = 30
BRAKE_STEP = 0.1
MAX_SPEED  = 30

# ========= SETUP =========
driver = Driver()
timestep_ms = int(driver.getBasicTimeStep())

cam = driver.getDevice("camera")
cam.enable(timestep_ms)

kb = Keyboard()
kb.enable(timestep_ms)

QUIT_KEYS = {27, ord('Q'), ord('q')}  # Esc/Q/q

# State
target_speed = 0.0
steer = 0.0
brake = 0.0

# Dirs & CSV
os.makedirs(IMG_DIR, exist_ok=True)
csv_file = open(CSV_PATH, "w", newline="")
csvw = csv.writer(csv_file)
csvw.writerow(["image", "steer_rad", "steer_norm"])

print(f"✅ manual_cam_logger running — logging to: {DATA_DIR}")
width, height = cam.getWidth(), cam.getHeight()
print(f"camera enabled: {width}x{height} @ ~{1000.0/timestep_ms:.1f} Hz sim step; capturing at {TARGET_HZ} Hz")

# 10 Hz scheduler
steps_per_sec   = 1000.0 / timestep_ms
steps_per_frame = max(1, int(round(steps_per_sec / TARGET_HZ)))
step_counter    = 0
frame_idx       = 0

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

try:
    while driver.step() != -1:
        # Keys pressed this step
        pressed = set()
        k = kb.getKey()
        while k != -1:
            pressed.add(k)
            k = kb.getKey()

        # Quit
        if any(k in QUIT_KEYS for k in pressed):
            print("🛑 Quit key pressed — stopping logging.")
            break

        # Steering
        if Keyboard.LEFT in pressed:
            steer -= STEER_STEP
        elif Keyboard.RIGHT in pressed:
            steer += STEER_STEP
        else:
            steer *= 0.9  # self-center

        steer = clamp(steer, -STEER_MAX, STEER_MAX)
        driver.setSteeringAngle(steer)

        # Throttle / brake (keep for driving)
        if Keyboard.UP in pressed:
            target_speed = clamp(target_speed + SPEED_STEP, 0.0, MAX_SPEED)
            brake = max(0.0, brake - BRAKE_STEP)
        elif Keyboard.DOWN in pressed:
            if driver.getCurrentSpeed() > 0.5:
                brake = clamp(brake + BRAKE_STEP, 0.0, 1.0)
            else:
                target_speed = max(0.0, target_speed - SPEED_STEP)
        else:
            target_speed *= 0.995
            brake *= 0.9

        driver.setBrakeIntensity(brake)
        driver.setCruisingSpeed(clamp(target_speed, 0.0, MAX_SPEED))

        # ===== 10 Hz capture =====
        step_counter += 1
        if step_counter >= steps_per_frame:
            step_counter = 0

            img_name = f"img_{frame_idx:06d}.png"  # switch to .jpg if you want SAVE_QUALITY to affect size
            img_path = os.path.join(IMG_DIR, img_name)
            cam.saveImage(img_path, SAVE_QUALITY)

            steer_norm = steer / STEER_MAX  # [-1, 1]

            csvw.writerow([
                os.path.join("images", img_name).replace("\\", "/"),
                f"{steer:.6f}",
                f"{steer_norm:.6f}",
            ])

            if frame_idx % 50 == 0:
                csv_file.flush()

            frame_idx += 1

finally:
    csv_file.flush()
    csv_file.close()
    print(f"✅ Logging stopped. Saved {frame_idx} frames to {DATA_DIR}")

