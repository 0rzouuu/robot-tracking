import cv2
import numpy as np
import time
import threading
import os
import json

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False

try:
    _orig_dir = os.getcwd()
    os.chdir('/tmp')
    import RPi.GPIO as gpio
    os.chdir(_orig_dir)
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
except Exception:
    HAS_GPIO = False
finally:
    try:
        os.chdir(_orig_dir)
    except Exception:
        pass

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')


class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limit=100):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.filtered_derivative = 0.0
        self.derivative_filter = 0.3

    def compute(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 0.03

        self.integral += error * dt
        self.integral = max(-self.output_limit, min(self.output_limit, self.integral))

        raw_derivative = (error - self.prev_error) / dt
        self.filtered_derivative += self.derivative_filter * (raw_derivative - self.filtered_derivative)

        output = self.kp * error + self.ki * self.integral + self.kd * self.filtered_derivative
        output = max(-self.output_limit, min(self.output_limit, output))

        self.prev_error = error
        self.last_time = now
        return output

    def reset(self):
        self.prev_error = 0
        self.integral = 0
        self.filtered_derivative = 0.0
        self.last_time = time.time()


class RobotController:
    def __init__(self):
        self.picam2 = None
        self.camera_running = False
        self.current_frame = None
        self.current_mask = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.last_command_time = 0
        self.command_cooldown = 0.05

        self.base_speed = 70

        # Following parameters
        self.target_area = 5000
        self.area_tolerance = 1000
        self.center_tolerance = 30

        # PID controllers for auto-follow
        self.pan_pid = PIDController(kp=0.2, ki=0.01, kd=0.05, output_limit=100)
        self.distance_pid = PIDController(kp=0.005, ki=0.001, kd=0.002, output_limit=100)

        # HSV parameters
        self.hue_lower = 20
        self.hue_upper = 42
        self.hue2_lower = 0
        self.hue2_upper = 0
        self.sat_low = 80
        self.sat_high = 255
        self.val_low = 80
        self.val_high = 255

        # Morphological kernel for mask noise reduction
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.auto_white_balance = True
        self.min_contour_area = 2000

        # Robot state
        self.current_direction = "stop"
        self.moving = False
        self.mode = "manual"
        self.no_object_counter = 0
        self.max_no_object_frames = 15
        self.search_speed = 25

        # GPIO status
        self.gpio_ready = False

        # Smooth motor control for auto-follow
        self.smooth_left = 0.0
        self.smooth_right = 0.0
        self.smoothing_factor = 0.3
        self.min_motor_speed = 15

        # Trajectory tracking
        self.trajectory = []
        self.max_trajectory_points = 150

        self.load_config()

        self.pwm_pins = {}
        self.setup_gpio()
        print("Robot controller initialized")

    def load_config(self):
        try:
            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
            self.hue_lower = cfg.get('hue_lower', self.hue_lower)
            self.hue_upper = cfg.get('hue_upper', self.hue_upper)
            self.hue2_lower = cfg.get('hue2_lower', self.hue2_lower)
            self.hue2_upper = cfg.get('hue2_upper', self.hue2_upper)
            self.sat_low = cfg.get('sat_low', self.sat_low)
            self.sat_high = cfg.get('sat_high', self.sat_high)
            self.val_low = cfg.get('val_low', self.val_low)
            self.val_high = cfg.get('val_high', self.val_high)
            self.base_speed = cfg.get('base_speed', self.base_speed)
            self.target_area = cfg.get('target_area', self.target_area)
            self.area_tolerance = cfg.get('area_tolerance', self.area_tolerance)
            self.center_tolerance = cfg.get('center_tolerance', self.center_tolerance)
            if 'pan_kp' in cfg:
                self.pan_pid.kp = cfg['pan_kp']
                self.pan_pid.ki = cfg.get('pan_ki', self.pan_pid.ki)
                self.pan_pid.kd = cfg.get('pan_kd', self.pan_pid.kd)
            if 'dist_kp' in cfg:
                self.distance_pid.kp = cfg['dist_kp']
                self.distance_pid.ki = cfg.get('dist_ki', self.distance_pid.ki)
                self.distance_pid.kd = cfg.get('dist_kd', self.distance_pid.kd)
            print(f"Config loaded from {CONFIG_PATH}")
        except (FileNotFoundError, json.JSONDecodeError):
            print("No config file found, using defaults")

    def save_config(self):
        cfg = {
            'hue_lower': self.hue_lower,
            'hue_upper': self.hue_upper,
            'hue2_lower': self.hue2_lower,
            'hue2_upper': self.hue2_upper,
            'sat_low': self.sat_low,
            'sat_high': self.sat_high,
            'val_low': self.val_low,
            'val_high': self.val_high,
            'base_speed': self.base_speed,
            'target_area': self.target_area,
            'area_tolerance': self.area_tolerance,
            'center_tolerance': self.center_tolerance,
            'pan_kp': self.pan_pid.kp,
            'pan_ki': self.pan_pid.ki,
            'pan_kd': self.pan_pid.kd,
            'dist_kp': self.distance_pid.kp,
            'dist_ki': self.distance_pid.ki,
            'dist_kd': self.distance_pid.kd,
        }
        try:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            print(f"Save config error: {e}")

    def setup_gpio(self):
        if not HAS_GPIO:
            print("GPIO library not available (not on Raspberry Pi)")
            print("Running in SIMULATION mode - motors disabled")
            return
        try:
            gpio.setmode(gpio.BCM)
            gpio.setup(20, gpio.OUT)
            gpio.setup(21, gpio.OUT)
            gpio.setup(23, gpio.OUT)
            gpio.setup(24, gpio.OUT)

            self.pwm_pins = {
                'left_fwd': gpio.PWM(21, 1000),
                'left_rev': gpio.PWM(20, 1000),
                'right_fwd': gpio.PWM(23, 1000),
                'right_rev': gpio.PWM(24, 1000),
            }
            for pwm in self.pwm_pins.values():
                pwm.start(0)

            self.stop()
            self.gpio_ready = True
            print("GPIO setup complete (PWM enabled)")
        except Exception as e:
            print(f"GPIO SETUP FAILED: {e}")
            print("Motors will not work! Check wiring and permissions.")

    def set_motors(self, left_speed, right_speed):
        """Set individual motor speeds. Range: -100 (full reverse) to 100 (full forward)."""
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))

        if 0 < abs(left_speed) < self.min_motor_speed:
            left_speed = 0
        if 0 < abs(right_speed) < self.min_motor_speed:
            right_speed = 0

        if left_speed == 0 and right_speed == 0:
            self.current_direction = 'stop'
            self.moving = False
        elif left_speed > 0 and right_speed > 0:
            self.current_direction = 'forward'
            self.moving = True
        elif left_speed < 0 and right_speed < 0:
            self.current_direction = 'reverse'
            self.moving = True
        elif left_speed > right_speed:
            self.current_direction = 'right'
            self.moving = True
        else:
            self.current_direction = 'left'
            self.moving = True

        if not self.gpio_ready:
            return

        try:
            if left_speed >= 0:
                self.pwm_pins['left_fwd'].ChangeDutyCycle(left_speed)
                self.pwm_pins['left_rev'].ChangeDutyCycle(0)
            else:
                self.pwm_pins['left_fwd'].ChangeDutyCycle(0)
                self.pwm_pins['left_rev'].ChangeDutyCycle(-left_speed)

            if right_speed >= 0:
                self.pwm_pins['right_fwd'].ChangeDutyCycle(right_speed)
                self.pwm_pins['right_rev'].ChangeDutyCycle(0)
            else:
                self.pwm_pins['right_fwd'].ChangeDutyCycle(0)
                self.pwm_pins['right_rev'].ChangeDutyCycle(-right_speed)
        except Exception as e:
            print(f"Set motors error: {e}")

    def forward(self, sec=0):
        self.set_motors(self.base_speed, self.base_speed)

    def reverse(self, sec=0):
        self.set_motors(-self.base_speed, -self.base_speed)

    def left_turn(self, sec=0):
        self.set_motors(-self.base_speed, self.base_speed)

    def right_turn(self, sec=0):
        self.set_motors(self.base_speed, -self.base_speed)

    def stop(self):
        self.smooth_left = 0.0
        self.smooth_right = 0.0
        self.set_motors(0, 0)

    def set_speed(self, speed):
        self.base_speed = max(0, min(100, int(speed)))
        self.save_config()

    def cleanup(self):
        try:
            self.stop()
            if HAS_GPIO and self.pwm_pins:
                for pwm in self.pwm_pins.values():
                    pwm.stop()
                gpio.cleanup()
            print("GPIO cleaned up")
        except Exception as e:
            print(f"Cleanup error: {e}")

    def shutdown_pi(self):
        print("SHUTTING DOWN RASPBERRY PI...")
        try:
            self.stop_camera()
            self.cleanup()
            time.sleep(2)
            os.system("sudo shutdown -h now")
        except Exception as e:
            print(f"Shutdown error: {e}")
            os.system("sudo halt")

    def set_mode(self, mode):
        self.mode = mode
        self.no_object_counter = 0
        self.trajectory.clear()
        self.pan_pid.reset()
        self.distance_pid.reset()
        self.stop()
        print(f"Mode switched to: {mode}")

    def start_camera(self):
        if not self.camera_running:
            if HAS_PICAMERA:
                try:
                    self.picam2 = Picamera2()
                    config = self.picam2.create_preview_configuration(
                        main={"size": (640, 480), "format": 'RGB888'}
                    )
                    self.picam2.configure(config)
                    self.picam2.start()
                    self.camera_running = True
                    self.running = True
                    self._use_webcam = False
                    print("PiCamera started successfully")
                except Exception as e:
                    print(f"PiCamera error: {e}, trying webcam fallback...")
                    self._start_webcam()
            else:
                self._start_webcam()

            if self.camera_running:
                self.capture_thread = threading.Thread(target=self._capture_loop)
                self.capture_thread.daemon = True
                self.capture_thread.start()

    def _start_webcam(self):
        self._webcam = cv2.VideoCapture(0)
        if self._webcam.isOpened():
            self._webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera_running = True
            self.running = True
            self._use_webcam = True
            print("Webcam started (simulation mode)")
        else:
            print("No camera available - video feeds will be blank")

    def _capture_loop(self):
        while self.running:
            if self.camera_running:
                try:
                    if self._use_webcam:
                        ret, frame = self._webcam.read()
                        if not ret:
                            time.sleep(0.03)
                            continue
                    else:
                        frame = self.picam2.capture_array()
                        if frame is None:
                            time.sleep(0.03)
                            continue
                    processed_frame, mask = self.process_frame(frame)
                    with self.frame_lock:
                        self.current_frame = processed_frame
                        self.current_mask = mask
                except Exception as e:
                    print(f"Capture error: {e}")
            time.sleep(0.03)

    def _normalize_frame(self, frame):
        result = frame.copy()
        if self.auto_white_balance:
            b, g, r = cv2.split(result)
            avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
            avg_all = (avg_b + avg_g + avg_r) / 3
            if avg_b > 0:
                b = np.clip((b * (avg_all / avg_b)), 0, 255).astype(np.uint8)
            if avg_g > 0:
                g = np.clip((g * (avg_all / avg_g)), 0, 255).astype(np.uint8)
            if avg_r > 0:
                r = np.clip((r * (avg_all / avg_r)), 0, 255).astype(np.uint8)
            result = cv2.merge([b, g, r])
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return result

    def process_frame(self, frame):
        try:
            normalized = self._normalize_frame(frame)
            blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            l_b = np.array([self.hue_lower, self.sat_low, self.val_low])
            u_b = np.array([self.hue_upper, self.sat_high, self.val_high])
            l_b2 = np.array([self.hue2_lower, self.sat_low, self.val_low])
            u_b2 = np.array([self.hue2_upper, self.sat_high, self.val_high])

            mask1 = cv2.inRange(hsv, l_b, u_b)
            mask2 = cv2.inRange(hsv, l_b2, u_b2)
            mask = cv2.add(mask1, mask2)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)

            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 2] = mask

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            area = 0
            object_found = False
            center_x = 0
            center_y = 0

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area >= self.min_contour_area:
                    object_found = True
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

                    center_x = x + w // 2
                    center_y = y + h // 2
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

                    self.trajectory.append((center_x, center_y))
                    if len(self.trajectory) > self.max_trajectory_points:
                        self.trajectory.pop(0)

                    frame_center = 320
                    cv2.line(frame, (frame_center - self.center_tolerance, 0),
                             (frame_center - self.center_tolerance, 480), (0, 255, 255), 1)
                    cv2.line(frame, (frame_center + self.center_tolerance, 0),
                             (frame_center + self.center_tolerance, 480), (0, 255, 255), 1)

                    if area < self.target_area - self.area_tolerance:
                        area_status = "TOO FAR"
                        area_color = (255, 255, 0)
                    elif area > self.target_area + self.area_tolerance:
                        area_status = "TOO CLOSE"
                        area_color = (0, 0, 255)
                    else:
                        area_status = "GOOD"
                        area_color = (0, 255, 0)

                    cv2.putText(frame, f"Area: {int(area)} {area_status}", (10, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, area_color, 2)

                    if self.mode == "automatic":
                        self.automatic_follow(area, center_x)

            if self.mode == "automatic":
                if not object_found:
                    self.no_object_counter += 1
                    if self.no_object_counter >= self.max_no_object_frames:
                        cv2.putText(frame, "SEARCHING...", (10, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        self.set_motors(-self.search_speed, self.search_speed)
                        self.pan_pid.reset()
                        self.distance_pid.reset()
                else:
                    self.no_object_counter = 0

            if len(self.trajectory) > 1:
                for i in range(1, len(self.trajectory)):
                    alpha = i / len(self.trajectory)
                    color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                    cv2.line(frame, self.trajectory[i - 1], self.trajectory[i], color, 2)

            cv2.putText(frame, f"Dir: {self.current_direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mode: {self.mode.upper()}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            status_text = "OBJECT DETECTED" if object_found else "NO OBJECT"
            status_color = (0, 255, 0) if object_found else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            if object_found:
                cv2.putText(frame, f"Pos X: {center_x} | Area: {int(area)}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(frame, f"Target: {self.target_area}+/-{self.area_tolerance}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Center Tol: +/-{self.center_tolerance}px", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Speed: {self.base_speed}%", (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if self.mode == "automatic":
                cv2.putText(frame, "AUTO MODE", (500, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "MANUAL MODE", (480, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            return frame, mask_colored

        except Exception as e:
            print(f"Process frame error: {e}")
            return frame, np.zeros_like(frame)

    def automatic_follow(self, area, center_x):
        """PID-based object following with simultaneous turning and driving."""
        try:
            frame_center = 320

            pan_error = center_x - frame_center
            distance_error = self.target_area - area

            centered = abs(pan_error) <= self.center_tolerance
            good_distance = abs(distance_error) <= self.area_tolerance

            if centered and good_distance:
                left_target = 0.0
                right_target = 0.0
                self.smooth_left += self.smoothing_factor * (0 - self.smooth_left)
                self.smooth_right += self.smoothing_factor * (0 - self.smooth_right)
                if abs(self.smooth_left) < 2 and abs(self.smooth_right) < 2:
                    self.stop()
                    return
                self.set_motors(self.smooth_left, self.smooth_right)
                return

            turn_output = self.pan_pid.compute(pan_error)
            forward_output = self.distance_pid.compute(distance_error)

            if centered:
                turn_output *= 0.3

            left_target = (forward_output + turn_output) * self.base_speed / 100
            right_target = (forward_output - turn_output) * self.base_speed / 100

            left_target = max(-self.base_speed, min(self.base_speed, left_target))
            right_target = max(-self.base_speed, min(self.base_speed, right_target))

            self.smooth_left += self.smoothing_factor * (left_target - self.smooth_left)
            self.smooth_right += self.smoothing_factor * (right_target - self.smooth_right)

            self.set_motors(self.smooth_left, self.smooth_right)

        except Exception as e:
            print(f"Auto follow error: {e}")

    def update_follow_params(self, target_area, area_tolerance, center_tolerance):
        try:
            self.target_area = int(target_area)
            self.area_tolerance = int(area_tolerance)
            self.center_tolerance = int(center_tolerance)
            self.save_config()
            print(f"Follow params updated: Target={self.target_area}, Tol={self.area_tolerance}, Center={self.center_tolerance}")
        except Exception as e:
            print(f"Update params error: {e}")

    def update_pid_params(self, params):
        try:
            self.pan_pid.kp = float(params.get('panKp', self.pan_pid.kp))
            self.pan_pid.ki = float(params.get('panKi', self.pan_pid.ki))
            self.pan_pid.kd = float(params.get('panKd', self.pan_pid.kd))
            self.distance_pid.kp = float(params.get('distKp', self.distance_pid.kp))
            self.distance_pid.ki = float(params.get('distKi', self.distance_pid.ki))
            self.distance_pid.kd = float(params.get('distKd', self.distance_pid.kd))
            self.pan_pid.reset()
            self.distance_pid.reset()
            self.save_config()
        except Exception as e:
            print(f"Update PID error: {e}")

    def get_pid_params(self):
        return {
            'panKp': self.pan_pid.kp,
            'panKi': self.pan_pid.ki,
            'panKd': self.pan_pid.kd,
            'distKp': self.distance_pid.kp,
            'distKi': self.distance_pid.ki,
            'distKd': self.distance_pid.kd,
        }

    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                try:
                    ret, jpeg = cv2.imencode('.jpg', self.current_frame)
                    return jpeg.tobytes()
                except Exception:
                    return None
        return None

    def get_mask_frame(self):
        with self.frame_lock:
            if self.current_mask is not None:
                try:
                    ret, jpeg = cv2.imencode('.jpg', self.current_mask)
                    return jpeg.tobytes()
                except Exception:
                    return None
        return None

    def get_trajectory(self):
        return list(self.trajectory)

    def clear_trajectory(self):
        self.trajectory.clear()

    def stop_camera(self):
        self.running = False
        time.sleep(0.1)
        if self.camera_running:
            try:
                if getattr(self, '_use_webcam', False):
                    self._webcam.release()
                else:
                    self.picam2.stop()
                    self.picam2.close()
            except Exception:
                pass
            self.camera_running = False
            self.cleanup()
            print("Camera stopped")

    def update_hsv_params(self, params):
        try:
            self.hue_lower = int(params.get('hueLower', self.hue_lower))
            self.hue_upper = int(params.get('hueUpper', self.hue_upper))
            self.hue2_lower = int(params.get('hue2Lower', self.hue2_lower))
            self.hue2_upper = int(params.get('hue2Upper', self.hue2_upper))
            self.sat_low = int(params.get('satLow', self.sat_low))
            self.sat_high = int(params.get('satHigh', self.sat_high))
            self.val_low = int(params.get('valLow', self.val_low))
            self.val_high = int(params.get('valHigh', self.val_high))
            self.save_config()
            print(f"HSV updated: H={self.hue_lower}-{self.hue_upper}")
        except Exception as e:
            print(f"Update HSV error: {e}")

    def continuous_command(self, action):
        if self.mode == "manual":
            try:
                current_time = time.time()
                if current_time - self.last_command_time > self.command_cooldown:
                    if action == 'forward':
                        self.forward()
                    elif action == 'reverse':
                        self.reverse()
                    elif action == 'left':
                        self.left_turn()
                    elif action == 'right':
                        self.right_turn()
                    elif action == 'stop':
                        self.stop()
                    self.last_command_time = current_time
            except Exception as e:
                print(f"Continuous command error: {e}")
