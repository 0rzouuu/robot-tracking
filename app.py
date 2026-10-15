from flask import Flask, render_template, Response, request, jsonify
from robot_controller import RobotController
import threading
import time
import os
import cv2
import numpy as np

AUTH_PASSWORD = ""

app = Flask(__name__)
robot = RobotController()

BLANK_JPEG = cv2.imencode('.jpg', np.zeros((480, 640, 3), np.uint8))[1].tobytes()


@app.before_request
def check_auth():
    if not AUTH_PASSWORD:
        return
    auth = request.authorization
    if not auth or auth.password != AUTH_PASSWORD:
        return Response('Login required', 401,
                       {'WWW-Authenticate': 'Basic realm="Robot Controller"'})


# Track active commands
active_commands = {}
command_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = robot.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + BLANK_JPEG + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mask_feed')
def mask_feed():
    def generate():
        while True:
            mask = robot.get_mask_frame()
            if mask is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + mask + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + BLANK_JPEG + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the Raspberry Pi"""
    print("Shutdown requested from web interface")
    
    def shutdown_pi():
        time.sleep(1)
        try:
            robot.shutdown_pi()
        except Exception as e:
            print(f"Shutdown error: {e}")
            os.system("sudo shutdown -h now")
    
    threading.Thread(target=shutdown_pi).start()
    return jsonify({'status': 'shutting_down'})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Switch between manual and automatic mode"""
    try:
        mode = request.json.get('mode')
        if mode in ['manual', 'automatic']:
            robot.set_mode(mode)
            with command_lock:
                active_commands.clear()
            return jsonify({'status': 'ok', 'mode': mode})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    return jsonify({'status': 'error', 'message': 'Invalid mode'}), 400

@app.route('/update_follow_params', methods=['POST'])
def update_follow_params():
    """Update automatic following parameters"""
    try:
        data = request.json
        robot.update_follow_params(
            data.get('targetArea', 5000),
            data.get('areaTolerance', 1000),
            data.get('centerTolerance', 30)
        )
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/control', methods=['POST'])
def control():
    """Endpoint for simple commands"""
    try:
        action = request.json.get('action')
        duration = request.json.get('duration', 0.1)
        
        if robot.mode == "manual":
            if action == 'forward':
                robot.forward(duration)
            elif action == 'reverse':
                robot.reverse(duration)
            elif action == 'left':
                robot.left_turn(duration)
            elif action == 'right':
                robot.right_turn(duration)
            elif action == 'stop':
                robot.stop()
            elif action == 'cleanup':
                robot.cleanup()
            return jsonify({'status': 'ok', 'action': action, 'mode': robot.mode})
        else:
            return jsonify({'status': 'error', 'message': 'Manual control disabled'}), 403
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/continuous_control', methods=['POST'])
def continuous_control():
    """Endpoint for continuous control"""
    try:
        action = request.json.get('action')
        state = request.json.get('state', 'pressed')
        
        if robot.mode == "manual":
            with command_lock:
                if state == 'pressed':
                    active_commands[action] = time.time()
                    robot.continuous_command(action)
                else:
                    if action in active_commands:
                        del active_commands[action]
                    
                    if active_commands:
                        latest_command = max(active_commands.items(), key=lambda x: x[1])[0]
                        robot.continuous_command(latest_command)
                    else:
                        robot.continuous_command('stop')
            
            return jsonify({'status': 'ok', 'action': action, 'state': state})
        else:
            return jsonify({'status': 'error', 'message': 'Manual control disabled'}), 403
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/get_robot_status')
def get_robot_status():
    """Get robot status"""
    try:
        return jsonify({
            'direction': robot.current_direction,
            'moving': robot.moving,
            'camera_running': robot.camera_running,
            'mode': robot.mode,
            'speed': robot.base_speed,
            'target_area': robot.target_area,
            'area_tolerance': robot.area_tolerance,
            'center_tolerance': robot.center_tolerance,
            'gpio_ready': robot.gpio_ready
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/update_hsv', methods=['POST'])
def update_hsv():
    try:
        params = request.json
        robot.update_hsv_params(params)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/get_hsv')
def get_hsv():
    try:
        return jsonify({
            'hueLower': robot.hue_lower,
            'hueUpper': robot.hue_upper,
            'hue2Lower': robot.hue2_lower,
            'hue2Upper': robot.hue2_upper,
            'satLow': robot.sat_low,
            'satHigh': robot.sat_high,
            'valLow': robot.val_low,
            'valHigh': robot.val_high
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get_trajectory')
def get_trajectory():
    return jsonify({'points': robot.get_trajectory()})


@app.route('/clear_trajectory', methods=['POST'])
def clear_trajectory():
    robot.clear_trajectory()
    return jsonify({'status': 'ok'})


@app.route('/get_pid')
def get_pid():
    return jsonify(robot.get_pid_params())


@app.route('/update_pid', methods=['POST'])
def update_pid():
    try:
        robot.update_pid_params(request.json)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/set_speed', methods=['POST'])
def set_speed():
    """Set motor speed (0-100)"""
    try:
        speed = request.json.get('speed', 70)
        robot.set_speed(speed)
        return jsonify({'status': 'ok', 'speed': robot.base_speed})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/emergency_stop', methods=['GET', 'POST'])
def emergency_stop():
    """Emergency stop"""
    try:
        with command_lock:
            active_commands.clear()
        robot.stop()
        return jsonify({'status': 'emergency_stop'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    try:
        try:
            robot.start_camera()
            print("Camera started")
        except Exception as e:
            print(f"Camera start error: {e}")

        print("=" * 50)
        print("ROBOT CONTROLLER SERVER")
        print("=" * 50)
        print("Starting server...")
        print("Local access: http://127.0.0.1:5000")
        print("Network access: http://[YOUR_PI_IP]:5000")
        print("-" * 50)
        print("Controls:")
        print("  - Manual mode: Keyboard arrows or ZQSD/WASD")
        print("  - Auto mode: Robot follows object")
        print("  - Space: Emergency stop")
        print("  - Shutdown button: Safely power off Pi")
        print("=" * 50)
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        robot.stop_camera()
        print("Goodbye!")
    except Exception as e:
        print(f"Server error: {e}")
        robot.stop_camera()