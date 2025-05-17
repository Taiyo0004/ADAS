# app.py
from web.models.model_pipeline import ModelPipeline
from flask_socketio import SocketIO, emit
from flask import Flask, jsonify, render_template, request
import serial.tools.list_ports
import serial
import cv2
import time
import threading
import signal
import logging
import base64
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Configuration ---
FORCE_MOCK = "--mock" in sys.argv
USE_DEBUG = "--debug" in sys.argv
DISABLE_CAMERA = "--no-camera" in sys.argv  # Add option to disable camera

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Arduino Connection Manager
class ArduinoConnection:
    """Class representing a connection to a single Arduino"""

    def __init__(self, port_name, baud_rate=115200, timeout=1):
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial = None
        self.connected = False
        self.last_data = ""
        self.role = None  # e.g., 'steering', 'motor', 'sensors'

    def connect(self):
        """Establish connection to the Arduino"""
        try:
            self.serial = serial.Serial(
                self.port_name, self.baud_rate, timeout=self.timeout
            )
            time.sleep(1)  # Allow time for connection to stabilize
            self.serial.reset_input_buffer()
            self.connected = True
            logger.info(f"Connected to Arduino on {self.port_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Arduino on {self.port_name}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Close the connection to the Arduino"""
        if self.serial:
            try:
                self.serial.close()
                logger.info(f"Disconnected from Arduino on {self.port_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {self.port_name}: {e}")
            finally:
                self.connected = False
                self.serial = None

    def send_command(self, command):
        """Send a command to the Arduino"""
        if not self.serial or not self.connected:
            logger.warning(f"Cannot send command to {self.port_name}: not connected")
            return False

        try:
            # Add newline for proper line termination
            if not command.endswith("\n"):
                command += "\n"

            self.serial.write(command.encode())
            self.serial.flush()
            logger.debug(f"Sent to {self.port_name}: {command.strip()}")
            return True
        except Exception as e:
            logger.error(f"Error sending command to {self.port_name}: {e}")
            self.connected = False
            return False

    def read_data(self):
        """Read data from the Arduino"""
        if not self.serial or not self.connected:
            return None

        try:
            raw_data = self.serial.readline()
            if raw_data:
                try:
                    # Try to decode as UTF-8
                    data = raw_data.decode("utf-8").strip()
                    self.last_data = data
                    return data
                except UnicodeDecodeError:
                    # If can't decode as text, return hex representation
                    hex_data = " ".join([f"0x{b:02X}" for b in raw_data])
                    self.last_data = f"Binary data: {hex_data}"
                    return self.last_data
        except Exception as e:
            logger.error(f"Error reading from {self.port_name}: {e}")
            self.connected = False

        return None

    def __str__(self):
        role_str = f" ({self.role})" if self.role else ""
        return f"Arduino on {self.port_name}{role_str} - {'Connected' if self.connected else 'Disconnected'}"


class ArduinoManager:
    """Manager for multiple Arduino connections"""

    def __init__(self, socketio=None):
        self.connections = {}  # Dictionary of port_name -> ArduinoConnection
        self.stop_event = threading.Event()
        self.read_thread = None
        self.socketio = socketio

    def scan_ports(self):
        """Scan for available serial ports"""
        ports = list(serial.tools.list_ports.comports())
        return [p.device for p in ports]

    def connect_all(self):
        """Connect to all available Arduino boards"""
        available_ports = self.scan_ports()
        logger.info(f"Available ports: {available_ports}")

        for port in available_ports:
            if port not in self.connections:
                arduino = ArduinoConnection(port)
                if arduino.connect():
                    self.connections[port] = arduino

        return len(self.connections)

    def connect_to(self, port_name, role=None):
        """Connect to a specific port and assign a role"""
        if port_name in self.connections:
            # Already connected
            if role:
                self.connections[port_name].role = role
            return True

        arduino = ArduinoConnection(port_name)
        if arduino.connect():
            arduino.role = role
            self.connections[port_name] = arduino
            return True

        return False

    def disconnect_all(self):
        """Disconnect from all Arduino boards"""
        for arduino in self.connections.values():
            arduino.disconnect()
        self.connections = {}

    def disconnect_from(self, port_name):
        """Disconnect from a specific port"""
        if port_name in self.connections:
            self.connections[port_name].disconnect()
            del self.connections[port_name]
            return True
        return False

    def send_command_all(self, command):
        """Send a command to all connected Arduino boards"""
        for arduino in self.connections.values():
            arduino.send_command(command)

    def send_command_to(self, port_name, command):
        """Send a command to a specific Arduino"""
        if port_name in self.connections:
            return self.connections[port_name].send_command(command)
        return False

    def send_command_by_role(self, role, command):
        """Send a command to all Arduinos with a specific role"""
        success = False
        for arduino in self.connections.values():
            if arduino.role == role:
                if arduino.send_command(command):
                    success = True
        return success

    def start_read_thread(self):
        """Start a thread to read data from all connected Arduino boards"""
        if self.read_thread is None or not self.read_thread.is_alive():
            self.stop_event.clear()
            self.read_thread = threading.Thread(
                target=self._read_thread_func, daemon=True
            )
            self.read_thread.start()
            logger.info("Arduino read thread started")
            return True
        return False

    def stop_read_thread(self):
        """Stop the read thread"""
        self.stop_event.set()
        if self.read_thread:
            self.read_thread.join(timeout=1.0)
            self.read_thread = None
            logger.info("Arduino read thread stopped")

    def _read_thread_func(self):
        """Thread function to read data from all connected Arduino boards"""
        while not self.stop_event.is_set():
            for port, arduino in list(self.connections.items()):
                data = arduino.read_data()
                if data:
                    logger.info(f"Received from {port}: {data}")

                    # If socketio is available, emit the data
                    if self.socketio:
                        self.socketio.emit(
                            "arduino_response",
                            {"data": data, "port": port, "role": arduino.role},
                        )

            time.sleep(0.01)

    def get_status(self):
        """Get the status of all connections"""
        return {
            port: {
                "connected": arduino.connected,
                "role": arduino.role,
                "last_data": arduino.last_data,
            }
            for port, arduino in self.connections.items()
        }

    def get_connection_by_role(self, role):
        """Get a list of connections with a specific role"""
        return [
            arduino for arduino in self.connections.values() if arduino.role == role
        ]


# Placeholder MockSerial for testing
class MockSerial:
    def __init__(self):
        logger.info("Using MockSerial")

    def write(self, data):
        logger.info(f"MockSerial write: {data}")

    def close(self):
        logger.info("MockSerial closed")

    def reset_input_buffer(self):
        pass

    def readline(self):
        return b"MOCK SERIAL DATA"


# Globals
current_controls = {"steering": 0, "brake": 0, "accelerator": 0, "speed_cap": False}

# --- Flask and SocketIO setup ---
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your-secret-key")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize Arduino manager
arduino_manager = ArduinoManager(socketio=socketio)

# Initialize model pipeline
model_pipeline = ModelPipeline(device="0")

# Thread control
stop_event = threading.Event()
camera_thread = None


# --- Thread definitions ---
def camera_thread_func():
    """Continuously capture frames and emit over socket."""
    logger.info("Camera thread starting")

    try:
        # Open local camera if available and not disabled
        local_cap = None if DISABLE_CAMERA else cv2.VideoCapture(0)
        if local_cap is not None and not local_cap.isOpened():
            logger.warning("Local camera not available")
            local_cap = None
        elif local_cap is not None:
            logger.info("Local camera opened successfully")

        while not stop_event.is_set():
            frame = None
            source_text = "No Camera"

            # Fallback to local camera
            if frame is None and local_cap is not None:
                ret, frame = local_cap.read()
                if not ret:
                    logger.warning("Failed to get frame from local camera")
                    time.sleep(0.05)
                    continue
                source_text = "Local Camera"

            # If we don't have a frame, sleep and try again
            if frame is None:
                time.sleep(0.05)
                continue

            # Process the frame with our model pipeline
            processed_frame = model_pipeline.process_frame(frame)

            # Add source indicator
            cv2.putText(
                processed_frame,
                source_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Encode to JPEG then base64
            _, buffer = cv2.imencode(".jpg", processed_frame)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")

            # Send frame to client
            socketio.emit("camera_frame", {"image": jpg_as_text, "source": source_text})

            time.sleep(0.05)

        # Clean up
        if local_cap is not None:
            local_cap.release()
    except Exception as e:
        logger.error(f"Camera thread error: {e}")
    logger.info("Camera thread ended")


def close_camera():
    stop_event.set()
    # camera release handled in thread


# Signal handler
def signal_handler(sig, frame):
    logger.info("Ctrl+C detected, shutting down...")
    stop_event.set()
    arduino_manager.disconnect_all()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# Helper function to log to UI
def append_log(message):
    socketio.emit("log_message", {"message": message})


# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/control")
def control():
    return render_template("control.html")


# Arduino management API endpoints
@app.route("/api/arduino/scan", methods=["POST"])
def arduino_scan():
    num_connected = arduino_manager.connect_all()

    # Start the read thread if it's not already running
    arduino_manager.start_read_thread()

    return jsonify({"success": True, "count": num_connected})


@app.route("/api/arduino/status", methods=["GET"])
def arduino_status():
    connections = arduino_manager.get_status()
    return jsonify(
        {"success": True, "count": len(connections), "connections": connections}
    )


@app.route("/api/arduino/disconnect", methods=["POST"])
def arduino_disconnect():
    port = request.args.get("port")
    if not port:
        return jsonify({"success": False, "error": "Port parameter is required"})

    success = arduino_manager.disconnect_from(port)
    return jsonify({"success": success})


@app.route("/api/arduino/disconnect_all", methods=["POST"])
def arduino_disconnect_all():
    arduino_manager.disconnect_all()
    return jsonify({"success": True})


@app.route("/api/arduino/assign_role", methods=["POST"])
def arduino_assign_role():
    data = request.json
    port = data.get("port")
    role = data.get("role")

    if not port or not role:
        return jsonify(
            {"success": False, "error": "Port and role parameters are required"}
        )

    if port in arduino_manager.connections:
        arduino_manager.connections[port].role = role
        logger.info(f"Assigned role '{role}' to Arduino on {port}")
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": f"No Arduino connected on {port}"})


# --- SocketIO event handlers ---
@socketio.on("connect")
def handle_connect():
    emit("control_update", current_controls)
    emit(
        "camera_status",
        {
            "active": camera_thread is not None
            and camera_thread.is_alive()
            and not stop_event.is_set(),
            "detection_enabled": model_pipeline.object_detection_enabled,
            "lane_detection_enabled": model_pipeline.lane_detection_enabled,
        },
    )
    logger.info("Client connected")
    append_log("WebSocket connected")


@socketio.on("control_change")
def handle_control_change(data):
    global current_controls

    # Log received commands to console
    logger.info(f"Command received: {data}")

    # Update current_controls with any values from data
    if "steering" in data:
        current_controls["steering"] = data["steering"]
    if "brake" in data:
        current_controls["brake"] = data["brake"]
    if "accelerator" in data:
        current_controls["accelerator"] = data["accelerator"]
    if "speed_cap" in data:
        current_controls["speed_cap"] = data["speed_cap"]

    # Broadcast to all clients
    emit("control_update", current_controls, broadcast=True)

    # Also broadcast a command_update event for the command monitor
    emit("command_update", data, broadcast=True)

    # Send commands to Arduino via the manager
    # Format as the Arduino expects: S<value>,B<value>,A<value>
    command = f"S{current_controls['steering']},B{current_controls['brake']},A{
        current_controls['accelerator']
    }"

    # Send to all connected Arduinos
    arduino_manager.send_command_all(command)

    # Send specialized commands to Arduinos with specific roles
    if "steering" in data:
        arduino_manager.send_command_by_role("steering", f"STEER:{data['steering']}")

    if "brake" in data:
        arduino_manager.send_command_by_role("motor", f"BRAKE:{data['brake']}")

    if "accelerator" in data:
        arduino_manager.send_command_by_role("motor", f"ACCEL:{data['accelerator']}")


@socketio.on("camera_control")
def handle_camera_control(data):
    global camera_thread

    action = data.get("action")
    if action == "start":
        # Start camera if not already running
        if camera_thread is None or not camera_thread.is_alive():
            logger.info("Starting camera")
            stop_event.clear()
            camera_thread = threading.Thread(target=camera_thread_func, daemon=True)
            camera_thread.start()
            append_log("Camera started")
    elif action == "stop":
        # Stop camera if running
        if camera_thread and camera_thread.is_alive():
            logger.info("Stopping camera")
            stop_event.set()
            camera_thread.join(timeout=1.0)
            camera_thread = None
            append_log("Camera stopped")
    elif action == "toggle_detection":
        # Toggle object detection
        enabled = not model_pipeline.object_detection_enabled
        status = model_pipeline.set_object_detection(enabled)
        logger.info(f"Object detection {'enabled' if status else 'disabled'}")
        append_log(f"Object detection {'enabled' if status else 'disabled'}")
    elif action == "toggle_lane_detection":
        # Toggle lane detection
        enabled = not model_pipeline.lane_detection_enabled
        status = model_pipeline.set_lane_detection(enabled)
        logger.info(f"Lane detection {'enabled' if status else 'disabled'}")
        append_log(f"Lane detection {'enabled' if status else 'disabled'}")

    # Get current status
    status = model_pipeline.get_status()

    # Send back current status
    emit(
        "camera_status",
        {
            "active": camera_thread is not None
            and camera_thread.is_alive()
            and not stop_event.is_set(),
            "detection_enabled": status["object_detection_enabled"],
            "lane_detection_enabled": status["lane_detection_enabled"],
        },
    )


@socketio.on("log_message")
def handle_log_message(data):
    message = data.get("message", "")
    if message:
        append_log(message)


@socketio.on("get_system_state")
def handle_get_system_state():
    emit("control_update", current_controls)
    # Log this request
    logger.info("System state requested")


# --- Main entry ---
if __name__ == "__main__":
    logger.info("Starting server...")

    # Scan for Arduino devices
    arduino_count = arduino_manager.connect_all()
    logger.info(f"Found {arduino_count} Arduino devices")

    # Start Arduino read thread
    if arduino_count > 0:
        arduino_manager.start_read_thread()

    # Only start camera thread if not disabled
    if not DISABLE_CAMERA:
        logger.info("Starting camera thread...")
        camera_thread = threading.Thread(target=camera_thread_func, daemon=True)
        camera_thread.start()
    else:
        logger.info("Camera disabled")

    logger.info("Starting Flask-SocketIO server...")
    try:
        socketio.run(
            app, host="0.0.0.0", port=5000, debug=USE_DEBUG, use_reloader=False
        )
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
    except (KeyboardInterrupt, SystemExit):
        logger.info("Server interrupted")
        signal_handler(None, None)
