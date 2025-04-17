# --- (Imports and Setup) ---
import socket
import time
import sys
import json
import threading
import queue
import math
import traceback
import os

# --- NEW IMPORTS ---
try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: OpenCV or NumPy not found. pip install opencv-python numpy")
    sys.exit(1)
# --- END NEW IMPORTS ---

try:
    import pygame
    # --- NEW PYGAME INIT --- Needed for surfarray
    pygame.init()
    pygame.font.init()
    # --- END NEW PYGAME INIT ---
except ImportError:
    print("ERROR: Pygame not found or failed to init. pip install pygame")
    sys.exit(1)

# --- Configuration and Constants ---
CAR_IP = "192.168.4.1"
CAR_PORT = 100
COMMAND_ID = "Elegoo" # Assuming 'Elegoo' is your car's ID
SEND_INTERVAL = 0.1
RECEIVE_BUFFER_SIZE = 1024
HEARTBEAT_INTERVAL = 1.0

# Video/Face Tracking Config
VIDEO_STREAM_URL = f"http://{CAR_IP}:81/stream" # Updated Port 81
DNN_PROTOTXT = "deploy.prototxt.txt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_CONFIDENCE_THRESHOLD = 0.6 # Minimum confidence to consider a detection
FACE_CENTER_TOLERANCE_X = 0.15 # % of frame width (+/-) to consider centered
NO_FACE_STOP_DELAY = 0.75 # Seconds without detecting face before stopping rotation

# --- UPDATED PYGAME WINDOW ---
STREAM_DISPLAY_WIDTH = 320 # Desired width for video in window
STREAM_DISPLAY_HEIGHT = 240 # Desired height for video in window
CONTROLS_HEIGHT = 200      # Height needed for joystick + text
WINDOW_WIDTH = STREAM_DISPLAY_WIDTH # Window width matches video display width
WINDOW_HEIGHT = STREAM_DISPLAY_HEIGHT + CONTROLS_HEIGHT # Total height

# --- Adjusted Joystick Params ---
JOYSTICK_CENTER_X = WINDOW_WIDTH // 2
JOYSTICK_CENTER_Y = STREAM_DISPLAY_HEIGHT + (CONTROLS_HEIGHT // 2) - 20 # Center in lower part
BOUNDARY_RADIUS = 60 # Adjusted size
KNOB_RADIUS = 18
DEAD_ZONE_RADIUS = 10

# Colors
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (150, 150, 150)
COLOR_BLUE = (100, 100, 255)
COLOR_RED = (255, 100, 100)
COLOR_GREEN = (100, 255, 100)

# Car Control Params & Actions (Hybrid N=4 / N=3 logic)
DEFAULT_SPEED = 100  # Speed for N=4 FORWARD/BACKWARD
TURNING_SPEED = 75  # Speed for N=3 LEFT/RIGHT turns
ACTION_STOP = 'STOP'         # Uses N=4
ACTION_FORWARD = 'FORWARD'     # Uses N=4
ACTION_BACKWARD = 'BACKWARD'   # Uses N=4 (with negative speed attempt)
ACTION_LEFT = 'LEFT'         # Uses N=3 (NEW)
ACTION_RIGHT = 'RIGHT'        # Uses N=3 (NEW)
ACTION_QUIT = 'QUIT'

# --- Shared State ---
action_queue = queue.Queue(maxsize=5)
frame_queue = queue.Queue(maxsize=2)
socket_lock = threading.Lock()
stop_event = threading.Event()
face_tracking_enabled = False
tracking_status_lock = threading.Lock()

# --- JSON Command Creation Functions ---

# == N=4 Functions (for Stop, Forward, Backward) ==
def create_stop_command_json():
    # Using N=4 for stop
    command_dict = {"H": COMMAND_ID, "N": 4, "D1": 0, "D2": 0 }
    return json.dumps(command_dict) + "\n"

def create_n4_command_json(left_speed, right_speed):
    """ Creates N=4 command, attempting negative speeds. Used for Fwd/Bwd. """
    left_speed_clamped = max(-255, min(255, int(left_speed)))
    right_speed_clamped = max(-255, min(255, int(right_speed)))
    command_dict = {"H": COMMAND_ID, "N": 4, "D1": left_speed_clamped, "D2": right_speed_clamped}
    return json.dumps(command_dict) + "\n"

# --- N3 TURNS FIX: Add N=3 Function & Codes (ONLY for Turning) ---
# == N=3 Function & Codes (ONLY for Left/Right Turns) ==
DIR_CODE_LEFT = 1  # Assumed code for Left Turn using N=3
DIR_CODE_RIGHT = 2 # Assumed code for Right Turn using N=3

def create_n3_command_json(direction_code, speed):
    """Creates a JSON command string using the N=3 format. Used for Turns."""
    speed_clamped = max(0, min(255, int(speed)))
    command_dict = {"H": COMMAND_ID, "N": 3, "D1": direction_code, "D2": speed_clamped}
    return json.dumps(command_dict) + "\n"
# --- END N3 TURNS FIX ---

# --- Pre-defined Commands (Hybrid N=4 / N=3) ---
# N=4 based commands
CMD_STOP = create_stop_command_json() # N=4, D1=0, D2=0
CMD_FORWARD = create_n4_command_json(DEFAULT_SPEED, DEFAULT_SPEED)
CMD_BACKWARD = create_n4_command_json(-DEFAULT_SPEED, -DEFAULT_SPEED) # N=4 w/ negative attempt

# --- N3 TURNS FIX: Define Left/Right using N=3 ---
CMD_LEFT = create_n3_command_json(DIR_CODE_LEFT, TURNING_SPEED)   # N=3 based turn
CMD_RIGHT = create_n3_command_json(DIR_CODE_RIGHT, TURNING_SPEED)  # N=3 based turn
# --- END N3 TURNS FIX ---

CMD_HEARTBEAT = "{Heartbeat}\n" # Unchanged


# --- Connection Function (Unchanged) ---
def connect_to_car(ip, port):
    print(f"Attempting persistent connection to {ip}:{port}...")
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((ip, port))
        sock.settimeout(1.0) # Shorter timeout for operations
        print("Connection successful.")
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            print("SO_KEEPALIVE enabled.")
        except (AttributeError, OSError) as e:
            print(f"Note: SO_KEEPALIVE not supported or error: {e}")
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("TCP_NODELAY enabled.")
        except (AttributeError, OSError) as e:
            print(f"Note: TCP_NODELAY not supported or error: {e}")
        return sock
    except socket.timeout:
        print(f"Error: Connection timeout.")
        if sock: sock.close()
        return None
    except socket.error as e:
        print(f"Error connecting: {e}")
        if sock: sock.close()
        return None
    except Exception as e:
        print(f"Unexpected connection error: {e}")
        if sock: sock.close()
        return None

# --- Send/Receive Function (MODIFIED DEBUG FOR HYBRID) ---
def send_json_command(sock, json_cmd_string, thread_name=""):
    if not sock or stop_event.is_set():
        return False
    with socket_lock:
        if stop_event.is_set(): # Double check inside lock
             return False
        try:
            # --- N3 TURNS FIX: Modified Debug Print for Hybrid N=3/N=4 ---
            is_n4_motor = '"N": 4' in json_cmd_string
            is_n3_motor = '"N": 3' in json_cmd_string
            is_heartbeat = json_cmd_string == CMD_HEARTBEAT
            is_stop_cmd = is_n4_motor and '"D1": 0' in json_cmd_string and '"D2": 0' in json_cmd_string

            # Print N=3 or N=4 motor commands (unless it's repetitive STOP) and Heartbeat
            if (is_n4_motor or is_n3_motor or is_heartbeat):
                # Avoid printing STOP continuously if it's the current command from Sender
                if not (is_stop_cmd and thread_name == "Sender"):
                    print(f"DEBUG ({thread_name}): Sending Command: {json_cmd_string.strip()}")
            # --- END N3 TURNS FIX ---

            # 1. Send
            sock.sendall(json_cmd_string.encode('utf-8'))

            # 2. Attempt Non-Blocking Read
            try:
                sock.setblocking(False)
                data = sock.recv(RECEIVE_BUFFER_SIZE)
                # if data: print(f"DEBUG ({thread_name}): Received: {data.decode('utf-8', errors='ignore').strip()}") # Verbose
            except BlockingIOError: pass
            except socket.timeout: pass
            except (ConnectionAbortedError, OSError) as e:
                print(f"FATAL ({thread_name}): Error during non-blocking read: {e}")
                traceback.print_exc(); stop_event.set(); return False
            except Exception as e:
                 print(f"FATAL ({thread_name}): Unexpected Exception during non-blocking read: {type(e).__name__}: {e}")
                 traceback.print_exc(); stop_event.set(); return False
            finally:
                try:
                    if sock and sock.fileno() != -1: sock.setblocking(True)
                except socket.error: pass
            return True

        except socket.timeout:
            print(f"Error ({thread_name}): Send operation timed out.")
            stop_event.set(); return False
        except (ConnectionAbortedError, OSError) as e:
            print(f"FATAL ({thread_name}): Error during send: {e}")
            traceback.print_exc(); stop_event.set(); return False
        except Exception as e:
            print(f"FATAL ({thread_name}): Unexpected Exception during send: {type(e).__name__}: {e}")
            traceback.print_exc(); stop_event.set(); return False


# --- Sending Thread Function (MODIFIED FOR HYBRID COMMAND MAP) ---
def command_sender(sock):
    print("Command sender thread started.")
    current_action = ACTION_STOP
    last_action_sent_json = None
    last_heartbeat_sent_time = time.time()

    # --- N3 TURNS FIX: Hybrid Command Map ---
    # Map actions to their corresponding JSON command strings
    command_map = {
        ACTION_STOP: CMD_STOP,          # N=4 based
        ACTION_FORWARD: CMD_FORWARD,    # N=4 based
        ACTION_BACKWARD: CMD_BACKWARD,  # N=4 based (attempting negative speed)
        ACTION_LEFT: CMD_LEFT,          # N=3 based Turn (NEW)
        ACTION_RIGHT: CMD_RIGHT,        # N=3 based Turn (NEW)
    }
    # --- END N3 TURNS FIX ---

    while not stop_event.is_set():
        now = time.time()
        new_action_received = False
        newest_action = None

        # Get latest action from queue
        try:
            while True:
                newest_action = action_queue.get_nowait()
                new_action_received = True
                action_queue.task_done()
        except queue.Empty:
            if new_action_received and newest_action != current_action:
                 current_action = newest_action
                 # print(f"Sender: New action received: {current_action}") # Optional Debug
            pass

        # Handle QUIT action
        if current_action == ACTION_QUIT:
            print("Sender: QUIT action. Sending final STOP (N=4).")
            # Ensure final stop uses N=4 stop command
            send_json_command(sock, CMD_STOP, "Sender-Quit")
            time.sleep(0.1)
            stop_event.set()
            break

        # Determine Control Command (using hybrid map)
        command_to_send = command_map.get(current_action)

        if command_to_send is None: # Unknown action
            print(f"Warning: Unknown action '{current_action}'. Sending STOP (N=4).")
            command_to_send = CMD_STOP
            current_action = ACTION_STOP

        # Send Control Command (if command JSON changed)
        if command_to_send is not None and command_to_send != last_action_sent_json:
             # print(f"Sender: Intending action: {current_action} -> JSON: {command_to_send.strip()}") # Optional debug
             if not send_json_command(sock, command_to_send, "Sender"):
                 print("DEBUG (Sender): Send/Read failure on control command, breaking loop.")
                 stop_event.set(); break
             else:
                 last_action_sent_json = command_to_send # Update only on successful send

        # Send Heartbeat Reply (if needed)
        if now - last_heartbeat_sent_time >= HEARTBEAT_INTERVAL:
            if not send_json_command(sock, CMD_HEARTBEAT, "Sender-HB"):
                 print("DEBUG (Sender): Send/Read failure on heartbeat reply, breaking loop.")
                 stop_event.set(); break
            else:
                 last_heartbeat_sent_time = now

        # Wait efficiently
        next_heartbeat_time = last_heartbeat_sent_time + HEARTBEAT_INTERVAL
        wait_time = max(0.01, min(next_heartbeat_time - now, SEND_INTERVAL))
        stop_event.wait(timeout=wait_time)

    print("Command sender thread finished.")


# --- Video Processing Thread (Unchanged - Still outputs ACTION_LEFT/RIGHT) ---
# The *effect* of ACTION_LEFT/RIGHT is now determined by the N=3 commands they map to.
def video_processor():
    global action_queue, frame_queue
    global face_tracking_enabled, tracking_status_lock
    print("Video processor thread started.")

    # Load DNN Model
    if not os.path.exists(DNN_PROTOTXT) or not os.path.exists(DNN_MODEL):
        print(f"ERROR: DNN model files not found! Ensure '{DNN_PROTOTXT}' and '{DNN_MODEL}' are present."); stop_event.set(); return
    try:
        print("Loading DNN face detection model..."); net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_MODEL); print("DNN model loaded successfully.")
    except cv2.error as e: print(f"ERROR: Failed to load DNN model: {e}"); traceback.print_exc(); stop_event.set(); return

    # Video Capture
    print(f"Attempting to open video stream: {VIDEO_STREAM_URL}"); cap = cv2.VideoCapture(VIDEO_STREAM_URL)
    if not cap.isOpened(): print(f"ERROR: Cannot open video stream at {VIDEO_STREAM_URL}"); stop_event.set(); return
    print("Video stream opened successfully.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0: print("WARNING: Using default frame size (320x240)."); frame_width, frame_height = 320, 240
    frame_center_x = frame_width // 2; tolerance_pixels_x = int(frame_width * FACE_CENTER_TOLERANCE_X)
    print(f"VidProc: Frame {frame_width}x{frame_height}, CenterX: {frame_center_x}, Tol: +/- {tolerance_pixels_x}px")

    last_face_detected_time = 0; last_tracking_action = ACTION_STOP

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None: print("VideoProc: Error reading frame."); time.sleep(0.5); continue

        with tracking_status_lock: is_tracking = face_tracking_enabled
        processed_frame = frame.copy()

        if is_tracking:
            (h, w) = processed_frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(processed_frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob); detections = net.forward()
            best_face_box = None; max_area = 0

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > DNN_CONFIDENCE_THRESHOLD:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    startX, startY = max(0, startX), max(0, startY); endX, endY = min(w - 1, endX), min(h - 1, endY)
                    area = (endX - startX) * (endY - startY)
                    if area > max_area: max_area = area; best_face_box = (startX, startY, endX, endY)

            current_action = ACTION_STOP
            if best_face_box is not None:
                last_face_detected_time = time.time()
                (startX, startY, endX, endY) = best_face_box
                face_center_x = startX + (endX - startX) // 2; face_center_y = startY + (endY - startY) // 2
                error_x = face_center_x - frame_center_x

                # Logic decides *which way* to turn, N=3 command executes it
                if error_x > tolerance_pixels_x: current_action = ACTION_RIGHT    # Face right -> Turn Left (N=3)
                elif error_x < -tolerance_pixels_x: current_action = ACTION_LEFT # Face left -> Turn Right (N=3)
                else: current_action = ACTION_STOP                               # Face centered -> Stop (N=4)

                cv2.rectangle(processed_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.circle(processed_frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
            else: # No face
                if time.time() - last_face_detected_time > NO_FACE_STOP_DELAY: current_action = ACTION_STOP
                else: current_action = last_tracking_action

            # Queue Action if Changed
            if current_action != last_tracking_action:
                 try:
                     while not action_queue.empty(): action_queue.get_nowait(); action_queue.task_done()
                     action_queue.put_nowait(current_action); last_tracking_action = current_action
                 except queue.Full: pass
        else: # Not tracking
            if last_tracking_action != ACTION_STOP:
                try:
                     while not action_queue.empty(): action_queue.get_nowait(); action_queue.task_done()
                     action_queue.put_nowait(ACTION_STOP); last_tracking_action = ACTION_STOP
                except queue.Full: pass

        # Put frame onto display queue
        try:
            while frame_queue.qsize() >= 2: frame_queue.get_nowait(); frame_queue.task_done()
            frame_queue.put_nowait(processed_frame)
        except queue.Full: pass
        except Exception as q_err: print(f"VideoProc: Error putting frame on queue: {q_err}")

        time.sleep(0.03)

    print("Video processor thread finishing..."); cap.release(); print("Video stream released.")


# --- Joystick Emulator Function (Unchanged - Still generates ACTION_ constants) ---
# The *effect* of ACTION_LEFT/RIGHT is now determined by N=3 commands.
def run_joystick_emulator():
    global action_queue, frame_queue
    global face_tracking_enabled, tracking_status_lock

    # --- N3 TURNS FIX: Update window caption ---
    pygame.display.set_caption("Car Control & Stream (N=4 Fwd/Bwd, N=3 Turns)")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    # --- END N3 TURNS FIX ---
    clock = pygame.time.Clock(); font = pygame.font.Font(None, 24)
    joystick_active = False; current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y]
    last_queued_action = None; is_tracking_local = False; latest_frame_surface = None

    # --- N3 TURNS FIX: Update Interface Info ---
    print("\n--- Control Interface (Hybrid: N=4 Fwd/Bwd, N=3 Turns) ---")
    print("Video Stream at Top")
    print("Joystick Control at Bottom")
    print("NOTE: Left/Right turns use N=3 (standard turns).")
    print("NOTE: Forward/Backward use N=4 (attempting pivot/neg speed).")
    # --- END N3 TURNS FIX ---
    print("T: Toggle Face Tracking")
    print("Close window to Quit.")
    print("--------------------------")

    running = True
    while running and not stop_event.is_set():
        manual_action = ACTION_STOP
        with tracking_status_lock: is_tracking_local = face_tracking_enabled

        for event in pygame.event.get():
            if event.type == pygame.QUIT: print("Joystick: Quit via window close."); manual_action = ACTION_QUIT; running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                     with tracking_status_lock: face_tracking_enabled = not face_tracking_enabled; is_tracking_local = face_tracking_enabled
                     print(f"Face Tracking {'ENABLED' if is_tracking_local else 'DISABLED'}")
                     if not is_tracking_local and not joystick_active: manual_action = ACTION_STOP
                     elif is_tracking_local: last_queued_action = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    if my > STREAM_DISPLAY_HEIGHT:
                         dist_sq = (mx - JOYSTICK_CENTER_X)**2 + (my - JOYSTICK_CENTER_Y)**2
                         if dist_sq <= BOUNDARY_RADIUS**2:
                             if not is_tracking_local: joystick_active = True; current_knob_pos = list(event.pos)
                             else: print("Joystick: Cannot activate while tracking is ON.")
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if joystick_active: joystick_active = False; current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y]; manual_action = ACTION_STOP
            elif event.type == pygame.MOUSEMOTION:
                if joystick_active and not is_tracking_local: current_knob_pos = list(event.pos)

        try:
            frame_bgr = frame_queue.get_nowait(); frame_queue.task_done()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            f_surf = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
            latest_frame_surface = pygame.transform.scale(f_surf, (STREAM_DISPLAY_WIDTH, STREAM_DISPLAY_HEIGHT))
        except queue.Empty: pass
        except Exception as e: print(f"Error processing frame for display: {type(e).__name__}: {e}")

        if not is_tracking_local:
            if joystick_active:
                dx = current_knob_pos[0] - JOYSTICK_CENTER_X; dy = current_knob_pos[1] - JOYSTICK_CENTER_Y
                dist = math.sqrt(dx**2 + dy**2)
                if dist > BOUNDARY_RADIUS: scale = BOUNDARY_RADIUS / dist; current_knob_pos[0] = JOYSTICK_CENTER_X + dx * scale; current_knob_pos[1] = JOYSTICK_CENTER_Y + dy * scale; dist = BOUNDARY_RADIUS
                if dist > DEAD_ZONE_RADIUS:
                    angle = math.atan2(-dy, dx)
                    if -math.pi*0.25 <= angle < math.pi*0.25: manual_action = ACTION_RIGHT    # Queues RIGHT (N=3)
                    elif math.pi*0.25 <= angle < math.pi*0.75: manual_action = ACTION_FORWARD   # Queues FORWARD (N=4)
                    elif angle >= math.pi*0.75 or angle < -math.pi*0.75: manual_action = ACTION_LEFT # Queues LEFT (N=3)
                    elif -math.pi*0.75 <= angle < -math.pi*0.25: manual_action = ACTION_BACKWARD # Queues BACKWARD (N=4)
                    else: manual_action = ACTION_STOP # Queues STOP (N=4)
                else: manual_action = ACTION_STOP; current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y]
            else: manual_action = ACTION_STOP; current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y]
        else: manual_action = ACTION_STOP; current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y]; joystick_active = False

        action_to_queue = manual_action
        if not is_tracking_local and (action_to_queue != last_queued_action or action_to_queue == ACTION_QUIT):
             try:
                 while not action_queue.empty(): action_queue.get_nowait(); action_queue.task_done()
                 action_queue.put_nowait(action_to_queue); last_queued_action = action_to_queue
             except queue.Full: print("Joystick ERROR: Queue full even after clearing.")
             if action_to_queue == ACTION_QUIT: running = False
        elif is_tracking_local:
             if last_queued_action is not None: last_queued_action = None

        screen.fill(COLOR_WHITE)
        if latest_frame_surface: screen.blit(latest_frame_surface, (0, 0))
        else: pygame.draw.rect(screen, COLOR_GRAY, (0, 0, STREAM_DISPLAY_WIDTH, STREAM_DISPLAY_HEIGHT)); placeholder = font.render("Waiting...", True, COLOR_BLACK); screen.blit(placeholder, placeholder.get_rect(center=(STREAM_DISPLAY_WIDTH//2, STREAM_DISPLAY_HEIGHT//2)))

        pygame.draw.circle(screen, COLOR_GRAY, (JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y), BOUNDARY_RADIUS, 2)
        pygame.draw.circle(screen, COLOR_RED, (JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y), DEAD_ZONE_RADIUS, 1)
        knob_color = COLOR_BLUE if joystick_active and not is_tracking_local else COLOR_BLACK
        knob_center_int = (int(round(current_knob_pos[0])), int(round(current_knob_pos[1])))
        pygame.draw.circle(screen, knob_color, knob_center_int, KNOB_RADIUS)

        mode_text = f"Mode: {'FACE TRACKING' if is_tracking_local else 'MANUAL'}"
        mode_color = COLOR_GREEN if is_tracking_local else COLOR_BLACK
        last_action_disp = '---'
        if not is_tracking_local and last_queued_action is not None: last_action_disp = last_queued_action
        elif is_tracking_local: last_action_disp = 'TRACKING ACTIVE'
        last_action_text = f"Action: {last_action_disp}"
        instruction_text = ["T: Toggle Mode", mode_text, last_action_text]
        y_offset = STREAM_DISPLAY_HEIGHT + CONTROLS_HEIGHT - 70
        for i, line in enumerate(instruction_text):
             color = mode_color if "Mode:" in line else COLOR_BLACK
             text_surface = font.render(line, True, color)
             text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, y_offset + i * 20))
             screen.blit(text_surface, text_rect)

        pygame.display.flip(); clock.tick(30)

    print("Joystick emulator loop finished.")
    if last_queued_action != ACTION_QUIT:
        print("Joystick: Loop exited without QUIT. Queuing final QUIT.")
        try:
             while not action_queue.empty(): action_queue.get_nowait(); action_queue.task_done()
             action_queue.put_nowait(ACTION_QUIT)
        except queue.Full: print("Joystick WARN: Could not queue final QUIT action on cleanup.")
    stop_event.set(); print("Joystick thread signalling stop.")


# --- Main Execution (MODIFIED TO REFLECT HYBRID CONTROL) ---
if __name__ == "__main__":
    # --- N3 TURNS FIX: Updated title and info ---
    print("--- Elegoo Car Controller w/ Face Tracking & Video (Hybrid Control) ---")
    print(f"Target: {CAR_IP}:{CAR_PORT} | Video: {VIDEO_STREAM_URL}")
    now_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {now_time}")
    print("----------------------------------------------------")
    print("!!! Using HYBRID control scheme:")
    print("!!!   - N=4 (Individual Wheels) for FORWARD, BACKWARD, STOP.")
    print("!!!     (Attempting negative speeds for BACKWARD)")
    print("!!!   - N=3 (Direction Code) for LEFT, RIGHT turns.")
    print("!!!     (Direction Codes Assumed: 3=Left, 4=Right)")
    print("!!!     (Turns will likely NOT be pivot turns)")
    print("----------------------------------------------------")
    # --- END N3 TURNS FIX ---

    car_socket = None; sender_thread = None; joystick_thread = None; video_thread = None
    try: action_queue.put_nowait(ACTION_STOP)
    except queue.Full: print("WARN: Could not put initial STOP action.")

    try:
        car_socket = connect_to_car(CAR_IP, CAR_PORT)
        if car_socket:
            print("Connection established. Starting threads...")
            stop_event.clear()
            sender_thread = threading.Thread(target=command_sender, args=(car_socket,), name="CommandSenderThread", daemon=True); sender_thread.start()
            video_thread = threading.Thread(target=video_processor, name="VideoProcessorThread", daemon=True); video_thread.start()
            joystick_thread = threading.Thread(target=run_joystick_emulator, name="JoystickThread"); joystick_thread.start()
            joystick_thread.join()
            print("Main: Joystick thread finished.")
        else: print("Failed initial connection. Exiting."); sys.exit(1)
    except KeyboardInterrupt: print("\nMain: KeyboardInterrupt received."); stop_event.set()
    except Exception as e: print(f"\nMain Thread Exception: {type(e).__name__}: {e}"); traceback.print_exc(); stop_event.set()
    finally:
        print("Main: Starting Cleanup...")
        stop_event.set()
        if joystick_thread and joystick_thread.is_alive(): print("Main: Waiting for joystick thread..."); joystick_thread.join(timeout=1.0)
        if video_thread and video_thread.is_alive(): print("Main: Waiting for video thread..."); video_thread.join(timeout=2.0)
        if sender_thread and sender_thread.is_alive(): print("Main: Waiting for sender thread..."); sender_thread.join(timeout=2.0)

        if car_socket: # Try sending final stop (N=4 based)
             print("Main: Attempting final STOP (N=4) command before closing socket.")
             send_json_command(car_socket, CMD_STOP, "Main-Cleanup")
             time.sleep(0.1)
             print("Main: Closing socket.")
             with socket_lock:
                  try: car_socket.getpeername(); car_socket.shutdown(socket.SHUT_RDWR)
                  except (OSError, socket.error): pass
                  finally: car_socket.close(); print("Main: Socket closed.")

        if pygame.get_init(): print("Main: Quitting Pygame."); pygame.quit()
        print("Main: Program finished.")