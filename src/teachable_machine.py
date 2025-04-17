# --- START OF FILE labels.txt ---
# 0 go
# 1 stop
# 2 turn_left
# 3 turn_right
# 4 do_nothing
# Modify this code to use the labels associated with the teachable machine model and control the car accordingly.

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

# --- Standard CV/NumPy/Pygame Imports ---
try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: OpenCV or NumPy not found. pip install opencv-python numpy")
    sys.exit(1)

try:
    import pygame
    pygame.init()
    pygame.font.init()
except ImportError:
    print("ERROR: Pygame not found or failed to init. pip install pygame")
    sys.exit(1)

# --- TensorFlow Import (for Keras .h5 model) ---
try:
    # Using the full TensorFlow library now
    import tensorflow as tf
    from keras.models import load_model
    # Optional: Disable excessive TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress INFO and WARNING messages
    tf.get_logger().setLevel('ERROR')        # Suppress Python logging messages below ERROR
    print(f"INFO: Using TensorFlow version: {tf.__version__}")
except ImportError:
    print("ERROR: TensorFlow not found.")
    print("Please install the full TensorFlow library: pip install tensorflow")
    sys.exit(1)
# --- END TensorFlow Import ---


# --- Configuration and Constants ---
CAR_IP = "192.168.4.1" # <<< --- CHANGE THIS TO YOUR CAR'S IP ADDRESS --- <<<
CAR_PORT = 100
COMMAND_ID = "Elegoo"
SEND_INTERVAL = 0.1
RECEIVE_BUFFER_SIZE = 1024
HEARTBEAT_INTERVAL = 1.0

# --- Teachable Machine (TM) Model Config ---
# <<< CHANGED FOR KERAS .H5 >>>
KERAS_MODEL_PATH = "keras_model.h5"  # <<< CHANGE TO YOUR .h5 MODEL FILE
LABELS_PATH = "labels.txt"  # <<< CHANGE TO YOUR LABELS FILE
MODEL_INPUT_HEIGHT = 224            # <<< CHANGE based on your TM/Keras model input
MODEL_INPUT_WIDTH = 224             # <<< CHANGE based on your TM/Keras model input
# Normalization params likely same as TM export, but verify if model expects something else
TM_INPUT_MEAN = 127.5
TM_INPUT_STD = 127.5
# <<< END OF KERAS CHANGE >>>
TM_CONFIDENCE_THRESHOLD = 0.7       # Minimum confidence to act on prediction

# --- Video Config ---
VIDEO_STREAM_URL = f"http://{CAR_IP}:81/stream"

# --- Pygame Window Config (Unchanged) ---
STREAM_DISPLAY_WIDTH = 320
STREAM_DISPLAY_HEIGHT = 240
CONTROLS_HEIGHT = 220
WINDOW_WIDTH = STREAM_DISPLAY_WIDTH
WINDOW_HEIGHT = STREAM_DISPLAY_HEIGHT + CONTROLS_HEIGHT

# --- Joystick Params (Unchanged) ---
JOYSTICK_CENTER_X = WINDOW_WIDTH // 2
JOYSTICK_CENTER_Y = STREAM_DISPLAY_HEIGHT + (CONTROLS_HEIGHT // 2) - 30
BOUNDARY_RADIUS = 50
KNOB_RADIUS = 15
DEAD_ZONE_RADIUS = 10

# Colors (Unchanged)
COLOR_WHITE = (255, 255, 255); COLOR_BLACK = (0, 0, 0); COLOR_GRAY = (150, 150, 150)
COLOR_BLUE = (100, 100, 255); COLOR_RED = (255, 100, 100); COLOR_GREEN = (100, 255, 100)
COLOR_ORANGE = (255, 165, 0)

# Car Control Params & Actions (Unchanged)
DEFAULT_SPEED = 100; TURNING_SPEED = 75
ACTION_STOP = 'STOP'; ACTION_FORWARD = 'FORWARD'; ACTION_BACKWARD = 'BACKWARD'
ACTION_LEFT = 'LEFT'; ACTION_RIGHT = 'RIGHT'; ACTION_QUIT = 'QUIT'

# --- Shared State (Unchanged) ---
action_queue = queue.Queue(maxsize=5)
frame_queue = queue.Queue(maxsize=2)
tm_info_queue = queue.Queue(maxsize=2)
socket_lock = threading.Lock()
stop_event = threading.Event()
tm_control_enabled = False
tm_status_lock = threading.Lock()

# --- JSON Command Creation Functions (Unchanged) ---
def create_stop_command_json():
    return json.dumps({"H": COMMAND_ID, "N": 4, "D1": 0, "D2": 0 }) + "\n"
def create_n4_command_json(left_speed, right_speed):
    l = max(-255, min(255, int(left_speed))); r = max(-255, min(255, int(right_speed)))
    return json.dumps({"H": COMMAND_ID, "N": 4, "D1": l, "D2": r}) + "\n"
DIR_CODE_LEFT = 1; DIR_CODE_RIGHT = 2
def create_n3_command_json(direction_code, speed):
    s = max(0, min(255, int(speed)))
    return json.dumps({"H": COMMAND_ID, "N": 3, "D1": direction_code, "D2": s}) + "\n"
CMD_STOP = create_stop_command_json(); CMD_FORWARD = create_n4_command_json(DEFAULT_SPEED, DEFAULT_SPEED)
CMD_BACKWARD = create_n4_command_json(-DEFAULT_SPEED, -DEFAULT_SPEED)
CMD_LEFT = create_n3_command_json(DIR_CODE_LEFT, TURNING_SPEED)
CMD_RIGHT = create_n3_command_json(DIR_CODE_RIGHT, TURNING_SPEED)
CMD_HEARTBEAT = "{Heartbeat}\n"

# --- Connection Function (Unchanged) ---
def connect_to_car(ip, port):
    # (Code is identical to previous version)
    print(f"Attempting persistent connection to {ip}:{port}...")
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0); sock.connect((ip, port)); sock.settimeout(1.0)
        print("Connection successful.")
        try: sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1); print("SO_KEEPALIVE enabled.")
        except (AttributeError, OSError) as e: print(f"Note: SO_KEEPALIVE not supported or error: {e}")
        try: sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); print("TCP_NODELAY enabled.")
        except (AttributeError, OSError) as e: print(f"Note: TCP_NODELAY not supported or error: {e}")
        return sock
    except socket.timeout: print(f"Error: Connection timeout.");
    except socket.error as e: print(f"Error connecting: {e}")
    except Exception as e: print(f"Unexpected connection error: {e}")
    if sock: sock.close()
    return None

# --- Send/Receive Function (Unchanged) ---
def send_json_command(sock, json_cmd_string, thread_name=""):
    # (Code is identical to previous version)
    if not sock or stop_event.is_set(): return False
    with socket_lock:
        if stop_event.is_set(): return False
        try:
            is_n4_motor = '"N": 4' in json_cmd_string; is_n3_motor = '"N": 3' in json_cmd_string
            is_heartbeat = json_cmd_string == CMD_HEARTBEAT
            is_stop_cmd = is_n4_motor and '"D1": 0' in json_cmd_string and '"D2": 0' in json_cmd_string
            # Reduce debug spam for frequent commands
            # if (is_n4_motor or is_n3_motor or is_heartbeat):
            #      if not (is_stop_cmd and thread_name == "Sender"): # Don't print every STOP from sender if it was already stopped
            #          print(f"DEBUG ({thread_name}): Sending Command: {json_cmd_string.strip()}")

            sock.sendall(json_cmd_string.encode('utf-8'))
            try:
                sock.setblocking(False); data = sock.recv(RECEIVE_BUFFER_SIZE) # Attempt non-blocking read
            except BlockingIOError: pass # Expected if no data ready
            except socket.timeout: pass # Could happen if timeout set low, unlikely here
            except (ConnectionAbortedError, OSError) as e: print(f"FATAL ({thread_name}): Error during non-blocking read: {e}"); traceback.print_exc(); stop_event.set(); return False
            except Exception as e: print(f"FATAL ({thread_name}): Unexpected Exception during non-blocking read: {type(e).__name__}: {e}"); traceback.print_exc(); stop_event.set(); return False
            finally:
                try: # Ensure socket is blocking again
                    if sock and sock.fileno() != -1: sock.setblocking(True)
                except socket.error: pass # Ignore if socket already closed
            return True
        except socket.timeout: print(f"Error ({thread_name}): Send operation timed out."); stop_event.set(); return False
        except (ConnectionAbortedError, OSError) as e: print(f"FATAL ({thread_name}): Error during send: {e}"); traceback.print_exc(); stop_event.set(); return False
        except Exception as e: print(f"FATAL ({thread_name}): Unexpected Exception during send: {type(e).__name__}: {e}"); traceback.print_exc(); stop_event.set(); return False

# --- Sending Thread Function (Unchanged) ---
def command_sender(sock):
    # (Code is identical to previous version)
    print("Command sender thread started.")
    current_action = ACTION_STOP; last_action_sent_json = None
    last_heartbeat_sent_time = time.time()
    command_map = { ACTION_STOP: CMD_STOP, ACTION_FORWARD: CMD_FORWARD, ACTION_BACKWARD: CMD_BACKWARD,
                    ACTION_LEFT: CMD_LEFT, ACTION_RIGHT: CMD_RIGHT, }
    while not stop_event.is_set():
        now = time.time(); new_action_received = False; newest_action = None
        try:
            # Get the very latest action from the queue
            while True: newest_action = action_queue.get_nowait(); new_action_received = True; action_queue.task_done()
        except queue.Empty:
            # If we received a new action, update the current action
            if new_action_received and newest_action != current_action:
                 # print(f"Sender: Action changing from {current_action} to {newest_action}") # Debug action changes
                 current_action = newest_action

        # Handle QUIT action
        if current_action == ACTION_QUIT:
            print("Sender: QUIT action received. Sending final STOP (N=4).")
            send_json_command(sock, CMD_STOP, "Sender-Quit"); time.sleep(0.1); stop_event.set(); break # Signal stop and exit

        # Determine command to send based on current action
        command_to_send = command_map.get(current_action, CMD_STOP) # Default to STOP

        # Send action command ONLY if it changed
        if command_to_send != last_action_sent_json:
             if not send_json_command(sock, command_to_send, "Sender"): stop_event.set(); break # Error during send
             else: last_action_sent_json = command_to_send

        # Send heartbeat periodically
        if now - last_heartbeat_sent_time >= HEARTBEAT_INTERVAL:
            if not send_json_command(sock, CMD_HEARTBEAT, "Sender-HB"): stop_event.set(); break # Error during send
            else: last_heartbeat_sent_time = now

        # Calculate wait time: Either until next heartbeat or standard interval, whichever is sooner
        next_heartbeat_time = last_heartbeat_sent_time + HEARTBEAT_INTERVAL
        wait_time = max(0.01, min(next_heartbeat_time - now, SEND_INTERVAL)) # Ensure positive wait time

        # Wait efficiently until next action or stop signal
        stop_event.wait(timeout=wait_time)

    print("Command sender thread finished.")


# --- MODIFIED Video Processing Thread (Uses Keras .h5 Model & labels.txt logic) ---
def video_processor():
    global action_queue, frame_queue, tm_info_queue
    global tm_control_enabled, tm_status_lock
    print("Video processor thread started.")

    # --- Load Keras Model and Labels ---
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"ERROR: Keras model file not found at '{KERAS_MODEL_PATH}'"); stop_event.set(); return
    if not os.path.exists(LABELS_PATH):
        print(f"ERROR: Labels file not found at '{LABELS_PATH}'"); stop_event.set(); return

    try:
        # --- Custom Object Definition (moved inside try for TensorFlow context) ---
        # Required if your model uses DepthwiseConv2D and was saved in a way
        # that includes the 'groups' argument, which might cause issues in some TF versions.
        from tensorflow.keras.layers import DepthwiseConv2D
        class CustomDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                # Check if 'groups' is in kwargs and remove it if it is
                # This handles potential compatibility issues with some saved model formats
                if 'groups' in kwargs:
                    kwargs.pop('groups')
                super().__init__(*args, **kwargs)

        print(f"Loading Keras model from: {KERAS_MODEL_PATH}")
        # <<< KERAS MODEL LOADING >>>
        # Use custom_objects to handle potential DepthwiseConv2D issues
        model = load_model(KERAS_MODEL_PATH, compile=False, custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D})
        model_input_shape = model.input_shape
        print(f"Keras model loaded successfully. Input shape: {model_input_shape}")

        # Verify input size consistency (optional but recommended)
        if len(model_input_shape) == 4:
            height = model_input_shape[1]
            width = model_input_shape[2]
            # Use != None check in case shape is (None, None, None, 3)
            if height is not None and width is not None and \
               (height != MODEL_INPUT_HEIGHT or width != MODEL_INPUT_WIDTH):
                print(f"WARNING: Model expects input size ({height}x{width}) but config is ({MODEL_INPUT_HEIGHT}x{MODEL_INPUT_WIDTH}). Using model's size for processing.")
                # Optionally update global config if needed, though resizing below handles it
                # MODEL_INPUT_HEIGHT = height
                # MODEL_INPUT_WIDTH = width
        else:
            print(f"WARNING: Could not automatically determine input H/W from model shape {model_input_shape}. Using configured values.")

        # Load labels
        print(f"Loading labels from: {LABELS_PATH}")
        with open(LABELS_PATH, 'r') as f:
            # Assumes "0 label1", "1 label2" format
            # Store both index and label name for clarity
            labels_dict = {}
            labels_list = []
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2 and parts[0].isdigit():
                    idx = int(parts[0])
                    name = parts[1]
                    labels_dict[idx] = name
                    # Ensure list matches index order, fill gaps if any (though unlikely from TM)
                    while len(labels_list) <= idx:
                        labels_list.append("unknown_gap_label")
                    labels_list[idx] = name
                else:
                    print(f"WARNING: Skipping invalid line in labels file: {line.strip()}")

        print(f"Labels loaded (index: name): {labels_dict}")
        if not labels_list:
             print("ERROR: Label file processing failed or file is empty/invalid.")
             stop_event.set(); return

        # Basic check if number of labels matches model output units
        try:
            output_units = model.output_shape[-1]
            # Use the highest index found + 1 as the effective number of labels needed
            num_labels_expected = max(labels_dict.keys()) + 1 if labels_dict else 0
            if output_units != num_labels_expected:
                 print(f"WARNING: Model output units ({output_units}) doesn't match highest label index + 1 ({num_labels_expected}). Check labels file and model.")
        except Exception: pass # Ignore errors checking output shape

    except ValueError as e: # Error likely during label parsing
        print(f"ERROR: Could not load labels. Check format (e.g., '0 stop', '1 go'): {e}"); stop_event.set(); return
    except Exception as e: # General error for model loading etc.
        print(f"ERROR: Failed to load Keras model or labels: {e}"); traceback.print_exc(); stop_event.set(); return

    # --- Video Capture ---
    print(f"Attempting to open video stream: {VIDEO_STREAM_URL}"); cap = cv2.VideoCapture(VIDEO_STREAM_URL)
    if not cap.isOpened(): print(f"ERROR: Cannot open video stream at {VIDEO_STREAM_URL}"); stop_event.set(); return
    print("Video stream opened successfully.")
    frame_width_actual = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height_actual = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"VidProc: Actual stream dimensions {frame_width_actual}x{frame_height_actual}")

    last_queued_action = ACTION_STOP
    last_tm_prediction = ("N/A", 0.0)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None: print("VideoProc: Error reading frame."); time.sleep(0.5); continue

        processed_frame = frame.copy() # For display queue

        with tm_status_lock: is_tm_active = tm_control_enabled

        current_action = ACTION_STOP # Default to stop unless overridden
        prediction_info = ("Inactive", 0.0)

        if is_tm_active:
            # --- Preprocess Frame for Keras Model ---
            try:
                # Resize to the dimensions the model expects
                resized_frame = cv2.resize(frame, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
                # Convert to float32 *before* normalization
                input_image = np.asarray(resized_frame, dtype=np.float32)
                # Normalize (Example uses Teachable Machine's common [-1, 1] range)
                input_image = (input_image - TM_INPUT_MEAN) / TM_INPUT_STD
                # Add batch dimension
                input_data = np.expand_dims(input_image, axis=0)

                # --- Run Inference using Keras model.predict ---
                predictions = model.predict(input_data, verbose=0) # verbose=0 prevents progress bar
                scores = predictions[0] # Get probabilities from the first (and only) batch item

                # --- Interpret Results ---
                predicted_index = np.argmax(scores)
                confidence = scores[predicted_index]

                # Get label name from our loaded list/dict
                if 0 <= predicted_index < len(labels_list):
                    predicted_label_name = labels_list[predicted_index]
                else:
                    predicted_label_name = f"error_index_{predicted_index}" # Indicates label file mismatch or unexpected index
                    print(f"ERROR: Predicted index {predicted_index} out of bounds for loaded labels (len={len(labels_list)})")

                # Prepare info for display (user-friendly format)
                prediction_info = (predicted_label_name.replace("_"," ").title(), float(confidence))

                # --- Decide action based on prediction and confidence ---
                if confidence >= TM_CONFIDENCE_THRESHOLD and not predicted_label_name.startswith("error_index"):
                    # --- Map Prediction Index to Action USING labels.txt mapping ---
                    #   0 go         -> ACTION_FORWARD
                    #   1 stop       -> ACTION_STOP
                    #   2 turn_left  -> ACTION_LEFT
                    #   3 turn_right -> ACTION_RIGHT
                    #   4 do_nothing -> ACTION_STOP
                    # -----------------------------------------------------
                    if predicted_index == 0:    # go
                        current_action = ACTION_FORWARD
                    elif predicted_index == 1:  # stop
                        current_action = ACTION_STOP
                    elif predicted_index == 2:  # turn_left
                        current_action = ACTION_LEFT
                    elif predicted_index == 3:  # turn_right
                        current_action = ACTION_RIGHT
                    elif predicted_index == 4:  # do_nothing
                        current_action = ACTION_STOP # Treat 'do_nothing' as stop
                    else:
                        # Fallback for unexpected valid indices (should not happen if labels.txt is correct)
                        print(f"Warning: Index {predicted_index} ('{predicted_label_name}') above threshold but not mapped. Stopping.")
                        current_action = ACTION_STOP

                else:
                    # Confidence too low or index error -> STOP the car for safety
                    current_action = ACTION_STOP
                    if not predicted_label_name.startswith("error_index"):
                         # Display low confidence message but keep the predicted label name
                         prediction_info = (f"Low Conf ({prediction_info[0]})", prediction_info[1])
                    # If it was an index error, prediction_info already reflects that

                # --- Queue Action if Changed ---
                if current_action != last_queued_action:
                    try:
                        # Clear queue before adding new action to prioritize latest command
                        while not action_queue.empty(): action_queue.get_nowait(); action_queue.task_done()
                        action_queue.put_nowait(current_action)
                        last_queued_action = current_action
                        # Optional: Print the action being sent based on TM detection
                        # print(f"VidProc: TM detected '{predicted_label_name}' ({confidence:.2f}), queuing action: {current_action}")
                    except queue.Full:
                        print("VidProc WARN: Action queue full when trying to queue TM action.")

            except Exception as e:
                 print(f"ERROR during Keras prediction/processing: {type(e).__name__}: {e}")
                 traceback.print_exc() # Print full traceback for debugging
                 current_action = ACTION_STOP # Safety stop on error
                 prediction_info = ("Processing Error", 0.0)
                 # Attempt to queue stop if error occurs and we weren't already stopped
                 if last_queued_action != ACTION_STOP:
                     try:
                         while not action_queue.empty(): action_queue.get_nowait(); action_queue.task_done()
                         action_queue.put_nowait(ACTION_STOP)
                         last_queued_action = ACTION_STOP
                     except queue.Full: pass # Ignore if queue is full

        else: # TM Control Disabled
            # Ensure the car stops if TM is turned off and it wasn't already stopped manually
            if last_queued_action != ACTION_STOP:
                try:
                    # Clear queue and send stop
                    while not action_queue.empty(): action_queue.get_nowait(); action_queue.task_done()
                    action_queue.put_nowait(ACTION_STOP)
                    last_queued_action = ACTION_STOP
                except queue.Full:
                     print("VidProc WARN: Action queue full trying to queue STOP after TM disable.")
            prediction_info = ("Manual Mode", 0.0) # Update status for display


        # --- Update Shared Info for Display ---
        try:
            # Keep only the latest frame if processing/display lags
            while frame_queue.qsize() >= 2: frame_queue.get_nowait(); frame_queue.task_done()
            frame_queue.put_nowait(processed_frame)
        except queue.Full: pass # Don't block if queue is full
        except Exception as q_err: print(f"VideoProc: Error putting frame on queue: {q_err}")

        # Only update TM info if it changed to reduce queue traffic and UI updates
        if prediction_info != last_tm_prediction:
            try:
                # Keep only the latest info if queue is full
                while tm_info_queue.qsize() >= 2: tm_info_queue.get_nowait(); tm_info_queue.task_done()
                tm_info_queue.put_nowait(prediction_info)
                last_tm_prediction = prediction_info
            except queue.Full: pass # Don't block
            except Exception as q_err: print(f"VideoProc: Error putting TM info on queue: {q_err}")

        time.sleep(0.02) # Small sleep to prevent 100% CPU usage in this thread

    print("Video processor thread finishing..."); cap.release(); print("Video stream released.")


# --- MODIFIED Joystick Emulator Function (Minor Text Update) ---
def run_joystick_emulator():
    global action_queue, frame_queue, tm_info_queue
    global tm_control_enabled, tm_status_lock

    pygame.display.set_caption("Car Control & Stream (Keras TM / Manual)") # <-- Caption Updated
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock(); font = pygame.font.Font(None, 24); font_small = pygame.font.Font(None, 20)
    joystick_active = False; current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y]
    last_queued_action = None; is_tm_active_local = False
    latest_frame_surface = None; current_tm_info = ("Initializing...", 0.0)

    # <<< UPDATED PRINT INFO FOR KERAS >>>
    print("\n--- Control Interface (Keras TM / Manual) ---")
    print("Video Stream at Top")
    print("Joystick Control / TM Status at Bottom")
    print("T: Toggle Teachable Machine Control Mode")
    print(f"  - Keras Model: {KERAS_MODEL_PATH}") # <-- Path name updated
    print(f"  - TM Labels: {LABELS_PATH}")
    print("Manual Control: N=4 Fwd/Bwd, N=3 Turns (when TM Mode is OFF)")
    print("Close window or press Ctrl+C in terminal to Quit.")
    print("----------------------------------------------------")
    # <<< END OF UPDATE >>>


    running = True
    # --- Main Loop (Identical logic to TFLite version) ---
    while running and not stop_event.is_set():
        manual_action = ACTION_STOP # Default manual action is stop unless joystick active
        with tm_status_lock: is_tm_active_local = tm_control_enabled # Get current TM mode

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                manual_action = ACTION_QUIT; running = False # Signal quit
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t: # Toggle TM mode
                     with tm_status_lock:
                         tm_control_enabled = not tm_control_enabled
                         is_tm_active_local = tm_control_enabled # Update local copy
                     print(f"Teachable Machine Control {'ENABLED' if is_tm_active_local else 'DISABLED'}")
                     # When switching TO manual, ensure car stops if joystick isn't active
                     if not is_tm_active_local and not joystick_active:
                         manual_action = ACTION_STOP
                         try: # Prioritize stop on mode switch
                             while not action_queue.empty(): action_queue.get_nowait(); action_queue.task_done() # Clear queue
                             action_queue.put_nowait(ACTION_STOP); last_queued_action = ACTION_STOP
                         except queue.Full: pass
                     # When switching TO TM, clear last manual action and deactivate joystick
                     elif is_tm_active_local:
                         last_queued_action = None # Let TM control take over
                         joystick_active = False # Disable joystick visually/logically
                         current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y] # Reset knob

            # Handle Mouse Clicks (only if TM is OFF)
            elif event.type == pygame.MOUSEBUTTONDOWN and not is_tm_active_local:
                if event.button == 1: # Left mouse button
                    mx, my = event.pos
                    # Check if click is within the joystick control area
                    if my > STREAM_DISPLAY_HEIGHT:
                         dist_sq = (mx - JOYSTICK_CENTER_X)**2 + (my - JOYSTICK_CENTER_Y)**2
                         if dist_sq <= BOUNDARY_RADIUS**2: # Clicked inside joystick boundary
                             joystick_active = True; current_knob_pos = list(event.pos)

            # Handle Mouse Release (only if joystick was active)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and joystick_active: # Left button released while joystick was active
                    joystick_active = False
                    current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y] # Reset knob
                    manual_action = ACTION_STOP # Stop the car on release

            # Handle Mouse Motion (only if joystick is active and TM is OFF)
            elif event.type == pygame.MOUSEMOTION and joystick_active and not is_tm_active_local:
                current_knob_pos = list(event.pos) # Update knob position

        # --- Get Latest Frame and TM Info from Queues ---
        try: # Get frame
            frame_bgr = frame_queue.get_nowait(); frame_queue.task_done()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # Convert for Pygame
            # Create Pygame surface (handle potential size mismatch if needed)
            try:
                 f_surf = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
            except ValueError as img_err: # Handle potential issue if frame size is incorrect
                 print(f"Joystick WARN: Error creating image from buffer: {img_err}. Check frame dimensions.")
                 f_surf = None # Avoid crashing, display placeholder
            if f_surf:
                 latest_frame_surface = pygame.transform.scale(f_surf, (STREAM_DISPLAY_WIDTH, STREAM_DISPLAY_HEIGHT))
        except queue.Empty: pass # No new frame, just keep the old one
        except Exception as e: print(f"Joystick: Error processing frame: {type(e).__name__}: {e}")

        try: # Get TM info
            current_tm_info = tm_info_queue.get_nowait(); tm_info_queue.task_done()
        except queue.Empty: pass # No new info, keep the old one

        # --- Determine Manual Action based on Joystick (if TM is OFF) ---
        if not is_tm_active_local: # Manual mode
            if joystick_active:
                # Calculate vector from center to knob
                dx = current_knob_pos[0] - JOYSTICK_CENTER_X
                dy = current_knob_pos[1] - JOYSTICK_CENTER_Y
                dist = math.sqrt(dx**2 + dy**2)

                # Clamp knob position to boundary circle
                if dist > BOUNDARY_RADIUS:
                    scale = BOUNDARY_RADIUS / dist
                    current_knob_pos[0] = JOYSTICK_CENTER_X + dx * scale
                    current_knob_pos[1] = JOYSTICK_CENTER_Y + dy * scale
                    dist = BOUNDARY_RADIUS # Recalculate distance after clamping

                # Determine action based on angle if outside dead zone
                if dist > DEAD_ZONE_RADIUS:
                    angle = math.atan2(-dy, dx) # -dy because pygame Y is inverted
                    # Map angle to car actions (adjust ranges as needed)
                    if -math.pi * 0.25 <= angle < math.pi * 0.25: manual_action = ACTION_RIGHT
                    elif math.pi * 0.25 <= angle < math.pi * 0.75: manual_action = ACTION_FORWARD
                    elif angle >= math.pi * 0.75 or angle < -math.pi * 0.75: manual_action = ACTION_LEFT
                    elif -math.pi * 0.75 <= angle < -math.pi * 0.25: manual_action = ACTION_BACKWARD
                    else: manual_action = ACTION_STOP # Should not happen with atan2
                else: # Inside dead zone
                    manual_action = ACTION_STOP
                    current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y] # Snap to center
            else: # Joystick not active
                manual_action = ACTION_STOP
                current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y] # Ensure knob is centered visually

        else: # TM mode is active
            manual_action = ACTION_STOP # No manual action when TM is on
            current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y] # Reset knob visually
            joystick_active = False # Ensure joystick logic is inactive

        # --- Queue Manual Action (if in manual mode and action changed or is QUIT) ---
        action_to_queue = manual_action
        if not is_tm_active_local and (action_to_queue != last_queued_action or action_to_queue == ACTION_QUIT):
             try:
                 # Clear queue before putting new manual action
                 while not action_queue.empty(): action_queue.get_nowait(); action_queue.task_done()
                 action_queue.put_nowait(action_to_queue); last_queued_action = action_to_queue
             except queue.Full: print("Joystick ERROR: Queue full for manual action.")
             if action_to_queue == ACTION_QUIT: running = False # Exit loop if quit action queued

        # Reset last_queued_action when switching to TM mode so TM can take over
        elif is_tm_active_local:
             if last_queued_action is not None: last_queued_action = None

        # --- Drawing ---
        screen.fill(COLOR_WHITE) # Clear screen

        # Draw Video Stream
        if latest_frame_surface: screen.blit(latest_frame_surface, (0, 0))
        else: # Placeholder if no frame yet
            pygame.draw.rect(screen, COLOR_GRAY, (0, 0, STREAM_DISPLAY_WIDTH, STREAM_DISPLAY_HEIGHT))
            placeholder = font.render("Waiting for video...", True, COLOR_BLACK)
            screen.blit(placeholder, placeholder.get_rect(center=(STREAM_DISPLAY_WIDTH//2, STREAM_DISPLAY_HEIGHT//2)))

        # Draw Controls Area Background (optional)
        pygame.draw.rect(screen, COLOR_WHITE, (0, STREAM_DISPLAY_HEIGHT, WINDOW_WIDTH, CONTROLS_HEIGHT))

        # Draw Joystick
        joy_color = COLOR_GRAY if is_tm_active_local else COLOR_BLACK # Dim joystick if TM active
        pygame.draw.circle(screen, joy_color, (JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y), BOUNDARY_RADIUS, 2) # Boundary
        pygame.draw.circle(screen, COLOR_RED if not is_tm_active_local else COLOR_GRAY, (JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y), DEAD_ZONE_RADIUS, 1) # Deadzone
        knob_color = COLOR_BLUE if joystick_active and not is_tm_active_local else joy_color # Knob color
        knob_center_int = (int(round(current_knob_pos[0])), int(round(current_knob_pos[1]))) # Use integer coords
        pygame.draw.circle(screen, knob_color, knob_center_int, KNOB_RADIUS) # Knob

        # Draw Status Text
        mode_text = f"Mode: {'KERAS TM ACTIVE' if is_tm_active_local else 'MANUAL CONTROL'}"
        mode_color = COLOR_GREEN if is_tm_active_local else COLOR_BLUE
        tm_pred_text = f"Prediction: {current_tm_info[0]} ({current_tm_info[1]:.2f})"

        # Determine what action is currently being sent (or attempted)
        last_action_disp = '---'
        if not is_tm_active_local and last_queued_action is not None: # Manual mode, show joystick action
            last_action_disp = last_queued_action
        elif is_tm_active_local: # TM mode, show info from TM
            # Use the prediction info directly for a more dynamic status
            if "Error" in current_tm_info[0]:
                 last_action_disp = 'TM Error - Stopping'
            elif "Low Conf" in current_tm_info[0]:
                 last_action_disp = f'TM Low Conf - Stopping'
            elif current_tm_info[0] == "Inactive" or current_tm_info[0] == "Manual Mode":
                 last_action_disp = 'TM Idle - Stopping'
            else:
                 # Map the displayed prediction back to the intended action for clarity
                 label_name_lower = current_tm_info[0].lower().replace(" ", "_")
                 action_from_tm = "STOP" # Default
                 if label_name_lower == "go": action_from_tm = "FORWARD"
                 elif label_name_lower == "turn_left": action_from_tm = "LEFT"
                 elif label_name_lower == "turn_right": action_from_tm = "RIGHT"
                 # stop and do_nothing map to STOP implicitly or handled above
                 last_action_disp = f'TM Control: {action_from_tm} ({current_tm_info[0]})'

        last_action_text = f"Car Action: {last_action_disp}"

        instruction_text = ["T: Toggle Mode", mode_text, tm_pred_text, last_action_text]
        y_offset = STREAM_DISPLAY_HEIGHT + 20 # Start text lower down in controls area
        for i, line in enumerate(instruction_text):
             color = mode_color if "Mode:" in line else COLOR_BLACK # Color mode text
             if "Prediction:" in line: # Color prediction based on state
                 if is_tm_active_local:
                      if "Error" in line or "Low Conf" in line or "Unknown" in line or "Index" in line:
                           color = COLOR_ORANGE # Warning color
                      else:
                           color = COLOR_GREEN # Good prediction color
                 else: # Dim prediction when manual
                      color = COLOR_GRAY
             if "Action:" in line: # Keep action text black (or customize)
                 color = COLOR_BLACK

             # Use smaller font for the potentially longer prediction/action lines
             current_font = font_small if i >= 2 else font
             text_surface = current_font.render(line, True, color)
             # Center text horizontally
             text_rect = text_surface.get_rect(midtop=(WINDOW_WIDTH // 2, y_offset + i * 25)) # Increased spacing
             screen.blit(text_surface, text_rect)

        pygame.display.flip(); clock.tick(30) # Limit FPS to ~30

    print("Joystick emulator loop finished.")
    # Ensure QUIT is queued if loop exited for other reasons (e.g., stop_event set by another thread)
    if last_queued_action != ACTION_QUIT:
        print("Joystick: Loop exited unexpectedly. Queuing final QUIT.")
        try:
             while not action_queue.empty(): action_queue.get_nowait(); action_queue.task_done()
             action_queue.put_nowait(ACTION_QUIT)
        except queue.Full: print("Joystick WARN: Could not queue final QUIT action.")
    stop_event.set(); print("Joystick thread signalling stop.")


# --- Main Execution (Minor Text Update) ---
if __name__ == "__main__":
    # <<< UPDATED PRINT INFO FOR KERAS >>>
    print("--- Elegoo Car Controller w/ Keras TM & Video ---")
    print(f"Target: {CAR_IP}:{CAR_PORT} | Video: {VIDEO_STREAM_URL}")
    print(f"Keras Model: {KERAS_MODEL_PATH} | Labels: {LABELS_PATH}")
    now_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {now_time}")
    print("----------------------------------------------------")
    print("Control Scheme:")
    print(" - Toggle 'T' for Teachable Machine (Keras) / Manual Mode.")
    print(" - TM Mode uses model prediction based on labels.txt mapping.")
    print(" - Manual Mode uses joystick (Click/drag in lower area).")
    print("----------------------------------------------------")
    # <<< END OF UPDATE >>>


    car_socket = None; sender_thread = None; joystick_thread = None; video_thread = None
    # Put initial stop action in the queue
    try: action_queue.put_nowait(ACTION_STOP)
    except queue.Full: print("WARN: Could not put initial STOP action on queue.")

    try:
        car_socket = connect_to_car(CAR_IP, CAR_PORT)
        if car_socket:
            print("Connection established. Starting threads...")
            stop_event.clear() # Ensure event is clear before starting threads

            # Start threads (daemon=True allows main thread to exit even if these are running,
            # but we join them explicitly later for cleaner shutdown)
            sender_thread = threading.Thread(target=command_sender, args=(car_socket,), name="CommandSenderThread", daemon=True); sender_thread.start()
            video_thread = threading.Thread(target=video_processor, name="VideoProcessorThread", daemon=True); video_thread.start()

            # Joystick thread runs the main interactive loop, not daemon, so we can join it
            joystick_thread = threading.Thread(target=run_joystick_emulator, name="JoystickThread"); joystick_thread.start()

            # Wait for the joystick thread to finish (means user closed window or quit)
            joystick_thread.join()
            print("Main: Joystick thread finished.")

        else:
            print("Failed initial connection. Exiting.")
            pygame.quit() # Quit pygame if connection failed
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nMain: KeyboardInterrupt received.")
        stop_event.set() # Signal threads to stop
    except Exception as e:
        print(f"\nMain Thread Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        stop_event.set() # Signal threads to stop on error
    finally:
        print("Main: Starting Cleanup...")
        stop_event.set() # Ensure stop is signaled regardless of exit reason

        # Wait briefly for threads to attempt graceful shutdown
        if joystick_thread and joystick_thread.is_alive():
            print("Main: Waiting for joystick thread...")
            joystick_thread.join(timeout=1.0) # Short timeout as it should exit quickly after stop_event
        if video_thread and video_thread.is_alive():
            print("Main: Waiting for video thread...")
            video_thread.join(timeout=2.0) # Give video slightly longer (e.g., to release capture)
        if sender_thread and sender_thread.is_alive():
            print("Main: Waiting for sender thread...")
            sender_thread.join(timeout=2.0) # Give sender time for final commands/heartbeat checks

        # Final check if threads are still running (force exit scenario)
        if joystick_thread and joystick_thread.is_alive(): print("WARN: Joystick thread did not exit cleanly.")
        if video_thread and video_thread.is_alive(): print("WARN: Video thread did not exit cleanly.")
        if sender_thread and sender_thread.is_alive(): print("WARN: Sender thread did not exit cleanly.")


        # Clean up socket connection
        if car_socket:
             print("Main: Attempting final STOP (N=4) command before closing socket.")
             # Use a final direct send attempt, ignoring errors if already closed/broken
             try:
                 # Ensure stop command is sent reliably even if sender thread died
                 with socket_lock:
                     car_socket.sendall(CMD_STOP.encode('utf-8'))
             except Exception as final_send_err:
                 print(f"Main: Error sending final stop command (socket likely closed): {final_send_err}")

             time.sleep(0.1) # Small delay for command processing on car side
             print("Main: Closing socket.")
             with socket_lock: # Use lock for final socket operations
                  try: car_socket.shutdown(socket.SHUT_RDWR) # Graceful shutdown if possible
                  except (OSError, socket.error): pass # Ignore errors if already closed/broken
                  finally:
                      try: car_socket.close()
                      except: pass # Ignore errors on final close
                      print("Main: Socket closed.")

        # Quit Pygame if it was initialized
        if pygame.get_init():
            print("Main: Quitting Pygame.")
            pygame.quit()

        print("Main: Program finished.")