# --- (Imports and Setup from v7.1) ---
import socket
import time
import sys
import json
import threading
import queue
import math
import traceback
try:
    import pygame
except ImportError:
    print("ERROR: Pygame not found. pip install pygame")
    sys.exit(1)

# --- Configuration and Constants ---
CAR_IP = "192.168.4.1"
CAR_PORT = 100
COMMAND_ID = "Elegoo"
SEND_INTERVAL = 0.1 # How often to check queue/send commands
RECEIVE_BUFFER_SIZE = 1024
HEARTBEAT_INTERVAL = 1.0 # Send heartbeat reply every 1 second

# Pygame Window & Joystick Params
WINDOW_WIDTH = 300
WINDOW_HEIGHT = 350
JOYSTICK_CENTER_X = WINDOW_WIDTH // 2
JOYSTICK_CENTER_Y = WINDOW_HEIGHT // 2 - 20
BOUNDARY_RADIUS = 100
KNOB_RADIUS = 25
DEAD_ZONE_RADIUS = 15

# Colors
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (150, 150, 150)
COLOR_BLUE = (100, 100, 255)
COLOR_RED = (255, 100, 100)

# Car Control Params (Using N=102 simple directions)
DEFAULT_SPEED = 255 # Speed for N=102 commands (0-255)
MOTOR_ALL = 0       # For N=1 STOP command
ROTATION_CW = 1     # For N=1 STOP command

# Direction Constants for N=102 command (D1 parameter)
DIRECTION_FORWARD = 1
DIRECTION_BACKWARD = 2
DIRECTION_LEFT = 3
DIRECTION_RIGHT = 4

# --- Action Definitions (Based on Joystick Emulator Output) ---
ACTION_STOP = 'STOP'
ACTION_FORWARD = 'FORWARD'
ACTION_BACKWARD = 'BACKWARD'
ACTION_LEFT = 'LEFT'
ACTION_RIGHT = 'RIGHT'
ACTION_QUIT = 'QUIT'

# --- Shared State ---
action_queue = queue.Queue(maxsize=5)
socket_lock = threading.Lock()
stop_event = threading.Event()

# --- JSON Command Creation Functions ---
def create_stop_command_json():
    """Creates the JSON command for stopping using N=1."""
    # {"H": ID, "N": 1, "D1": 0, "D2": 0, "D3": 1}
    command_dict = {"H": COMMAND_ID, "N": 1, "D1": MOTOR_ALL, "D2": 0, "D3": ROTATION_CW}
    return json.dumps(command_dict) + "\n"

def create_joystick_command_json(direction, speed):
    """Creates the JSON command for simple joystick movement (N=102)."""
    # {"H": ID, "N": 102, "D1": direction(1-4), "D2": speed(0-255)}
    speed = max(0, min(255, int(speed))) # Ensure speed is valid
    command_dict = {"H": COMMAND_ID, "N": 102, "D1": int(direction), "D2": speed}
    return json.dumps(command_dict) + "\n"

# --- Pre-defined Commands ---
CMD_STOP = create_stop_command_json()
CMD_FORWARD = create_joystick_command_json(DIRECTION_FORWARD, DEFAULT_SPEED)
CMD_BACKWARD = create_joystick_command_json(DIRECTION_BACKWARD, DEFAULT_SPEED)
CMD_LEFT = create_joystick_command_json(DIRECTION_LEFT, DEFAULT_SPEED)
CMD_RIGHT = create_joystick_command_json(DIRECTION_RIGHT, DEFAULT_SPEED)
CMD_HEARTBEAT = "{Heartbeat}\n" # Heartbeat reply string

# --- Connection Function ---
def connect_to_car(ip, port):
    print(f"Attempting persistent connection to {ip}:{port}...")
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0); sock.connect((ip, port)); sock.settimeout(1.0); print("Connection successful.")
        try: sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1); print("SO_KEEPALIVE enabled.")
        except (AttributeError, OSError) as e: print(f"Note: SO_KEEPALIVE: {e}")
        try: sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); print("TCP_NODELAY enabled.")
        except (AttributeError, OSError) as e: print(f"Note: TCP_NODELAY: {e}")
        return sock
    except socket.timeout: print(f"Error: Connection timeout."); sock.close(); return None
    except socket.error as e: print(f"Error connecting: {e}"); sock.close(); return None
    except Exception as e: print(f"Unexpected connection error: {e}"); sock.close(); return None

# --- Send/Receive Function ---
def send_json_command(sock, json_cmd_string, thread_name=""):
    """Sends a command and attempts a non-blocking read."""
    if not sock or stop_event.is_set(): return False
    with socket_lock:
        if stop_event.is_set(): return False # Double check inside lock
        try:
            # 1. Send
            # print(f"DEBUG ({thread_name}): Sending: {json_cmd_string.strip()}") # Verbose
            sock.sendall(json_cmd_string.encode('utf-8'))

            # 2. Attempt Non-Blocking Read
            try:
                sock.setblocking(False)
                data = sock.recv(RECEIVE_BUFFER_SIZE)
                # if data: print(f"DEBUG ({thread_name}): Received: {data.decode('utf-8', errors='ignore').strip()}") # Verbose
            except BlockingIOError: pass # Expected if no data
            except socket.timeout: pass # Shouldn't happen in non-blocking, but ignore
            except (ConnectionAbortedError, OSError) as e:
                print(f"FATAL ({thread_name}): Error during non-blocking read: {e}")
                traceback.print_exc()
                stop_event.set()
                return False
            except Exception as e:
                 print(f"FATAL ({thread_name}): Unexpected Exception during non-blocking read: {type(e).__name__}: {e}")
                 traceback.print_exc()
                 stop_event.set()
                 return False
            finally:
                try: # Set back to blocking
                    if sock.fileno() != -1: sock.setblocking(True)
                except socket.error: pass # Ignore if socket already closed
            return True # Send successful

        except socket.timeout:
            print(f"Error ({thread_name}): Send operation timed out."); stop_event.set(); return False
        except (ConnectionAbortedError, OSError) as e:
            print(f"FATAL ({thread_name}): Error during send: {e}"); traceback.print_exc(); stop_event.set(); return False
        except Exception as e:
            print(f"FATAL ({thread_name}): Unexpected Exception during send: {type(e).__name__}: {e}"); traceback.print_exc(); stop_event.set(); return False


# --- Sending Thread Function (Sends movement based on queue + heartbeat) ---
def command_sender(sock):
    print("Command sender thread started (v7.2 - Sending Heartbeat Replies).")
    current_action = ACTION_STOP # Tracks the active movement command
    last_action_sent = None     # Tracks the last command actually sent
    last_heartbeat_sent_time = time.time() # Track when we last sent a heartbeat reply

    while not stop_event.is_set():
        now = time.time()
        new_action_received = False
        newest_action = None

        # --- Get latest action from queue ---
        try:
            # Drain queue to get the most recent action
            while True:
                newest_action = action_queue.get_nowait()
                new_action_received = True
                action_queue.task_done()
        except queue.Empty:
            # If we got a new action, update the current_action
            if new_action_received and newest_action != current_action:
                 current_action = newest_action
            pass # Keep current_action if queue was empty or action was the same

        # --- Handle QUIT action ---
        if current_action == ACTION_QUIT:
            print("Sender: QUIT action. Sending final STOP.")
            send_json_command(sock, CMD_STOP, "Sender-Quit")
            time.sleep(0.1); stop_event.set(); break

        # --- Determine Control Command ---
        command_to_send = None
        is_stop_command = False
        if current_action == ACTION_STOP: command_to_send, is_stop_command = CMD_STOP, True
        elif current_action == ACTION_FORWARD: command_to_send = CMD_FORWARD
        elif current_action == ACTION_BACKWARD: command_to_send = CMD_BACKWARD
        elif current_action == ACTION_LEFT: command_to_send = CMD_LEFT
        elif current_action == ACTION_RIGHT: command_to_send = CMD_RIGHT
        else: # Unknown action, default to STOP
            print(f"Warning: Unknown action '{current_action}'. Sending STOP.")
            command_to_send, current_action, is_stop_command = CMD_STOP, ACTION_STOP, True

        # --- Send Control Command (if action changed or it's STOP) ---
        # Send if the action we *intend* to send is different from the last one *actually* sent
        if command_to_send is not None and (current_action != last_action_sent or is_stop_command):
             if not send_json_command(sock, command_to_send, "Sender"):
                 print("DEBUG (Sender): Send/Read failure on control command, breaking loop.")
                 stop_event.set() # Signal exit on critical failure
                 break
             else:
                 last_action_sent = current_action # Update only on successful send

        # --- Send Heartbeat Reply (if needed) ---
        if now - last_heartbeat_sent_time >= HEARTBEAT_INTERVAL:
            # print("DEBUG (Sender): Sending Heartbeat Reply.") # Verbose
            if not send_json_command(sock, CMD_HEARTBEAT, "Sender-HB"):
                 print("DEBUG (Sender): Send/Read failure on heartbeat reply, breaking loop.")
                 stop_event.set() # Signal exit on critical failure
                 break
            else:
                 last_heartbeat_sent_time = now # Update time only on successful send

        # --- Wait efficiently ---
        # Calculate time until next heartbeat is due, or use SEND_INTERVAL as max wait
        next_heartbeat_time = last_heartbeat_sent_time + HEARTBEAT_INTERVAL
        wait_time = max(0.01, min(next_heartbeat_time - now, SEND_INTERVAL))

        stop_event.wait(timeout=wait_time) # Wait for interval or until stop signal

    print("Command sender thread finished.")


# --- Joystick Emulator Function (Generates 4-direction actions) ---
def run_joystick_emulator():
    """Runs the Pygame event loop for joystick emulation."""
    global action_queue # Ensure we modify the global queue

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Car Joystick")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24) # Default font

    joystick_active = False
    current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y]
    last_queued_action = None # Track the last action put *into* the queue by joystick

    print("\n--- Joystick Emulator Control ---")
    print("Click and drag inside the circle. Release to stop.")
    print("Close the Pygame window to Quit.")
    print("---------------------------------")

    running = True
    while running and not stop_event.is_set():
        current_determined_action = ACTION_STOP # Default action for the loop iteration

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Joystick: Quit requested via window close.")
                current_determined_action = ACTION_QUIT
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left mouse button
                    mouse_x, mouse_y = event.pos
                    dist_sq = (mouse_x - JOYSTICK_CENTER_X)**2 + (mouse_y - JOYSTICK_CENTER_Y)**2
                    if dist_sq <= BOUNDARY_RADIUS**2: joystick_active = True; current_knob_pos = list(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: # Left mouse button
                    if joystick_active: joystick_active = False; current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y]; current_determined_action = ACTION_STOP
            elif event.type == pygame.MOUSEMOTION:
                if joystick_active: current_knob_pos = list(event.pos)

        # --- Determine Action based on Knob Position (4 directions) ---
        if joystick_active:
            dx = current_knob_pos[0] - JOYSTICK_CENTER_X; dy = current_knob_pos[1] - JOYSTICK_CENTER_Y; dist = math.sqrt(dx**2 + dy**2)
            # Clamp knob to boundary
            if dist > BOUNDARY_RADIUS: scale = BOUNDARY_RADIUS / dist; current_knob_pos[0] = JOYSTICK_CENTER_X + dx * scale; current_knob_pos[1] = JOYSTICK_CENTER_Y + dy * scale; dist = BOUNDARY_RADIUS
            # Check if outside deadzone
            if dist > DEAD_ZONE_RADIUS:
                angle = math.atan2(-dy, dx) # Standard angle (Y inverted for screen coords)
                # Determine 4 cardinal directions
                if -math.pi * 0.25 <= angle < math.pi * 0.25: current_determined_action = ACTION_RIGHT
                elif math.pi * 0.25 <= angle < math.pi * 0.75: current_determined_action = ACTION_FORWARD
                elif angle >= math.pi * 0.75 or angle < -math.pi * 0.75: current_determined_action = ACTION_LEFT
                elif -math.pi * 0.75 <= angle < -math.pi * 0.25: current_determined_action = ACTION_BACKWARD
                else: current_determined_action = ACTION_STOP # Should not happen
            else: # Inside dead zone
                current_determined_action = ACTION_STOP
                current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y] # Snap to center
        else: # Joystick not active
             current_determined_action = ACTION_STOP
             current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y]

        # --- Update Action Queue (if action changed) ---
        if current_determined_action != last_queued_action:
             try:
                 # Clear queue and add new action (ensures responsiveness)
                 while not action_queue.empty(): action_queue.get_nowait(); action_queue.task_done()
                 action_queue.put_nowait(current_determined_action)
                 last_queued_action = current_determined_action
                 # print(f"DEBUG (Joystick): Queued {current_determined_action}") # Verbose
             except queue.Full:
                 print("Joystick ERROR: Queue full even after clearing.")
             # If quit action determined, stop the loop
             if current_determined_action == ACTION_QUIT:
                 print("Joystick: Queued QUIT action.")
                 running = False # Stop the loop after queueing

        # --- Drawing ---
        screen.fill(COLOR_WHITE)
        # Draw joystick viz
        pygame.draw.circle(screen, COLOR_GRAY, (JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y), BOUNDARY_RADIUS, 2)
        pygame.draw.circle(screen, COLOR_RED, (JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y), DEAD_ZONE_RADIUS, 1)
        knob_color = COLOR_BLUE if joystick_active else COLOR_BLACK
        knob_center_int = (int(round(current_knob_pos[0])), int(round(current_knob_pos[1])))
        pygame.draw.circle(screen, knob_color, knob_center_int, KNOB_RADIUS)
        # Draw status text
        status_text = f"Action: {last_queued_action if last_queued_action is not None else 'INIT'}"
        instruction_text = ["Click & Drag in Circle", "Release to Stop", status_text]
        y_offset = WINDOW_HEIGHT - 75
        for i, line in enumerate(instruction_text):
            text_surface = font.render(line, True, COLOR_BLACK)
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, y_offset + i * 25))
            screen.blit(text_surface, text_rect)

        pygame.display.flip()
        clock.tick(30) # Frame rate limit

    # --- Joystick Cleanup ---
    print("Joystick emulator loop finished.")
    # Ensure QUIT action is queued if loop exited for other reasons
    if last_queued_action != ACTION_QUIT:
        print("Joystick: Loop exited without QUIT. Queuing final QUIT.")
        try:
             while not action_queue.empty(): action_queue.get_nowait(); action_queue.task_done()
             action_queue.put_nowait(ACTION_QUIT)
        except queue.Full:
             print("Joystick WARN: Could not queue final QUIT action on cleanup.")
    stop_event.set(); pygame.quit(); print("Pygame quit.")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Elegoo Smart Car Wi-Fi Controller (v7.2 - Heartbeat Reply) ---")
    print(f"Target: {CAR_IP}:{CAR_PORT}")
    print(f"Send Interval: {SEND_INTERVAL * 1000:.0f} ms | Heartbeat Reply Interval: {HEARTBEAT_INTERVAL * 1000:.0f} ms")
    now = time.strftime("%Y-%m-%d %H:%M:%S"); print(f"Start time: {now}")
    print("Sending {Heartbeat} reply periodically.")
    print("----------------------------------------------------")

    car_socket = None; sender_thread = None; joystick_thread = None
    try: action_queue.put_nowait(ACTION_STOP) # Start in stopped state
    except queue.Full: print("WARN: Could not put initial STOP action.")

    try:
        car_socket = connect_to_car(CAR_IP, CAR_PORT)
        if car_socket:
            print("Connection established. Starting threads..."); stop_event.clear()
            # Start sender thread (daemon so it exits if main thread exits)
            sender_thread = threading.Thread(target=command_sender, args=(car_socket,), name="CommandSenderThread", daemon=True)
            sender_thread.start()
            # Start joystick thread (non-daemon, main thread waits for it)
            joystick_thread = threading.Thread(target=run_joystick_emulator, name="JoystickThread")
            joystick_thread.start()
            joystick_thread.join() # Wait here until joystick window is closed
            print("Main: Joystick thread finished.")
        else:
            print("Failed initial connection. Exiting."); sys.exit(1)
    except KeyboardInterrupt:
        print("\nMain: KeyboardInterrupt."); stop_event.set()
    except Exception as e:
        print(f"\nMain Exception: {e}"); traceback.print_exc(); stop_event.set()
    finally:
        # --- Cleanup ---
        print("Main: Cleanup..."); stop_event.set() # Signal threads to stop

        # Join threads with timeouts
        if joystick_thread and joystick_thread.is_alive():
             print("Main: Waiting for joystick thread...")
             joystick_thread.join(timeout=1.0)
        if sender_thread and sender_thread.is_alive():
             print("Main: Waiting for sender thread...")
             sender_thread.join(timeout=2.0) # Give sender time for final commands/heartbeat

        # Close socket
        if car_socket:
             print("Main: Closing socket.")
             with socket_lock: # Ensure sender isn't using socket during close
                  try: car_socket.shutdown(socket.SHUT_RDWR) # Attempt graceful shutdown
                  except (OSError, socket.error): pass # Ignore errors if already closed
                  finally: car_socket.close(); print("Main: Socket closed.")

        # Ensure Pygame quits
        if pygame.get_init(): pygame.quit()
        print("Main: Program finished.")