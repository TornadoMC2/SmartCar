import socket
import time
import sys
import json
import threading
import queue # Using queue for potentially smoother state updates

# --- Configuration ---
CAR_IP = "192.168.4.1"
CAR_PORT = 100
COMMAND_ID = "Elegoo"
SEND_INTERVAL = 0.05 # Send commands roughly 20 times per second (adjust as needed)

# --- Control Parameters ---
DEFAULT_SPEED = 255
MOTOR_ALL = 0
MOTOR_LEFT = 1
MOTOR_RIGHT = 2
ROTATION_CW = 1  # Clockwise (Forward for whole car, or specific wheel direction)
ROTATION_CCW = 2 # Counter-Clockwise (Backward for whole car)

# --- Action States ---
ACTION_STOP = 'STOP'
ACTION_FORWARD = 'FORWARD'
ACTION_BACKWARD = 'BACKWARD'
ACTION_LEFT = 'LEFT'
ACTION_RIGHT = 'RIGHT'
ACTION_QUIT = 'QUIT' # Special action to signal exit

# --- Shared State ---
# Use a queue for thread-safe communication of the desired action
# Maxsize=1 ensures the sender always gets the *latest* command quickly
action_queue = queue.Queue(maxsize=1)
# Lock for protecting socket access if needed (sendall might be atomic enough, but safer)
socket_lock = threading.Lock()
stop_event = threading.Event() # To signal threads to terminate

# --- JSON Command Generation ---
def create_command_json(motor_selection, speed, rotation_direction):
    """Creates the JSON command string."""
    command_dict = {
        "H": COMMAND_ID, "N": 1, "D1": int(motor_selection),
        "D2": int(speed), "D3": int(rotation_direction)
    }
    json_string = json.dumps(command_dict) + "\n"
    return json_string

# Pre-generate command strings for efficiency
CMD_STOP = create_command_json(MOTOR_ALL, 0, ROTATION_CW)
CMD_FORWARD = create_command_json(MOTOR_ALL, DEFAULT_SPEED, ROTATION_CW)
CMD_BACKWARD = create_command_json(MOTOR_ALL, DEFAULT_SPEED, ROTATION_CCW)
# Turn Left: Left motor backwards (CCW), Right motor forwards (CW)
CMD_TURN_LEFT_1 = create_command_json(MOTOR_LEFT, DEFAULT_SPEED, ROTATION_CCW)
CMD_TURN_LEFT_2 = create_command_json(MOTOR_RIGHT, DEFAULT_SPEED, ROTATION_CW)
# Turn Right: Left motor forwards (CW), Right motor backwards (CCW)
CMD_TURN_RIGHT_1 = create_command_json(MOTOR_LEFT, DEFAULT_SPEED, ROTATION_CW)
CMD_TURN_RIGHT_2 = create_command_json(MOTOR_RIGHT, DEFAULT_SPEED, ROTATION_CCW)


# --- Connection Function (Same as before) ---
def connect_to_car(ip, port):
    """Attempts to establish a persistent connection."""
    print(f"Attempting persistent connection to {ip}:{port}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Optional: Set TCP keepalive options (OS level)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 3)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
        except (AttributeError, OSError):
             print("Warning: TCP Keepalive options not fully available/settable on this platform.")

        sock.settimeout(5.0) # Initial connection timeout
        sock.connect((ip, port))
        sock.settimeout(2.0) # Shorter timeout for operations
        print("Connection successful.")
        return sock
    except socket.timeout:
        print(f"Error: Connection timed out connecting to {ip}:{port}.")
        return None
    except socket.error as e:
        print(f"Error connecting to {ip}:{port}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during connection: {e}")
        return None

# --- Send Command Function ---
def send_json_command(sock, json_cmd_string, thread_name=""):
    """Encodes and sends the JSON command string via the persistent socket."""
    if not sock or stop_event.is_set(): # Don't send if stopping or no socket
        return False
    with socket_lock: # Protect socket access
        # print(f"Sending ({thread_name}): {json_cmd_string.strip()}") # Optional: Verbose logging
        try:
            sock.sendall(json_cmd_string.encode('utf-8'))
            return True
        except socket.timeout:
            print(f"Error ({thread_name}): Send operation timed out. Connection may be lost.")
            stop_event.set() # Signal termination on error
            return False
        except socket.error as e:
            print(f"Socket Error ({thread_name}): {e}. Connection lost.")
            stop_event.set() # Signal termination on error
            return False
        except Exception as e:
            print(f"An unexpected error occurred during send ({thread_name}): {e}")
            stop_event.set() # Signal termination on error
            return False

# --- Sending Thread Function ---
def command_sender(sock):
    """Continuously sends commands based on the latest action from the queue."""
    print("Command sender thread started.")
    current_action = ACTION_STOP # Start in stopped state

    while not stop_event.is_set():
        # Check for a new action from the input thread (non-blocking)
        try:
            new_action = action_queue.get_nowait()
            current_action = new_action
            print(f"Sender received new action: {current_action}") # Debug
            action_queue.task_done() # Mark task as processed
        except queue.Empty:
            # No new action, continue with the current one
            pass

        if current_action == ACTION_QUIT:
            print("Sender received QUIT action. Stopping.")
            # Send final stop command
            send_json_command(sock, CMD_STOP, "Sender")
            time.sleep(0.1) # Allow time for command to send
            stop_event.set() # Ensure event is set
            break

        commands_to_send = []
        action_desc = ""

        # Determine commands based on current action state
        if current_action == ACTION_STOP:
            commands_to_send = [CMD_STOP]
            action_desc = "STOP"
        elif current_action == ACTION_FORWARD:
            commands_to_send = [CMD_FORWARD]
            action_desc = "FORWARD"
        elif current_action == ACTION_BACKWARD:
            commands_to_send = [CMD_BACKWARD]
            action_desc = "BACKWARD"
        elif current_action == ACTION_LEFT:
            commands_to_send = [CMD_TURN_LEFT_1, CMD_TURN_LEFT_2]
            action_desc = "LEFT"
        elif current_action == ACTION_RIGHT:
            commands_to_send = [CMD_TURN_RIGHT_1, CMD_TURN_RIGHT_2]
            action_desc = "RIGHT"
        else: # Should not happen, but default to STOP
             commands_to_send = [CMD_STOP]
             action_desc = "UNKNOWN->STOP"


        # print(f"Sender sending: {action_desc}") # Optional: Verbose logging

        # Send the command(s) for the current action
        send_success = True
        for cmd in commands_to_send:
            if not send_json_command(sock, cmd, "Sender"):
                send_success = False
                break # Stop sending if one fails

        if not send_success:
             print("Sender thread breaking due to send failure.")
             break # Exit loop if sending failed

        # Wait for the defined interval, but check stop_event frequently
        # This makes the loop responsive to the stop signal
        stop_event.wait(timeout=SEND_INTERVAL)

    print("Command sender thread finished.")

# --- Input Thread Function ---
def user_input_handler():
    """Handles user input and puts the desired action onto the queue."""
    print("\n--- Continuous Car Control ---")
    print(f"Using Speed: {DEFAULT_SPEED}")
    print("Commands: F (Fwd), B (Back), L (Left), R (Right), S (Stop), Q (Quit)")
    print("Press a key to start/change action. Press S to stop.")
    print("--------------------")

    last_action = None

    while not stop_event.is_set():
        try:
            user_input = input("Enter command: ").strip().upper()
            new_action = None

            if not user_input:
                continue

            cmd_key = user_input[0]

            if cmd_key == 'Q':
                print("Input: Quit requested.")
                new_action = ACTION_QUIT
            elif cmd_key == 'S':
                new_action = ACTION_STOP
            elif cmd_key == 'F':
                new_action = ACTION_FORWARD
            elif cmd_key == 'B':
                new_action = ACTION_BACKWARD
            elif cmd_key == 'L':
                new_action = ACTION_LEFT
            elif cmd_key == 'R':
                new_action = ACTION_RIGHT
            else:
                print(f"Invalid command '{user_input}'. Use F, B, L, R, S, Q.")
                continue

            if new_action != last_action:
                 # Try to put the new action in the queue.
                 # If the queue is full (because the sender hasn't processed the last one),
                 # remove the old one and put the new one. This prioritizes the latest input.
                try:
                    action_queue.put_nowait(new_action)
                except queue.Full:
                    try:
                        _ = action_queue.get_nowait() # Remove the old item
                        action_queue.task_done()
                    except queue.Empty:
                        pass # Should not happen if queue was full, but handle anyway
                    try:
                        action_queue.put_nowait(new_action) # Add the new item
                    except queue.Full:
                         print("ERROR: Queue still full after trying to clear. This shouldn't happen.")


                last_action = new_action
                print(f"Input: Action '{new_action}' sent to sender.") # Debug

                if new_action == ACTION_QUIT:
                    print("Input: QUIT action sent, stopping input loop.")
                    stop_event.set() # Signal all threads to stop
                    break


        except (KeyboardInterrupt, EOFError):
            print("\nInput: Ctrl+C or EOF detected. Stopping.")
            # Try to send a final STOP action, then QUIT
            try:
                _ = action_queue.get_nowait()
                action_queue.task_done()
            except queue.Empty: pass
            try:
                action_queue.put_nowait(ACTION_STOP)
            except queue.Full: pass # Skip if full
            time.sleep(0.1) # Give sender a moment to process STOP
            try:
                 _ = action_queue.get_nowait()
                 action_queue.task_done()
            except queue.Empty: pass
            try:
                 action_queue.put_nowait(ACTION_QUIT)
            except queue.Full: pass # Skip if full

            stop_event.set() # Ensure stop event is set
            break # Exit input loop

    print("User input handler thread finished.")


# --- Script Entry Point ---
if __name__ == "__main__":
    print("--- Elegoo Smart Car Wi-Fi Controller (Continuous Mode) ---")
    print(f"Target: {CAR_IP}:{CAR_PORT}")
    print(f"Send Interval: {SEND_INTERVAL * 1000:.0f} ms")
    now = time.strftime("%A, %B %d, %Y at %I:%M:%S %p")
    print(f"Current time: {now}")
    print("Make sure computer is connected to the car's Wi-Fi network.")

    car_socket = None
    sender_thread = None
    input_thread = None

    # Put initial STOP action in the queue
    try:
        action_queue.put_nowait(ACTION_STOP)
    except queue.Full: pass # Should not be full initially

    try:
        car_socket = connect_to_car(CAR_IP, CAR_PORT)

        if car_socket:
            # Create and start the sender thread
            sender_thread = threading.Thread(
                target=command_sender,
                args=(car_socket,),
                name="CommandSenderThread",
                daemon=True # Allows main program to exit even if this thread hangs (though cleanup is better)
            )
            sender_thread.start()

            # Create and start the input thread
            input_thread = threading.Thread(
                target=user_input_handler,
                name="UserInputThread"
                # Not daemon, we want to wait for user to quit normally
            )
            input_thread.start()

            # Keep the main thread alive until the input thread finishes
            # (either by user quitting or error)
            input_thread.join()
            print("Input thread has joined.")

            # If input thread finished, ensure stop event is set for sender thread
            if not stop_event.is_set():
                print("Main: Setting stop event after input thread joined.")
                stop_event.set()
                 # Optionally send QUIT action if not already sent
                try:
                    action_queue.put_nowait(ACTION_QUIT)
                except queue.Full: pass


        else:
            print("Failed to establish initial connection. Exiting.")
            sys.exit(1)

    except Exception as e:
        print(f"\nAn unexpected error occurred in the main script: {e}")
        stop_event.set() # Signal threads to stop on main error
    finally:
        print("Main script entering finally block...")
        if not stop_event.is_set():
             print("Setting stop event in finally block...")
             stop_event.set() # Ensure stop event is set

        # Wait briefly for the sender thread to finish
        if sender_thread and sender_thread.is_alive():
            print("Waiting briefly for sender thread to exit...")
            sender_thread.join(timeout=1.5) # Wait a bit longer

        if car_socket:
            print("Closing connection.")
            with socket_lock: # Use lock for final socket operations
                try:
                    # Attempt graceful shutdown
                    car_socket.shutdown(socket.SHUT_RDWR)
                except (OSError, socket.error) as sd_err:
                     # Ignore errors if socket is already closed/broken
                     # print(f"Note: Error during socket shutdown (ignorable): {sd_err}")
                     pass
                finally:
                    car_socket.close() # Close the socket definitively
        print("Program finished.")