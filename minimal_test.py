import socket
import time
import sys
import json
import threading
import queue
import math

# --- Configuration ---
CAR_IP = "192.168.4.1"
CAR_PORT = 100
COMMAND_ID = "Elegoo"
READ_INTERVAL = 0.1
READ_TIMEOUT = 0.05
RECEIVE_BUFFER_SIZE = 1024
INITIAL_READ_DELAY = 0.1 # <<< Added Delay

# --- Shared State ---
socket_lock = threading.Lock()
stop_event = threading.Event()

# --- JSON Command Generation ---
def create_command_json(motor_selection, speed, rotation_direction):
    command_dict = {
        "H": COMMAND_ID, "N": 1, "D1": int(motor_selection),
        "D2": int(speed), "D3": int(rotation_direction)
    }
    json_string = json.dumps(command_dict) + "\n"
    return json_string

CMD_STOP = create_command_json(0, 0, 1)

# --- Connection Function (Unchanged) ---
def connect_to_car(ip, port):
    print(f"Attempting persistent connection to {ip}:{port}...")
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((ip, port))
        sock.settimeout(READ_TIMEOUT) # Set short timeout for reads
        print("Connection successful.")
        try: sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1); print("SO_KEEPALIVE enabled.")
        except (AttributeError, OSError) as e: print(f"Note: SO_KEEPALIVE: {e}")
        try: sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); print("TCP_NODELAY enabled.")
        except (AttributeError, OSError) as e: print(f"Note: TCP_NODELAY: {e}")
        return sock
    except socket.timeout: print(f"Error: Connection attempt timed out."); sock.close(); return None
    except socket.error as e: print(f"Error connecting: {e}"); sock.close(); return None
    except Exception as e: print(f"Unexpected connection error: {e}"); sock.close(); return None

# --- Simple Read Function (Using Blocking Read with Timeout - Unchanged) ---
def attempt_blocking_read(sock, thread_name="Reader"):
    if not sock or stop_event.is_set(): return False
    read_success = False
    with socket_lock:
        try:
            data = sock.recv(RECEIVE_BUFFER_SIZE)
            if data: print(f"DEBUG ({thread_name}): Received: {data.decode('utf-8', errors='ignore').strip()}")
            else: print(f"DEBUG ({thread_name}): Received empty bytes - connection likely closed."); stop_event.set(); return False
            read_success = True
        except socket.timeout: read_success = True; pass # Expected
        except socket.error as e: print(f"Socket Error ({thread_name}) read: {e}"); stop_event.set(); read_success = False
        except Exception as e: print(f"Unexpected error read: {e}"); stop_event.set(); read_success = False
    return read_success

# --- Script Entry Point ---
if __name__ == "__main__":
    print("--- Elegoo Smart Car Minimal Read Test (v10 - Added Delay) ---")
    print(f"Target: {CAR_IP}:{CAR_PORT}")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {now}")
    print(f"Connecting, sending ONE stop command, waiting {INITIAL_READ_DELAY}s, then reading.")
    print("----------------------------------------------------")

    car_socket = None
    loop_count = 0

    try:
        car_socket = connect_to_car(CAR_IP, CAR_PORT)
        if car_socket:
            print("Connection established.")
            stop_event.clear()
            print(f"Sending initial command: {CMD_STOP.strip()}")
            send_success = False
            with socket_lock:
                try:
                    car_socket.sendall(CMD_STOP.encode('utf-8'))
                    send_success = True
                    print("Initial command sent.")
                    # >>> ADDED DELAY HERE <<<
                    print(f"Waiting {INITIAL_READ_DELAY}s before initial read...")
                    time.sleep(INITIAL_READ_DELAY)
                    # >>> END DELAY <<<
                    car_socket.settimeout(READ_TIMEOUT) # Ensure timeout is set for read
                    print("Attempting initial read...")
                    attempt_blocking_read(car_socket, "InitialRead")
                except socket.error as e: print(f"Error sending/initial read setup: {e}"); stop_event.set()
                except Exception as e: print(f"Unexpected error sending/initial read setup: {e}"); stop_event.set()

            if send_success and not stop_event.is_set(): # Check stop_event too
                print(f"Entering read loop (Timeout={READ_TIMEOUT}s, Interval={READ_INTERVAL}s). Press Ctrl+C...")
                while not stop_event.is_set():
                    loop_count += 1
                    # print(f"Loop {loop_count}, Calling read...") # Reduce noise
                    if not attempt_blocking_read(car_socket, "ReadLoop"):
                        print("Read attempt function signaled failure, exiting loop.")
                        break
                    # print(f"Loop {loop_count}, Waiting...") # Reduce noise
                    stop_event.wait(timeout=READ_INTERVAL)
                    if stop_event.is_set(): print("Stop event detected after wait."); break
                print(f"Exited read loop after {loop_count} iterations.")
            elif not send_success:
                 print("Send failed, did not enter read loop.")
            else: # stop_event must be set
                 print("Stop event set before read loop, not entering.")
        else:
            print("Failed to establish initial connection. Exiting.")
            sys.exit(1)
    except KeyboardInterrupt: print("\nKeyboardInterrupt detected."); stop_event.set()
    except Exception as e: print(f"\nMain Exception: {e}"); import traceback; traceback.print_exc(); stop_event.set()
    finally:
        print("Cleanup: Setting stop event...")
        stop_event.set()
        if car_socket:
            print("Cleanup: Closing socket...")
            with socket_lock:
                 try: car_socket.shutdown(socket.SHUT_RDWR)
                 except (OSError, socket.error): pass
                 finally: car_socket.close(); print("Cleanup: Socket closed.")
        print("Cleanup: Program finished.")