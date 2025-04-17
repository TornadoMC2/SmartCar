import sys
import math
import json # Need this for command formatting

# --- Pygame Requirement ---
try:
    import pygame
except ImportError:
    print("----------------------------------------------------")
    print("ERROR: Pygame library not found.")
    print("Please install it using: pip install pygame")
    print("----------------------------------------------------")
    sys.exit(1)

# --- Configuration (Needed for Commands) ---
COMMAND_ID = "Elegoo"

# --- Pygame Joystick Emulator Configuration ---
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

# --- Control Parameters (Needed for Commands) ---
DEFAULT_SPEED = 255
MOTOR_ALL = 0
MOTOR_LEFT = 1
MOTOR_RIGHT = 2
ROTATION_CW = 1  # Clockwise
ROTATION_CCW = 2 # Counter-Clockwise

# --- Action States (Used for display and command mapping) ---
ACTION_STOP = 'STOP'
ACTION_FORWARD = 'FORWARD'
ACTION_BACKWARD = 'BACKWARD'
ACTION_LEFT = 'LEFT'
ACTION_RIGHT = 'RIGHT'

# --- JSON Command Generation ---
def create_command_json(motor_selection, speed, rotation_direction):
    """Creates the JSON command string (without newline for cleaner printing)."""
    command_dict = {
        "H": COMMAND_ID, "N": 1, "D1": int(motor_selection),
        "D2": int(speed), "D3": int(rotation_direction)
    }
    # Return just the JSON part, not the trailing newline for console output
    json_string = json.dumps(command_dict)
    return json_string

# --- Pre-generate Command Strings ---
# (These represent what *would* be sent)
CMD_STOP = create_command_json(MOTOR_ALL, 0, ROTATION_CW)
CMD_FORWARD = create_command_json(MOTOR_ALL, DEFAULT_SPEED, ROTATION_CW)
CMD_BACKWARD = create_command_json(MOTOR_ALL, DEFAULT_SPEED, ROTATION_CCW)
CMD_TURN_LEFT_1 = create_command_json(MOTOR_LEFT, DEFAULT_SPEED, ROTATION_CCW)
CMD_TURN_LEFT_2 = create_command_json(MOTOR_RIGHT, DEFAULT_SPEED, ROTATION_CW)
CMD_TURN_RIGHT_1 = create_command_json(MOTOR_LEFT, DEFAULT_SPEED, ROTATION_CW)
CMD_TURN_RIGHT_2 = create_command_json(MOTOR_RIGHT, DEFAULT_SPEED, ROTATION_CCW)


def run_joystick_gui_with_print():
    """Runs the Pygame joystick GUI and prints corresponding commands to console."""

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Joystick Cmd Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    joystick_active = False
    current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y]
    current_action = ACTION_STOP
    last_printed_action = None # Track the last action for which commands were printed

    print("\n--- Joystick Command Simulator ---")
    print("Control the joystick in the Pygame window.")
    print("Commands that *would* be sent will appear here when the action changes.")
    print("Close the Pygame window to Quit.")
    print("------------------------------------")

    running = True
    while running:
        calculated_action_this_frame = ACTION_STOP # Default unless active drag

        # --- Event Handling (Same as before) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("\nSimulator: Quit requested via window close.")
                running = False
                break

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_x, mouse_y = event.pos
                    dist_sq = (mouse_x - JOYSTICK_CENTER_X)**2 + (mouse_y - JOYSTICK_CENTER_Y)**2
                    if dist_sq <= BOUNDARY_RADIUS**2:
                        joystick_active = True
                        current_knob_pos = list(event.pos)
                    else:
                        joystick_active = False

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    joystick_active = False
                    # Action determined below will be STOP

            elif event.type == pygame.MOUSEMOTION:
                if joystick_active:
                    current_knob_pos = list(event.pos)
        if not running: break # Exit loop if QUIT event occurred

        # --- Determine Action based on Knob Position (Same as before) ---
        if joystick_active:
            dx = current_knob_pos[0] - JOYSTICK_CENTER_X
            dy = current_knob_pos[1] - JOYSTICK_CENTER_Y
            dist = math.sqrt(dx**2 + dy**2)

            # Clamp knob position
            if dist > BOUNDARY_RADIUS:
                scale = BOUNDARY_RADIUS / dist
                current_knob_pos[0] = JOYSTICK_CENTER_X + dx * scale
                current_knob_pos[1] = JOYSTICK_CENTER_Y + dy * scale
                dx = current_knob_pos[0] - JOYSTICK_CENTER_X
                dy = current_knob_pos[1] - JOYSTICK_CENTER_Y
                dist = BOUNDARY_RADIUS

            # Determine action if outside dead zone
            if dist > DEAD_ZONE_RADIUS:
                angle = math.atan2(-dy, dx) # y is inverted in pygame coords
                if -math.pi * 0.25 <= angle < math.pi * 0.25:
                    calculated_action_this_frame = ACTION_RIGHT
                elif math.pi * 0.25 <= angle < math.pi * 0.75:
                    calculated_action_this_frame = ACTION_FORWARD
                elif math.pi * 0.75 <= angle or angle < -math.pi * 0.75 :
                      calculated_action_this_frame = ACTION_LEFT
                elif -math.pi * 0.75 <= angle < -math.pi * 0.25:
                    calculated_action_this_frame = ACTION_BACKWARD
            else:
                calculated_action_this_frame = ACTION_STOP
        else:
            current_knob_pos = [JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y]
            calculated_action_this_frame = ACTION_STOP

        # Update the current action state
        current_action = calculated_action_this_frame

        # --- Map Action to Commands and Print if Action Changed ---
        if current_action != last_printed_action:
            commands_to_print = []
            action_desc = current_action # Use the action name directly

            if current_action == ACTION_STOP:
                commands_to_print = [CMD_STOP]
            elif current_action == ACTION_FORWARD:
                commands_to_print = [CMD_FORWARD]
            elif current_action == ACTION_BACKWARD:
                commands_to_print = [CMD_BACKWARD]
            elif current_action == ACTION_LEFT:
                commands_to_print = [CMD_TURN_LEFT_1, CMD_TURN_LEFT_2]
            elif current_action == ACTION_RIGHT:
                commands_to_print = [CMD_TURN_RIGHT_1, CMD_TURN_RIGHT_2]

            print(f"\n--- Action Changed: {action_desc} ---")
            if commands_to_print:
                for i, cmd_json in enumerate(commands_to_print):
                    print(f"Would send ({i+1}/{len(commands_to_print)}): {cmd_json}")
            else:
                # Should not happen with current actions, but handle just in case
                print("(No command defined for this action)")

            last_printed_action = current_action # Update last printed action

        # --- Drawing (Same as before) ---
        screen.fill(COLOR_WHITE)
        pygame.draw.circle(screen, COLOR_GRAY, (JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y), BOUNDARY_RADIUS, 2)
        pygame.draw.circle(screen, COLOR_RED, (JOYSTICK_CENTER_X, JOYSTICK_CENTER_Y), DEAD_ZONE_RADIUS, 1)
        knob_color = COLOR_BLUE if joystick_active else COLOR_BLACK
        pygame.draw.circle(screen, knob_color, (int(current_knob_pos[0]), int(current_knob_pos[1])), KNOB_RADIUS)
        instruction_text = [
            "Click & Drag in Circle",
            "Release to Stop",
            f"Detected Action: {current_action}"
        ]
        for i, line in enumerate(instruction_text):
            text_surface = font.render(line, True, COLOR_BLACK)
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 70 + i * 25))
            screen.blit(text_surface, text_rect)

        pygame.display.flip()
        clock.tick(30)

    # --- Cleanup ---
    print("\nExiting joystick command simulator.")
    pygame.quit()

# --- Script Entry Point ---
if __name__ == "__main__":
    if 'pygame' not in sys.modules:
        print("Pygame not loaded. Please install it.")
    else:
        run_joystick_gui_with_print()

    print("Program finished.")