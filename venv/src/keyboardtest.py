import keyboard

print("Keyboard module imported successfully.")

while True:
    if keyboard.is_pressed('q'):
        print("You pressed 'q'. Exiting...")
        break
