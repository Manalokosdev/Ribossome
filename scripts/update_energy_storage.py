import json
import re

# Read the JSON file
with open(r'c:\Filipe\ALsimulatorv3\config\part_properties.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Update amino acids (indices 0-19) to store 1.0 energy
for part in data['parts']:
    if part['index'] >= 0 and part['index'] <= 19:
        # vec4[2][3] is the energy storage value (4th element of 3rd vec4)
        part['vec4'][2][3] = 1.0
        print(f"Updated {part['name']} (index {part['index']}) energy storage to 1.0")

    # Update storage organ (index 28) to store 10.0 energy
    if part['index'] == 28:
        part['vec4'][2][3] = 10.0
        print(f"Updated {part['name']} (index {part['index']}) energy storage to 10.0")

# Write back to the JSON file
with open(r'c:\Filipe\ALsimulatorv3\config\part_properties.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("\nSuccessfully updated part_properties.json")
