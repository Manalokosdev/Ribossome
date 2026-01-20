import json
import random

# Set seed for reproducibility
random.seed(42)

# Read the part properties
with open('config/part_properties.json', 'r') as f:
    data = json.load(f)

# Randomize alpha_sensitivity and beta_sensitivity for all parts
for part in data['parts']:
    # vec4[1][0] is alpha_sensitivity
    # vec4[1][1] is beta_sensitivity
    part['vec4'][1][0] = round(random.uniform(-1.0, 1.0), 6)
    part['vec4'][1][1] = round(random.uniform(-1.0, 1.0), 6)

# Write back
with open('config/part_properties.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Randomized alpha_sensitivity and beta_sensitivity for {len(data['parts'])} parts")
