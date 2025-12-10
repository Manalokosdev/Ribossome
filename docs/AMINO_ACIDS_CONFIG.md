# Amino Acid Configuration System

## Overview

The amino acid properties that define agent body segments are now fully configurable via an external JSON file (`amino_acids.json`). This allows you to experiment with different properties without recompiling the application.

## Usage

1. **Edit Properties**: Open `amino_acids.json` and modify any amino acid properties you want to change
2. **Restart Application**: Simply restart the application to load the new properties
3. **Fallback**: If the JSON file is missing or invalid, the application will automatically use hardcoded default values

## JSON Structure

The `amino_acids.json` file contains an array of 20 amino acid definitions, each with the following properties:

```json
{
  "amino_acids": [
    {
      "code": "A",
      "name": "Alanine",
      "segment_length": 6.5,
      "thickness": 2.5,
      "base_angle": 0.0,
      "alpha_sensitivity": -0.5,
      "beta_sensitivity": 0.5,
      "is_propeller": false,
      "thrust_force": 0.0,
      "color": [0.3, 0.3, 0.3],
      "is_mouth": false,
      "energy_absorption_rate": 0.0,
      "beta_absorption_rate": 0.3,
      "beta_damage": 0.5,
      "energy_storage": 0.0,
      "energy_consumption": 0.001,
      "is_alpha_sensor": false,
      "is_beta_sensor": false,
      "signal_decay": 0.2,
      "alpha_left_mult": 0.8,
      "alpha_right_mult": 0.2,
      "beta_left_mult": 0.7,
      "beta_right_mult": 0.3
    },
    ...
  ]
}
```

## Property Descriptions

### Structural Properties
- **segment_length** (float): Length of the body segment
- **thickness** (float): Width/thickness of the segment
- **base_angle** (float): Default angle offset for this segment type
- **color** (array of 3 floats): RGB color values [R, G, B] in range 0.0-1.0

### Signal Processing
- **alpha_sensitivity** (float): How much this segment responds to alpha (food) signals
- **beta_sensitivity** (float): How much this segment responds to beta (poison) signals
- **signal_decay** (float): How fast signals decay through this segment
- **alpha_left_mult** (float): Signal routing to left neighbor for alpha
- **alpha_right_mult** (float): Signal routing to right neighbor for alpha
- **beta_left_mult** (float): Signal routing to left neighbor for beta
- **beta_right_mult** (float): Signal routing to right neighbor for beta

### Sensing
- **is_alpha_sensor** (boolean): If true, this segment can detect alpha (food) in the environment
- **is_beta_sensor** (boolean): If true, this segment can detect beta (poison) in the environment

### Movement
- **is_propeller** (boolean): If true, this segment provides thrust for movement
- **thrust_force** (float): Amount of thrust force (only used if is_propeller is true)

### Energy & Feeding
- **is_mouth** (boolean): If true, this segment can consume resources from the environment
- **energy_absorption_rate** (float): Rate of alpha (food) absorption
- **beta_absorption_rate** (float): Rate of beta (poison) absorption
- **beta_damage** (float): Damage multiplier for beta consumption
- **energy_storage** (float): Additional energy capacity provided by this segment
- **energy_consumption** (float): Energy cost per simulation step

## Special Amino Acids

The current configuration includes several specialized amino acids:

- **K (Lysine)** - MOUTH: `is_mouth: true`, provides energy absorption and storage
- **P (Proline)** - PROPELLER: `is_propeller: true`, provides thrust for movement
- **S (Serine)** - ALPHA SENSOR: `is_alpha_sensor: true`, detects food (green color)
- **C (Cysteine)** - BETA SENSOR: `is_beta_sensor: true`, detects poison (red color)
- **W (Tryptophan)** - STORAGE: Provides additional energy storage capacity (orange color)
- **M (Methionine)** - START: Traditional start codon marker (pale yellow color)

## Tips for Experimentation

1. **Backup the Original**: Make a copy of `amino_acids.json` before making changes
2. **Small Changes**: Start with small adjustments to individual properties
3. **Balance Energy**: Ensure energy consumption doesn't make survival impossible
4. **Test Iterations**: Run the simulation after each change to observe effects
5. **Version Control**: Use git to track different configurations

## Example Modifications

### Making Agents Faster
Increase the `thrust_force` for Proline (P):
```json
"thrust_force": 10.0  // Increased from 5.0
```

### Increasing Sensing Range
Boost sensor sensitivity:
```json
"alpha_sensitivity": -1.0  // Increased magnitude
```

### Creating Energy-Efficient Agents
Reduce consumption for all amino acids:
```json
"energy_consumption": 0.0005  // Reduced from 0.001
```

### Larger Body Segments
Increase segment lengths and thickness:
```json
"segment_length": 10.0,  // Increased
"thickness": 5.0         // Increased
```

## Implementation Details

- **File Format**: JSON with UTF-8 encoding
- **Validation**: Exactly 20 amino acids must be defined
- **Loading**: Happens once at application startup
- **GPU Transfer**: Properties are uploaded to GPU as a uniform buffer
- **Performance**: No runtime overhead compared to hardcoded values

## Troubleshooting

- **"Failed to load amino_acids.json"**: Check that the file exists in the same directory as the executable
- **Invalid JSON**: Verify JSON syntax using a validator (e.g., jsonlint.com)
- **Missing Properties**: Ensure all 21 properties are defined for each amino acid
- **Wrong Count**: Verify exactly 20 amino acids are in the array
