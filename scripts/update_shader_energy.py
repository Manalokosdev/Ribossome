import re

# Read the shader file
with open(r'c:\Filipe\ALsimulatorv3\shaders\shared.wgsl', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the amino acid indices that need updating (1-19, since 0 was already done)
amino_acids_to_update = list(range(1, 20))

# Pattern to match amino acid definitions (indices 1-19)
# We need to match the third vec4 in each array and change the last value from 0.0 to 1.0
pattern = r'(//\s+\d+\s+[A-Z]\s+-\s+\w+.*?array<vec4<f32>,6>\(\s+vec4<f32>\([^)]+\),\s+vec4<f32>\([^)]+\),\s+vec4<f32>\([^,]+,[^,]+,[^,]+,\s*)0\.0(\s*\),)'

def replace_func(match):
    return match.group(1) + '1.0' + match.group(2)

# Replace all occurrences
new_content = re.sub(pattern, replace_func, content)

# Write back to the shader file
with open(r'c:\Filipe\ALsimulatorv3\shaders\shared.wgsl', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Successfully updated shaders/shared.wgsl")
print("Changed all amino acids (0-19) to store 1.0 energy")
