# Agent File Format

## Format

Agents are stored as JSON files with a single field:

```json
{
  "genome_string": "XXXXXXXXXX...XXXXXX"
}
```

## Genome String Format

- **Length**: 256 characters (representing 256 bases)
- **Characters**: 
  - `A`, `C`, `G`, `U` - RNA bases (genes)
  - `X` - Non-coding/junk DNA
- **Structure**: The genome is scanned for genes during morphology build on first frame

## Gene Structure

Genes follow promoter-codon-terminator patterns:

### Amino Acid Genes (Body Parts)
- **Start**: Promoter (`K` for alpha signal, `C` for beta signal)
- **Codons**: 3-base sequences that encode amino acids (20 types: A-Y, excluding X)
- **End**: Terminator (`UAA`, `UAG`, or `UGA`)

Example: `KAAAAAA UAA` = Enabler organ (K+A modifier)

### Organ Genes
- **Promoters**: `V`, `M`, `L`, `P`, `K`, `C`, `H`, `Q` (8 specific types)
- **Modifier**: Single amino acid (A-Y) that determines organ type
- **Result**: Specific organ based on promoter+modifier combination

See `ORGAN_TABLE.csv` for complete organ mappings.

## Example Genomes

### Vampire Tester
```json
{
  "genome_string": "XXXXXXXXKAVAAVAAVAAHAHAHAHAHVHVHVHVHVHVHVHXXXXXX..."
}
```
- `KA` genes = Enabler organs (emit alpha signal)
- `VA` genes = Amino acid chain
- `VH` genes = Vampire mouth organs (K+H = type 33)

### Hunter (from reference)
```json
{
  "genome_string": "XXXACGCAAUGGAUGACAUCAUACAUUGACAAAGAACUGUAACUAGUUGGAGUCGGGCAUGUGGGAUAGUUUUAGACAGUGAGGCGUGCGGGCXXX..."
}
```
- Contains various genes that will be decoded during morphology build
- Junk DNA (`X`) separates and surrounds genes

## Notes

- The genome is **NOT** pre-decoded - the shader decodes it on first frame
- Only `genome_string` field is needed - position, velocity, energy, etc. are set by spawn system
- Morphology (body structure) is built by scanning the genome for valid gene patterns
- Body parts are connected sequentially in the order genes are found
