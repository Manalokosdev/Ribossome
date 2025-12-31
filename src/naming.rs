pub mod sim {
    // Simulation run naming

    // Kept as simple static slices (no extra crates needed).
    static SIM_NAME_ADJECTIVES: &[&str] = &[
        "Tranquillus", "Placidus", "Quietum", "Serenus", "Aequus", "Lenis", "Mitis", "Pacificus",
        "Stabilis", "Immobilis", "Constans", "Fixus", "Tacitus", "Silens", "Calmus", "Languidus",
        "Vivax", "Vitalis", "Mobilis", "Agilis", "Celer", "Rapidum", "Fluxus", "Motus",
        "Dinamicus", "Vigens", "Ardens", "Fervens", "Vehemens", "Intensus", "Acutus", "Vividus",
        "Turbidus", "Tumultuosus", "Procellosus", "Tempestivus", "Fulminans", "Turbulentus", "Violentus",
        "Atrox", "Ferox", "Saevius", "Furibundus", "Rabidus", "Insanus", "Caecus", "Vastus", "Horridus",
        "Verdans", "Viridis", "Florens", "Frondosus", "Nemorosus", "Silvestris", "Aquosus", "Umbracus",
        "Montanus", "Campester", "Maritimus", "Fluvialis", "Lacustris", "Palustris", "Aridus", "Siccus",
        "Gelidus", "Frigidus", "Algidus", "Nivalis", "Glacialis", "Calidus", "Tepidum", "Ardentem",
        "Ignavus", "Torpidus", "Lethargicus", "Iners", "Segnis", "Tardus", "Lentus", "Piger",
        "Acerrimus", "Acerbus", "Asper", "Durus", "Rigidas", "Severus", "Austerus", "Gravus",
        "Levis", "Suavis", "Dulcis", "Amoenus", "Iucundus", "Festivus", "Laetus", "Hilarius",
        "Obscurus", "Tenebrosus", "Caliginosus", "Nebulosus", "Umbrifer", "Lucidus", "Clarus", "Splendidus",
        "Magnificus", "Grandis", "Vastus", "Immensus", "Infinitus", "Aetherius", "Caelestis", "Sublimis",
        "Humilis", "Modestus", "Parvus", "Exiguus", "Minutus", "Tenui", "Fragilis", "Delicatus",
        "Ventosus", "Aurae", "Zephyrus", "Borealis", "Australis", "Orientalis", "Occidentalis", "Polaris",
        "Tropicus", "Aequatorialis", "Alpinus", "Cavernosus", "Volcanicus", "Seismicus", "Telluricus", "Cosmicus",
    ];

    static SIM_NAME_NOUNS: &[&str] = &[
        "Vastitas", "Eremus", "Solitudo", "Desertum", "Vastus", "Aridum", "Sterilis", "Inhospitus",
        "Barathrum", "Abyssus", "Chaos", "Vacuum", "Nihilum", "Tenebrae", "Umbra", "Caligo",
        "Locus", "Regio", "Territorium", "Area", "Spacium", "Campus", "Planities", "Pratum",
        "Ager", "Fundus", "Terra", "Tellus", "Orbis", "Mundus", "Natura", "Cosmos",
        "Oecosystema", "Biota", "Vita", "Animus", "Populus", "Gens", "Multitudo", "Turba",
        "Silva", "Saltus", "Nemus", "Lucus", "Dumus", "Frutex", "Virgultum", "Spissum",
        "Selva", "Jungla", "Dense", "Opacus", "Intricatus", "Impenetrabilis", "Ferax", "Uber",
        "Aqua", "Flumen", "Lacus", "Mare", "Oceanus", "Rivus", "Fons", "Palus",
        "Mons", "Collis", "Vallis", "Cavum", "Specus", "Rupes", "Saxum", "Lapis",
        "Arbor", "Frons", "Ramus", "Folium", "Herba", "Flos", "Fructus", "Semen",
        "Ventus", "Aura", "Turbo", "Procella", "Tempestas", "Nimbus", "Imber", "Pluvia",
        "Ignis", "Flamma", "Focus", "Vulcanus", "Aestus", "Calor", "Frigus", "Gelum",
        "Sol", "Luna", "Stella", "Caelum", "Aether", "Nubes", "Arcus", "Iris",
        "Fauna", "Flora", "Bestia", "Avis", "Piscis", "Serpens", "Insectum", "Vermis",
        "Paradises", "Elysium", "Arcadia", "Utopia", "Hortus", "Viridarium", "Pomarium", "Olivetum",
        "Vinea", "Frumentum", "Seges", "Messis", "Abundantia", "Opulentia", "Fertilitas", "Largitas",
    ];

    fn lerp_index01(score01: f32, len: usize) -> usize {
        if len == 0 {
            return 0;
        }
        let t = score01.clamp(0.0, 1.0);
        ((t * (len as f32 - 1.0)).round() as usize).min(len - 1)
    }

    pub(super) fn safe01(numer: f32, denom: f32) -> f32 {
        if denom <= 0.0 {
            return 0.0;
        }
        (numer / denom).clamp(0.0, 1.0)
    }

    fn mix32(mut x: u32) -> u32 {
        // A small non-cryptographic mixer (good enough for stable jitter).
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^= x >> 16;
        x
    }

    pub(super) fn jittered_index(score01: f32, len: usize, seed: u32, salt: u32) -> usize {
        if len <= 1 {
            return 0;
        }
        let base = lerp_index01(score01, len);

        // Keep it "coherent": only jitter within a small neighborhood.
        let window = (len / 16).max(2);
        let span = (window * 2 + 1) as u32;
        let h = mix32(seed ^ salt);
        let offset = (h % span) as i32 - window as i32;

        (base as i32 + offset).clamp(0, len as i32 - 1) as usize
    }

    /// Generates a deterministic-ish human-readable run name from current settings.
    ///
    /// Output format: `Ribossome_<Adj>-<Noun>-<seed%10000>_<YYYYMMDD_HHMMSS_mmm>UTC`
    pub fn generate_sim_name(settings: &crate::SimulationSettings, run_seed: u32, max_agents: u32) -> String {
        // Environment score (0 = calm, 1 = stormy)
        let mut env_terms = 0.0f32;
        let mut env_count = 0.0f32;

        // These are only meaningful when the fluid sim is enabled.
        if settings.fluid_enabled {
            env_terms += safe01(settings.fluid_vorticity, 50.0);
            env_terms += safe01(settings.fluid_viscosity, 5.0);
            env_count += 2.0;
        }

        // Terrain/fluid coupling.
        env_terms += safe01(settings.fluid_slope_force_scale, 500.0);
        env_terms += safe01(settings.fluid_obstacle_strength, 1000.0);
        env_count += 2.0;

        // Repulsion is always active.
        env_terms += safe01(settings.repulsion_strength, 100.0);
        env_count += 1.0;

        let env_score = if env_count > 0.0 { (env_terms / env_count).clamp(0.0, 1.0) } else { 0.0 };

        // Population score (0 = wasteland, 1 = thriving)
        let spawn_norm = safe01(settings.spawn_probability, 5.0);
        let death_inv = (1.0 - safe01(settings.death_probability, 0.1)).clamp(0.0, 1.0);
        let mutation_norm = safe01(settings.mutation_rate, 0.1);
        let food_norm = safe01(settings.food_power, 10.0);
        let poison_inv = (1.0 - safe01(settings.poison_power, 10.0)).clamp(0.0, 1.0);
        // Current builds use ~60k max agents; normalize to that for a meaningful 0..1 scale.
        let capacity_norm = safe01(max_agents as f32, 60_000.0);

        let pop_score =
            ((spawn_norm + death_inv + mutation_norm + food_norm + poison_inv + capacity_norm) / 6.0).clamp(0.0, 1.0);

        let adj = SIM_NAME_ADJECTIVES[jittered_index(env_score, SIM_NAME_ADJECTIVES.len(), run_seed, 0xA11C_E551)];
        let noun = SIM_NAME_NOUNS[jittered_index(pop_score, SIM_NAME_NOUNS.len(), run_seed, 0xBADC_0DE5)];

        let num = run_seed % 10_000;

        // Include milliseconds so rapid successive resets still get unique names.
        let now = chrono::Utc::now();
        let ts = format!(
            "{}_{:03}",
            now.format("%Y%m%d_%H%M%S"),
            now.timestamp_subsec_millis()
        );

        format!("Ribossome_{}-{}-{:04}_{}UTC", adj, noun, num, ts)
    }
}

pub mod agent {
    use crate::{Agent, MAX_BODY_PARTS, PART_TYPE_COUNT};
    use super::sim::{jittered_index, safe01};

    // Part type indices (keep in sync with PART_TYPE_NAMES in src/main.rs)
    const PT_MOUTH: usize = 20;
    const PT_PROPELLER: usize = 21;
    const PT_ALPHA_SENSOR: usize = 22;
    const PT_BETA_SENSOR: usize = 23;
    const PT_ENERGY_SENSOR: usize = 24;
    const PT_DISPLACER_A: usize = 25;
    const PT_ENABLER: usize = 26;
    const PT_DISPLACER_B: usize = 27;
    const PT_STORAGE: usize = 28;
    const PT_POISON_RESIST: usize = 29;
    const PT_CHIRAL_FLIPPER: usize = 30;
    const PT_CLOCK: usize = 31;
    const PT_SLOPE_SENSOR: usize = 32;
    const PT_VAMPIRE_MOUTH: usize = 33;
    const PT_AGENT_ALPHA_SENSOR: usize = 34;
    const PT_AGENT_BETA_SENSOR: usize = 35;
    const PT_TRAIL_ENERGY_SENSOR: usize = 37;
    const PT_ALPHA_MAG_SENSOR: usize = 38;
    const PT_ALPHA_MAG_SENSOR_V2: usize = 39;
    const PT_BETA_MAG_SENSOR: usize = 40;
    const PT_BETA_MAG_SENSOR_V2: usize = 41;
    const PT_ANCHOR: usize = 42;

    const SALT_GENUS: u32 = 0x51D1_7E02;
    const SALT_SPECIES: u32 = 0x51D1_7E03;

    const SALT_FAMILY_GENUS: u32 = 0x51D1_7E11;
    const SALT_FAMILY_SPECIES: u32 = 0x51D1_7E12;

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    enum GenusCategory {
        Small,
        Large,
        Predator,
        Mobile,
        Sensory,
        Defensive,
        General,
    }

    // === Words grouped thematically ===

    // Multiple vocab “families” (lexicons). Selection is hash-based so the same
    // composition logic yields different word sets, reducing collisions while keeping
    // only two name tokens (genus-species).

    static GENUS_SMALL_FAMILIES: &[&[&str]] = &[
        &["Parvus", "Minimus", "Exiguus", "Pumilus", "Humilis", "Modestus", "Tenui", "Fragilis", "Delicatus", "Nanoides"],
        &["Minutus", "Pusillus", "Angustus", "Brevis", "Levis", "Subtilis", "Inops", "Exilis", "Tenuis", "Parvulus"],
        &["Micrus", "Nanicus", "Pauper", "Humectus", "Tardus", "Tacitus", "Simplex", "Lenis", "Lentus", "Placatus"],
        &["Mediocris", "Modicus", "Curtus", "Strictus", "Adstrictus", "Aridus", "Siccus", "Sordidus", "Opacus", "Iners"],
    ];

    static GENUS_LARGE_FAMILIES: &[&[&str]] = &[
        &["Giganteus", "Magnus", "Grandis", "Vastus", "Immensus", "Multiformis", "Colossus", "Maximus"],
        &["Robustus", "Potens", "Validus", "Gravis", "Altus", "Longus", "Amplus", "Praegrandis"],
        &["Titanicus", "Herculeus", "Formidabilis", "Immanis", "Ingens", "Superbus", "Praeclarus", "Regalis"],
        &["Crassus", "Turgidus", "Massivus", "Plenarius", "Fertilis", "Opulentus", "Abundans", "Copiosus"],
    ];

    static GENUS_PREDATOR_FAMILIES: &[&[&str]] = &[
        &["Carnivorus", "Ferox", "Atrox", "Vorax", "Rapax", "Venator", "Mordax", "Saevius"],
        &["Furibundus", "Rabidus", "Hostilis", "Perniciosus", "Acer", "Avidus", "Raptor", "Sagax"],
        &["Dentatus", "Mandibulatus", "Laniator", "Praedo", "Predator", "Lupinus", "Tigrinus", "Draco"],
        &["Vampyricus", "Sanguineus", "Nocturnus", "Tenebrosus", "Insanus", "Caecus", "Obscurus", "Umbrosus"],
    ];

    static GENUS_MOBILE_FAMILIES: &[&[&str]] = &[
        &["Velox", "Agilis", "Celer", "Rapidum", "Cursor", "Erraticus", "Vagus", "Fugax"],
        &["Natans", "Volitans", "Fluitans", "Mobilis", "Motus", "Migrator", "Peregrinus", "Salientis"],
        &["Turbidus", "Procellosus", "Impetuosus", "Vehemens", "Acer", "Strenuus", "Vividus", "Incitatus"],
        &["Flexilis", "Curvatus", "Sinuatus", "Rotundus", "Circinus", "Torqueus", "Oscillans", "Cyclicus"],
    ];

    static GENUS_SENSORY_FAMILIES: &[&[&str]] = &[
        &["Acutus", "Argutus", "Vigilis", "Perceptus", "Explorator", "Lynceus", "Sensorus", "Intuitivus"],
        &["Clarus", "Lucidus", "Splendidus", "Perspicuus", "Sagax", "Providus", "Prudens", "Attentus"],
        &["Electivus", "Magneticus", "Tactilis", "Auditor", "Olfactus", "Visorius", "Sensilis", "Cognitus"],
        &["Calibrator", "Mensorius", "Index", "Spector", "Vigilans", "Curiosus", "Inquisitor", "Observator"],
    ];

    static GENUS_DEFENSIVE_FAMILIES: &[&[&str]] = &[
        &["Fortis", "Durus", "Loricatus", "Defensus", "Inermis", "Severus", "Rigidus", "Resistens"],
        &["Conservus", "Reservatus", "Firmus", "Stabilis", "Constans", "Fixus", "Securus", "Tutela"],
        &["Armatus", "Scutatus", "Coriaceus", "Crassus", "Spinosus", "Munitor", "Vallatus", "Aegis"],
        &["Antidotus", "Immunis", "Chiralis", "Mutatio", "Purgatus", "Neutralis", "Purificus", "Sanus"],
    ];

    static GENUS_GENERAL_FAMILIES: &[&[&str]] = &[
        &["Vulgaris", "Communis", "Simplex", "Domesticus", "Gregarius", "Ordinarius", "Typicus"],
        &["Medius", "Intermedius", "Moderatus", "Aequus", "Affinis", "Consuetus", "Familiaris", "Trivialis"],
        &["Varius", "Mutabilis", "Diversus", "Multus", "Compositus", "Mixtus", "Conexus", "Concolor"],
        &["Civicus", "Urbanus", "Rusticus", "Silvestris", "Campestris", "Maritimus", "Montanus", "Palustris"],
    ];

    // Species tiers — rare (1–2), balanced (3–7), massed (8+)
    // Each has multiple families.
    static SPECIES_FEEDER_RARE_FAMILIES: &[&[&str]] = &[
        &["dentatus", "mandibulatus", "vorax", "rapax"],
        &["mordax", "laniatus", "hians", "edax"],
        &["aculeatus", "rostratus", "hamatus", "uncinatus"],
        &["sanguivorus", "noctivorus", "tenebricus", "umbrosus"],
    ];
    static SPECIES_FEEDER_BALANCED_FAMILIES: &[&[&str]] = &[
        &["carnivorus", "praedator", "edax"],
        &["venator", "raptor", "sectator"],
        &["ferox", "acer", "avidus"],
        &["sagax", "callidus", "astutus"],
    ];
    static SPECIES_FEEDER_MASSED_FAMILIES: &[&[&str]] = &[
        &["multidentatus", "polyvorax", "hians", "gulosus"],
        &["polycephalus", "multirostris", "plurimus", "abundans"],
        &["insatiabilis", "voracissimus", "rapacissimus", "edacissimus"],
        &["sanguineus", "cruentus", "nocturnus", "tenebrosus"],
    ];

    static SPECIES_MOVER_RARE_FAMILIES: &[&[&str]] = &[
        &["agilis", "cursor", "propulsus"],
        &["motilis", "mobilis", "saltans"],
        &["natans", "volitans", "fluitans"],
        &["oscillans", "circularis", "rotator"],
    ];
    static SPECIES_MOVER_BALANCED_FAMILIES: &[&[&str]] = &[
        &["velox", "erraticus", "natans"],
        &["celer", "rapidus", "fugax"],
        &["mobilis", "motorius", "agilis"],
        &["sinuatus", "flexilis", "curvatus"],
    ];
    static SPECIES_MOVER_MASSED_FAMILIES: &[&[&str]] = &[
        &["polypropulsus", "multipes", "rapidissimus", "fugax"],
        &["celerissimus", "impetuosus", "vehemens", "incitatus"],
        &["motus", "turbidus", "procellosus", "tempestivus"],
        &["cursus", "peregrinus", "migrator", "vagus"],
    ];

    static SPECIES_SENSOR_RARE_FAMILIES: &[&[&str]] = &[
        &["acutus", "sensorius", "vigilans"],
        &["lucidus", "clarus", "perspicuus"],
        &["explorator", "observator", "inquisitor"],
        &["magneticus", "tactilis", "mensorius"],
    ];
    static SPECIES_SENSOR_BALANCED_FAMILIES: &[&[&str]] = &[
        &["perceptivus", "argutus", "explorator"],
        &["attentus", "prudens", "sagax"],
        &["visorius", "auditor", "olfactus"],
        &["index", "speculator", "vigilus"],
    ];
    static SPECIES_SENSOR_MASSED_FAMILIES: &[&[&str]] = &[
        &["omnisensorius", "panopticus", "multivigilans"],
        &["perspicacissimus", "acutissimus", "lucidissimus"],
        &["multisensor", "polysensorius", "omniindex"],
        &["spectator", "observator", "inquisitor"],
    ];

    static SPECIES_UTILITY_FAMILIES: &[&[&str]] = &[
        &["reservatus", "conservativus", "enablatus", "poisonoresistens", "mutatioprotectus", "chiralis", "anchora"],
        &["stabilis", "constans", "fixus", "firmus", "securus", "tutus"],
        &["immunis", "antidotus", "purgatus", "neutralis", "purificus"],
        &["opulens", "copiosus", "plenus", "locuples", "fertilis"],
    ];

    static SPECIES_SIZE_FAMILIES: &[&[&str]] = &[
        &["parvus", "minutus", "exiguus", "giganteus", "magnus", "longus", "vastus"],
        &["brevis", "curtus", "altus", "latus", "angustus", "amplus"],
        &["crassus", "turgidus", "plenus", "tenuis", "subtilis", "fragilis"],
        &["grandis", "immensus", "maximus", "ingens", "immanis", "colossus"],
    ];

    static SPECIES_DIVERSITY_HIGH_FAMILIES: &[&[&str]] = &[
        &["multiformis", "polymorphus", "diversus", "variabilis", "complexus"],
        &["compositus", "mixtus", "conexus", "multiplex", "conflatus"],
        &["mutabilis", "versatilis", "variegatus", "alternans", "cathena"],
        &["articulatus", "modulatus", "complicatus", "intricatus", "multus"],
    ];
    static SPECIES_DIVERSITY_LOW_FAMILIES: &[&[&str]] = &[
        &["uniformis", "repetitus", "monomorphus", "simplex"],
        &["aequalis", "constans", "unicus", "solus"],
        &["regularis", "ordinatus", "typicus", "trivialis"],
        &["iteratus", "geminatus", "duplicatus", "serialis"],
    ];

    fn mix32(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^= x >> 16;
        x
    }

    fn classify_genus_category(
        body_count: usize,
        feeder: u32,
        mover: u32,
        sensor: u32,
        utility: u32,
        counts: &[u32; PART_TYPE_COUNT],
    ) -> GenusCategory {
        // "Genus" is meant to be a stable niche/morphology label.
        // Use coarse bins and presence flags so small sibling mutations don't flip genus.
        if body_count <= 8 {
            return GenusCategory::Small;
        }
        if body_count >= 40 {
            return GenusCategory::Large;
        }

        if counts[PT_VAMPIRE_MOUTH] > 0 {
            return GenusCategory::Predator;
        }

        let total_special = feeder + mover + sensor + utility;
        if total_special == 0 {
            return GenusCategory::General;
        }

        // Bucket counts to reduce jitter (e.g. 1 extra sensor won't necessarily flip category).
        let b_feeder = feeder / 3;
        let b_mover = mover / 3;
        let b_sensor = sensor / 3;
        let b_utility = utility / 3;

        let mut best = b_feeder;
        let mut best_cat = GenusCategory::Predator;

        if b_mover > best {
            best = b_mover;
            best_cat = GenusCategory::Mobile;
        }
        if b_sensor > best {
            best = b_sensor;
            best_cat = GenusCategory::Sensory;
        }
        if b_utility > best {
            best = b_utility;
            best_cat = GenusCategory::Defensive;
        }

        // If all buckets are zero (only 1–2 specials total), treat as generalist.
        if best == 0 {
            return GenusCategory::General;
        }

        best_cat
    }

    fn genus_signature(
        body_count: usize,
        category: GenusCategory,
        unique_types: usize,
        counts: &[u32; PART_TYPE_COUNT],
    ) -> u32 {
        // Coarse signature for genus stability across siblings.
        // - size_class: 0 small, 1 medium, 2 large
        // - category: niche/morphology group
        // - diversity_bin: 0..3 (coarse)
        // - presence flags: a few big phenotype toggles

        let size_class = if body_count <= 8 { 0u32 } else if body_count >= 40 { 2u32 } else { 1u32 };
        let cat_id = match category {
            GenusCategory::Small => 0u32,
            GenusCategory::Large => 1u32,
            GenusCategory::Predator => 2u32,
            GenusCategory::Mobile => 3u32,
            GenusCategory::Sensory => 4u32,
            GenusCategory::Defensive => 5u32,
            GenusCategory::General => 6u32,
        };

        let diversity_num = (unique_types * 4) as u32;
        let diversity_den = body_count.max(1) as u32;
        let diversity_bin = (diversity_num / diversity_den).min(3);

        let mut flags = 0u32;
        if counts[PT_VAMPIRE_MOUTH] > 0 {
            flags |= 1 << 0;
        }
        if counts[PT_ANCHOR] > 0 {
            flags |= 1 << 1;
        }
        if counts[PT_POISON_RESIST] > 0 {
            flags |= 1 << 2;
        }
        if counts[PT_STORAGE] > 0 {
            flags |= 1 << 3;
        }
        if counts[PT_CLOCK] > 0 {
            flags |= 1 << 4;
        }
        if counts[PT_CHIRAL_FLIPPER] > 0 {
            flags |= 1 << 5;
        }
        if counts[PT_ALPHA_MAG_SENSOR] + counts[PT_ALPHA_MAG_SENSOR_V2] + counts[PT_BETA_MAG_SENSOR]
            + counts[PT_BETA_MAG_SENSOR_V2]
            > 0
        {
            flags |= 1 << 6;
        }

        // Pack into a single u32
        (size_class)
            | (cat_id << 2)
            | (diversity_bin << 6)
            | (flags << 8)
    }

    fn fnv1a64_init() -> u64 {
        14695981039346656037u64
    }

    fn fnv1a64_add_u32(mut hash: u64, value: u32) -> u64 {
        for b in value.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(1099511628211u64);
        }
        hash
    }

    fn agent_fingerprint64(agent: &Agent, counts: &[u32; PART_TYPE_COUNT]) -> u64 {
        let body_count = (agent.body_count as usize).min(MAX_BODY_PARTS);
        let mut h = fnv1a64_init();

        h = fnv1a64_add_u32(h, agent.body_count);

        // Composition (counts) and order (body base_type sequence)
        for (i, &c) in counts.iter().enumerate() {
            if c == 0 {
                continue;
            }
            h = fnv1a64_add_u32(h, (i as u32) ^ c.wrapping_mul(0x9E37_79B9));
        }

        for i in 0..body_count {
            h = fnv1a64_add_u32(h, agent.body[i].base_type());
        }

        h
    }

    fn seed_from_agent(agent: &Agent, counts: &[u32; PART_TYPE_COUNT]) -> u32 {
        let fp = agent_fingerprint64(agent, counts);
        let lo = fp as u32;
        let hi = (fp >> 32) as u32;
        mix32(lo ^ hi ^ 0xA55A_1234)
    }

    fn pick_family<'a>(families: &'a [&'a [&'a str]], score01: f32, seed: u32, salt: u32) -> &'a [&'a str] {
        if families.is_empty() {
            return &[];
        }
        // Use jittered_index so similar seeds choose nearby families.
        let idx = jittered_index(score01, families.len(), seed, salt);
        families[idx]
    }

    /// Main call site API (matches existing callers).
    pub fn generate_agent_name(agent: &Agent) -> String {
        let body_count = (agent.body_count as usize).min(MAX_BODY_PARTS);
        if body_count == 0 {
            return "Anonimus-Nullus".to_string();
        }

        let mut counts = [0u32; PART_TYPE_COUNT];
        for i in 0..body_count {
            let t = agent.body[i].base_type() as usize;
            if t < PART_TYPE_COUNT {
                counts[t] += 1;
            }
        }

        let seed = seed_from_agent(agent, &counts);
        generate_agent_name_seeded(agent, seed, &counts)
    }

    fn generate_agent_name_seeded(agent: &Agent, seed: u32, counts: &[u32; PART_TYPE_COUNT]) -> String {
        let body_count = (agent.body_count as usize).min(MAX_BODY_PARTS);

        // Composition features from part_type indices (0–19 amino acids, 20–42 organs)
        let feeder = counts[PT_MOUTH] + counts[PT_VAMPIRE_MOUTH];
        let mover = counts[PT_PROPELLER] + counts[PT_DISPLACER_A] + counts[PT_DISPLACER_B];
        let sensor = counts[PT_ALPHA_SENSOR]
            + counts[PT_BETA_SENSOR]
            + counts[PT_ENERGY_SENSOR]
            + counts[PT_SLOPE_SENSOR]
            + counts[PT_AGENT_ALPHA_SENSOR]
            + counts[PT_AGENT_BETA_SENSOR]
            + counts[PT_TRAIL_ENERGY_SENSOR]
            + counts[PT_ALPHA_MAG_SENSOR]
            + counts[PT_ALPHA_MAG_SENSOR_V2]
            + counts[PT_BETA_MAG_SENSOR]
            + counts[PT_BETA_MAG_SENSOR_V2];
        let utility = counts[PT_ENABLER]
            + counts[PT_STORAGE]
            + counts[PT_POISON_RESIST]
            + counts[PT_CHIRAL_FLIPPER]
            + counts[PT_CLOCK]
            + counts[PT_ANCHOR];

        let _total_special = feeder + mover + sensor + utility;
        let unique_types = counts.iter().filter(|&&c| c > 0).count();

        let size_score = safe01(body_count as f32, 50.0);
        let diversity_score = unique_types as f32 / body_count.max(1) as f32;

        // === Genus (stable) ===
        // Genus is a coarse, stable label: "what kind of body/ecological niche is this?"
        // It should not change for minor sibling mutations, so it is derived from a coarse
        // phenotype signature (not the full fingerprint seed).
        let genus_category = classify_genus_category(body_count, feeder, mover, sensor, utility, counts);
        let gsig = genus_signature(body_count, genus_category, unique_types, counts);

        let genus_families: &[&[&str]] = match genus_category {
            GenusCategory::Small => GENUS_SMALL_FAMILIES,
            GenusCategory::Large => GENUS_LARGE_FAMILIES,
            GenusCategory::Predator => GENUS_PREDATOR_FAMILIES,
            GenusCategory::Mobile => GENUS_MOBILE_FAMILIES,
            GenusCategory::Sensory => GENUS_SENSORY_FAMILIES,
            GenusCategory::Defensive => GENUS_DEFENSIVE_FAMILIES,
            GenusCategory::General => GENUS_GENERAL_FAMILIES,
        };

        let genus_seed = mix32(gsig ^ SALT_FAMILY_GENUS);
        let genus_family = if genus_families.is_empty() {
            &[]
        } else {
            // Deterministic family selection from genus_seed (no jitter).
            let fi = (genus_seed as usize) % genus_families.len();
            genus_families[fi]
        };

        let genus = if genus_family.is_empty() {
            "Anonimus"
        } else {
            let wi = (mix32(genus_seed ^ SALT_GENUS) as usize) % genus_family.len();
            genus_family[wi]
        };

        // === Choose species family + word based on dominant trait + count tier ===
        let (species_families, score01) = if feeder >= 8 {
            (SPECIES_FEEDER_MASSED_FAMILIES, safe01(feeder as f32, 20.0))
        } else if feeder >= 3 {
            (SPECIES_FEEDER_BALANCED_FAMILIES, safe01(feeder as f32, 12.0))
        } else if feeder >= 1 {
            (SPECIES_FEEDER_RARE_FAMILIES, safe01(feeder as f32, 4.0))
        } else if mover >= 10 {
            (SPECIES_MOVER_MASSED_FAMILIES, safe01(mover as f32, 20.0))
        } else if mover >= 3 {
            (SPECIES_MOVER_BALANCED_FAMILIES, safe01(mover as f32, 12.0))
        } else if mover >= 1 {
            (SPECIES_MOVER_RARE_FAMILIES, safe01(mover as f32, 4.0))
        } else if sensor >= 12 {
            (SPECIES_SENSOR_MASSED_FAMILIES, safe01(sensor as f32, 20.0))
        } else if sensor >= 3 {
            (SPECIES_SENSOR_BALANCED_FAMILIES, safe01(sensor as f32, 12.0))
        } else if sensor >= 1 {
            (SPECIES_SENSOR_RARE_FAMILIES, safe01(sensor as f32, 4.0))
        } else if utility >= 5 {
            (SPECIES_UTILITY_FAMILIES, safe01(utility as f32, 12.0))
        } else if diversity_score > 0.7 {
            (SPECIES_DIVERSITY_HIGH_FAMILIES, diversity_score)
        } else if diversity_score < 0.4 {
            (SPECIES_DIVERSITY_LOW_FAMILIES, 1.0 - diversity_score)
        } else {
            (SPECIES_SIZE_FAMILIES, size_score)
        };

        let species_family_seed = mix32(seed ^ SALT_FAMILY_SPECIES);
        let species_family = pick_family(species_families, score01, species_family_seed, SALT_FAMILY_SPECIES);
        let species = if species_family.is_empty() {
            "nullus"
        } else {
            species_family[jittered_index(score01, species_family.len(), seed, SALT_SPECIES)]
        };

        format!("{}-{}", genus, species)
    }
}
