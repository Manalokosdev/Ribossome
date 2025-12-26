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

    fn safe01(numer: f32, denom: f32) -> f32 {
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

    fn jittered_index(score01: f32, len: usize, seed: u32, salt: u32) -> usize {
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
    ///
    /// Notes:
    /// - Scales are adapted to Ribossome's `SimulationSettings::sanitize()` ranges.
    /// - `run_seed` should be a stable seed for the session (e.g. captured at startup).
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
    // Agent naming (genus/species style)

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

    // NOTE: This list is intentionally kept in this dedicated module file (not in main.rs).
    // If you want the full 256+ lists, paste them here and we'll just index modulo len().
    static AGENT_GENUS_WORDS: &[&str] = &[
        "Acanthus", "Aculeatus", "Acuminatus", "Aequalis", "Aestivus", "Affinis", "Agrestis", "Alatus",
        "Albus", "Alpinus", "Amabilis", "Ambiguus", "Americanus", "Aquaticus", "Arborescens", "Arctos",
        "Arenarius", "Argenteus", "Arvensis", "Ater", "Aureus", "Auritus", "Australis", "Barbatus",
        "Bicolor", "Borealis", "Brevis", "Britannicus", "Caeruleus", "Californicus", "Canadensis", "Candidus",
        "Canescens", "Caprae", "Castaneus", "Cinereus", "Clathratus", "Collaris", "Communis", "Concolor",
        "Cordatus", "Coriaceus", "Coronatus", "Costatus", "Crassus", "Cristatus", "Crypticus", "Curvirostris",
        "Cyano", "Dactylus", "Digitatus", "Diffusus", "Domesticus", "Dulcis", "Echinatus", "Edulis",
        "Elegans", "Emarginatus", "Erectus", "Erythro", "Esculentus", "Europaeus", "Excelsus", "Exiguus",
    ];

    static AGENT_SPECIES_WORDS: &[&str] = &[
        "Lancea", "Lateralis", "Lapponicus", "Latifolium", "Laxus", "Lepidus", "Leucodon", "Lineatus",
        "Longicaudatus", "Longicollis", "Longifolius", "Longirostris", "Luctuosus", "Luminosus", "Lupus", "Luteus",
        "Maculatus", "Madagascariensis", "Magnus", "Major", "Marginatus", "Maritimus", "Maximus", "Megacephalus",
        "Melanocephalus", "Meridionalis", "Microphyllus", "Minimus", "Minor", "Monspeliensis", "Montanus", "Muralis",
        "Nanus", "Natans", "Nipponensis", "Nitidus", "Nivalis", "Niveus", "Norvegicus", "Obscurus",
        "Obsoletus", "Occidentalis", "Oceanicus", "Officinalis", "Oleraceus", "Orientalis", "Ovatus", "Palustris",
        "Paradoxa", "Parviflorus", "Parvifolius", "Parvus", "Pelagicus", "Petrophilus", "Pictus", "Plumosus",
        "Ponticus", "Praecox", "Pratensis", "Princeps", "Pruinosus", "Pumilus", "Punctatus", "Purpureus",
    ];

    pub fn generate_agent_name(agent: &crate::Agent) -> String {
        let body_count = (agent.body_count as usize).min(crate::MAX_BODY_PARTS);
        if body_count == 0 {
            return "Anonimus-Nullus".to_string();
        }

        // Extract encoded base types in order.
        let mut sequence: Vec<u32> = Vec::with_capacity(body_count);
        for i in 0..body_count {
            sequence.push(agent.body[i].base_type());
        }

        // Genus: hash of unique sorted organ types (base_type >= 20).
        let mut unique_organs: Vec<u32> = sequence.iter().copied().filter(|&t| t >= 20).collect();
        unique_organs.sort_unstable();
        unique_organs.dedup();

        let mut genus_hash = fnv1a64_init();
        genus_hash = fnv1a64_add_u32(genus_hash, unique_organs.len() as u32);
        for t in &unique_organs {
            genus_hash = fnv1a64_add_u32(genus_hash, *t);
        }
        let genus_index = (genus_hash as usize) % AGENT_GENUS_WORDS.len();
        let genus = AGENT_GENUS_WORDS[genus_index];

        // Species: hash of full ordered sequence + counts per type.
        let mut counts = [0u32; crate::PART_TYPE_COUNT];
        for &t in &sequence {
            if (t as usize) < crate::PART_TYPE_COUNT {
                counts[t as usize] += 1;
            }
        }

        let mut species_hash = fnv1a64_init();
        for &c in &counts {
            species_hash = fnv1a64_add_u32(species_hash, c);
        }
        for &t in &sequence {
            species_hash = fnv1a64_add_u32(species_hash, t);
        }
        species_hash = fnv1a64_add_u32(species_hash, body_count as u32);

        let species_index = ((species_hash >> 32) as usize) % AGENT_SPECIES_WORDS.len();
        let species = AGENT_SPECIES_WORDS[species_index];

        format!("{}-{}", genus, species)
    }
}
