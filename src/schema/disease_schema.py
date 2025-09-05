node_types = [
    {
        "label": "CROP",
        "description": "A cultivated plant grown for food, fiber, or other use (e.g., rice, durian).",
        "properties": [
            {
                "name": "description",
                "type": "STRING",
                "description": "Comprehensive overview of the crop.",
            }
        ],
    },
    {
        "label": "VARIETY",
        "description": "A cultivar or variety of a crop.",
        "properties": [
            {
                "name": "description",
                "type": "STRING",
                "description": "Comprehensive overview of the variety.",
            }
        ],
    },
    {
        "label": "DISEASE",
        "description": "A disorder affecting a crop or its parts.",
        "properties": [
            {
                "name": "description",
                "type": "STRING",
                "description": "Comprehensive overview of the disease, including cause (biotic/abiotic), key symptoms, severity, progression, and impact on growth and yield; include economic relevance if known.",
            }
        ],
    },
    {
        "label": "CROP_PART",
        "description": "A plant organ or morphological part (e.g., leaf, stem, root, fruit, flower, seed). The output must be in singular form, e.g., trunk, branch (not trunks, branches).",
    },
    {
        "label": "SYMPTOM",
        "description": "An observable sign or manifestation of a disease (e.g., leaf spots, cankers, wilting). Each node contains only one such sign or manifestation.",
        "properties": [
            {
                "name": "description",
                "type": "STRING",
                "description": "Comprehensive overview of the symptom.",
            }
        ],
    },
    {
        "label": "PATHOGEN",
        "description": "An organism or agent that causes disease in crops.",
        "properties": [
            {
                "name": "type",
                "type": "STRING",
                "description": "The pathogen type (fungus, bacterium, virus, nematode, abiotic, etc.).",
                "required": True,
            },
        ],
    },
    {
        "label": "CONDITION",
        "description": "An environmental or management condition contributing to disease development (e.g., poor drainage, nutrient imbalance, mechanical injury).",
    },
    {
        "label": "SEASONALITY",
        "description": "Time of year when the disease is most likely to occur (e.g., rainy season).",
    },
    {
        "label": "LOCATION",
        "description": "Geographic area where the disease has been reported (e.g., Southern Vietnam, Eastern Thailand, Sabah, Malaysia).",
    },
    {
        "label": "TREATMENT",
        "description": "An intervention used to manage or cure a disease (e.g., fungicide application, pruning).",
    },
    {
        "label": "PREVENTION_METHOD",
        "description": "A practice that reduces the likelihood of disease (e.g., sanitation, resistant varieties).",
    },
    {
        "label": "SPREAD_METHOD",
        "description": "A transmission pathway by which the disease or pathogen spreads (e.g., wind, insects, irrigation water, tools).",
    },
    {
        "label": "RISK_FACTOR",
        "description": "A factor that increases the likelihood or severity of disease (e.g., high humidity, dense canopy).",
    },
]

# Relation types for the disease knowledge graph
relation_types = [
    {
        "label": "HAS_VARIETY",
        "description": "Connects a crop to its varieties.",
    },
    {
        "label": "AFFECTED_BY",
        "description": "Indicates that a crop is affected by a disease.",
    },
    {
        "label": "SUSCEPTIBLE_TO",
        "description": "Indicates that a crop variety is susceptible to a disease.",
    },
    {
        "label": "HAS_CROP_PART",
        "description": "Connects a crop to its plant parts.",
    },
    {
        "label": "HAS_SYMPTOM",
        "description": "Indicates that a crop part exhibits a symptom.",
    },
    {
        "label": "HAS_DISEASE",
        "description": "Indicates that a disease occurs on a specific crop part.",
    },
    {
        "label": "CAUSED_BY",
        "description": "Links a disease to its causal agent or contributing condition.",
    },
    {
        "label": "PEAKS_DURING",
        "description": "Links a disease to the season or growth stage when incidence is highest.",
    },
    {
        "label": "MANAGED_BY",
        "description": "Links a disease to its management or treatment.",
    },
    {
        "label": "PREVENTED_BY",
        "description": "Links a disease to preventive methods.",
    },
    {
        "label": "SPREADS_VIA",
        "description": "Links a disease or pathogen to its spread method.",
    },
    {
        "label": "TRIGGERED_BY",
        "description": "Links a disease to risk factors that trigger or exacerbate it.",
    },
    {
        "label": "OCCURS_IN",
        "description": "Links a disease to locations where it occurs.",
    },
]

# Allowed relationship patterns (source_node -> relationship -> target_node)
allowed_relationships = [
    ("CROP", "HAS_VARIETY", "VARIETY"),
    ("CROP", "AFFECTED_BY", "DISEASE"),
    ("VARIETY", "SUSCEPTIBLE_TO", "DISEASE"),
    ("CROP", "HAS_CROP_PART", "CROP_PART"),
    ("CROP_PART", "HAS_SYMPTOM", "SYMPTOM"),
    ("CROP_PART", "HAS_DISEASE", "DISEASE"),
    ("DISEASE", "CAUSED_BY", "PATHOGEN"),
    ("DISEASE", "CAUSED_BY", "CONDITION"),
    ("DISEASE", "PEAKS_DURING", "SEASONALITY"),
    ("DISEASE", "MANAGED_BY", "TREATMENT"),
    ("DISEASE", "PREVENTED_BY", "PREVENTION_METHOD"),
    ("DISEASE", "SPREADS_VIA", "SPREAD_METHOD"),
    ("DISEASE", "TRIGGERED_BY", "RISK_FACTOR"),
    ("DISEASE", "OCCURS_IN", "LOCATION"),
]
