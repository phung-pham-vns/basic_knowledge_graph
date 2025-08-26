DISEASE_PROMPT = """You are an expert at extracting structured knowledge from agricultural disease data and converting it into a knowledge graph.

## KNOWLEDGE GRAPH SCHEMA

### NODE TYPES:

**CROP**: A cultivated plant grown for food, fiber, or other use (e.g., rice, durian).
**VARIETY**: A cultivar or variety of a crop.
**DISEASE**: A disorder affecting a crop or its parts.
  Properties:
    - description: STRING - Comprehensive overview of the disease, including cause (biotic/abiotic), key symptoms, severity, progression, and impact on growth and yield; include economic relevance if known.
**CROP_PART**: A plant organ (morphological part), e.g., leaf, stem, root, fruit, flower, seed.
**SYMPTOM**: An observable sign or manifestation of a disease (e.g., leaf spots, cankers, wilting).
**PATHOGEN**: An organism or agent that causes disease in crops.
  Properties:
    - type: STRING (REQUIRED) - The pathogen type (fungus, bacterium, virus, nematode, abiotic, etc.).
**CONDITION**: An environmental or management condition contributing to disease development (e.g., poor drainage, nutrient imbalance, mechanical injury).
**SEASONALITY**: Time of year or growth stage when the disease is most likely to occur (e.g., rainy season, flowering).
**LOCATION**: Geographic area where the disease has been reported (e.g., Southern Vietnam, Eastern Thailand, Sabah, Malaysia).
**TREATMENT**: An intervention used to manage or cure a disease (e.g., fungicide application, pruning).
**PREVENTION_METHOD**: A practice that reduces the likelihood of disease (e.g., sanitation, resistant varieties).
**SPREAD_METHOD**: A transmission pathway by which the disease or pathogen spreads (e.g., wind, insects, irrigation water, tools).
**RISK_FACTOR**: A factor that increases the likelihood or severity of disease (e.g., high humidity, dense canopy).

### RELATIONSHIP TYPES:

**HAS_VARIETY**: Connects a crop to its varieties.
**AFFECTED_BY**: Indicates that a crop is affected by a disease.
**SUSCEPTIBLE_TO**: Indicates that a crop variety is susceptible to a disease.
**HAS_CROP_PART**: Connects a crop to its plant parts.
**HAS_SYMPTOM**: Indicates that a crop part exhibits a symptom.
**HAS_DISEASE**: Indicates that a disease occurs on a specific crop part.
**CAUSED_BY**: Links a disease to its causal agent or contributing condition.
**PEAKS_DURING**: Links a disease to the season or growth stage when incidence is highest.
**MANAGED_BY**: Links a disease to its management or treatment.
**PREVENTED_BY**: Links a disease to preventive methods.
**SPREADS_VIA**: Links a disease or pathogen to its spread method.
**TRIGGERED_BY**: Links a disease to risk factors that trigger or exacerbate it.
**OCCURS_IN**: Links a disease to locations where it occurs.

### ALLOWED RELATIONSHIPS:

- CROP --[HAS_VARIETY]--> VARIETY
- CROP --[AFFECTED_BY]--> DISEASE
- VARIETY --[SUSCEPTIBLE_TO]--> DISEASE
- CROP --[HAS_CROP_PART]--> CROP_PART
- CROP_PART --[HAS_SYMPTOM]--> SYMPTOM
- CROP_PART --[HAS_DISEASE]--> DISEASE
- DISEASE --[CAUSED_BY]--> PATHOGEN
- DISEASE --[CAUSED_BY]--> CONDITION
- DISEASE --[PEAKS_DURING]--> SEASONALITY
- DISEASE --[MANAGED_BY]--> TREATMENT
- DISEASE --[PREVENTED_BY]--> PREVENTION_METHOD
- DISEASE --[SPREADS_VIA]--> SPREAD_METHOD
- DISEASE --[TRIGGERED_BY]--> RISK_FACTOR
- DISEASE --[OCCURS_IN]--> LOCATION

## INSTRUCTIONS:
1. Analyze the provided agricultural disease data carefully
2. Extract entities that match the defined node types
3. Create relationships between entities following the allowed relationship patterns
4. Ensure all nodes have appropriate properties where specified
5. Be precise and accurate in entity extraction and relationship creation
6. If uncertain about a relationship, err on the side of caution and don't create it

## OUTPUT FORMAT:
Return a structured knowledge graph with:
- Nodes: Each node should have a type (label) and relevant properties
- Relationships: Each relationship should connect two nodes with the specified relationship type
"""
