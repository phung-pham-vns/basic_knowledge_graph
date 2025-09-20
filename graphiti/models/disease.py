from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# Custom Entity Types for Disease Knowledge Graph
class Crop(BaseModel):
    """A cultivated plant grown for food, fiber, or other use (e.g., rice, durian)."""

    scientific_name: Optional[str] = Field(None, description="Scientific name of the crop")
    common_name: Optional[str] = Field(None, description="Common name of the crop")
    family: Optional[str] = Field(None, description="Botanical family of the crop")
    origin: Optional[str] = Field(None, description="Geographic origin of the crop")


class Variety(BaseModel):
    """A cultivar or variety of a crop."""

    cultivar_name: Optional[str] = Field(None, description="Name of the cultivar or variety")
    breeding_year: Optional[int] = Field(None, description="Year when variety was developed")
    characteristics: Optional[str] = Field(None, description="Key characteristics of the variety")
    resistance_traits: Optional[str] = Field(None, description="Disease or pest resistance traits")


class Disease(BaseModel):
    """A disorder affecting a crop or its parts."""

    description: Optional[str] = Field(
        None,
        description="Comprehensive overview of the disease, including cause (biotic/abiotic), key symptoms, severity, progression, and impact on growth and yield; include economic relevance if known",
    )
    severity: Optional[str] = Field(None, description="Disease severity level (mild, moderate, severe)")
    economic_impact: Optional[str] = Field(None, description="Economic impact of the disease")
    first_reported: Optional[datetime] = Field(None, description="Date when disease was first reported")


class CropPart(BaseModel):
    """A plant organ or morphological part (e.g., leaf, stem, root, fruit, flower, seed)."""

    anatomical_type: Optional[str] = Field(None, description="Type of plant organ (vegetative, reproductive)")
    function: Optional[str] = Field(None, description="Primary function of the plant part")
    location: Optional[str] = Field(None, description="Location on the plant")


class Symptom(BaseModel):
    """An observable sign or manifestation of a disease (e.g., leaf spots, cankers, wilting)."""

    appearance: Optional[str] = Field(None, description="Visual appearance of the symptom")
    progression: Optional[str] = Field(None, description="How the symptom develops over time")
    diagnostic_value: Optional[str] = Field(None, description="Diagnostic importance of the symptom")


class Pathogen(BaseModel):
    """An organism or agent that causes disease in crops."""

    type: Optional[str] = Field(
        None, description="The pathogen type (fungus, bacterium, virus, nematode, abiotic, etc.)"
    )
    scientific_name: Optional[str] = Field(None, description="Scientific name of the pathogen")
    taxonomy: Optional[str] = Field(None, description="Taxonomic classification")
    host_range: Optional[str] = Field(None, description="Range of hosts the pathogen can infect")


class Condition(BaseModel):
    """An environmental or management condition contributing to disease development."""

    condition_type: Optional[str] = Field(None, description="Type of condition (environmental, management, etc.)")
    optimal_range: Optional[str] = Field(None, description="Optimal range for disease development")
    measurement_unit: Optional[str] = Field(None, description="Unit of measurement for the condition")


class Seasonality(BaseModel):
    """Time of year when the disease is most likely to occur (e.g., rainy season)."""

    season_name: Optional[str] = Field(None, description="Name of the season")
    months: Optional[str] = Field(None, description="Specific months when disease peaks")
    climate_factors: Optional[str] = Field(None, description="Climate factors during this season")


class Location(BaseModel):
    """Geographic area where the disease has been reported."""

    country: Optional[str] = Field(None, description="Country where disease occurs")
    region: Optional[str] = Field(None, description="Specific region or state")
    coordinates: Optional[str] = Field(None, description="Geographic coordinates if available")
    climate_zone: Optional[str] = Field(None, description="Climate zone classification")


class Treatment(BaseModel):
    """An intervention used to manage or cure a disease."""

    treatment_type: Optional[str] = Field(None, description="Type of treatment (chemical, biological, cultural)")
    active_ingredient: Optional[str] = Field(None, description="Active ingredient in treatment")
    application_method: Optional[str] = Field(None, description="How the treatment is applied")
    effectiveness: Optional[str] = Field(None, description="Effectiveness rating of the treatment")
    cost: Optional[float] = Field(None, description="Cost of treatment per unit area")


class PreventionMethod(BaseModel):
    """A practice that reduces the likelihood of disease."""

    method_type: Optional[str] = Field(None, description="Type of prevention method")
    implementation_difficulty: Optional[str] = Field(None, description="Difficulty level of implementation")
    cost_effectiveness: Optional[str] = Field(None, description="Cost-effectiveness rating")
    sustainability: Optional[str] = Field(None, description="Environmental sustainability rating")


class SpreadMethod(BaseModel):
    """A transmission pathway by which the disease or pathogen spreads."""

    transmission_type: Optional[str] = Field(None, description="Type of transmission (vector, airborne, soil, etc.)")
    transmission_rate: Optional[str] = Field(None, description="Rate of transmission")
    distance_range: Optional[str] = Field(None, description="Distance over which pathogen can spread")
    environmental_factors: Optional[str] = Field(None, description="Environmental factors affecting spread")


class RiskFactor(BaseModel):
    """A factor that increases the likelihood or severity of disease."""

    risk_level: Optional[str] = Field(None, description="Level of risk (low, medium, high)")
    controllability: Optional[str] = Field(None, description="How controllable the risk factor is")
    impact_magnitude: Optional[str] = Field(None, description="Magnitude of impact on disease development")


# Custom Edge Types for Disease Knowledge Graph
class HasVariety(BaseModel):
    """Connects a crop to its varieties."""

    breeding_program: Optional[str] = Field(None, description="Breeding program that developed the variety")
    release_date: Optional[datetime] = Field(None, description="Date when variety was released")
    adaptation_zone: Optional[str] = Field(None, description="Geographic zone where variety is adapted")


class AffectedBy(BaseModel):
    """Indicates that a crop is affected by a disease."""

    susceptibility_level: Optional[str] = Field(None, description="Level of susceptibility (low, medium, high)")
    yield_loss: Optional[float] = Field(None, description="Percentage yield loss caused by disease")
    first_occurrence: Optional[datetime] = Field(None, description="First recorded occurrence of disease on crop")


class SusceptibleTo(BaseModel):
    """Indicates that a crop variety is susceptible to a disease."""

    resistance_level: Optional[str] = Field(None, description="Level of resistance (susceptible, tolerant, resistant)")
    field_performance: Optional[str] = Field(None, description="Performance under disease pressure")
    breeding_notes: Optional[str] = Field(None, description="Notes from breeding programs")


class HasCropPart(BaseModel):
    """Connects a crop to its plant parts."""

    development_stage: Optional[str] = Field(None, description="Development stage when part is present")
    importance: Optional[str] = Field(None, description="Economic or biological importance of the part")


class HasSymptom(BaseModel):
    """Indicates that a crop part exhibits a symptom."""

    symptom_severity: Optional[str] = Field(None, description="Severity of symptom on this part")
    frequency: Optional[str] = Field(None, description="Frequency of symptom occurrence")
    diagnostic_reliability: Optional[str] = Field(None, description="Reliability for diagnosis")


class HasDisease(BaseModel):
    """Indicates that a disease occurs on a specific crop part."""

    infection_pattern: Optional[str] = Field(None, description="Pattern of infection on the plant part")
    damage_level: Optional[str] = Field(None, description="Level of damage to the plant part")


class CausedBy(BaseModel):
    """Links a disease to its causal agent or contributing condition."""

    causal_relationship: Optional[str] = Field(
        None, description="Type of causal relationship (primary, secondary, contributing)"
    )
    evidence_level: Optional[str] = Field(None, description="Level of scientific evidence for causation")


class PeaksDuring(BaseModel):
    """Links a disease to the season or growth stage when incidence is highest."""

    peak_intensity: Optional[str] = Field(None, description="Intensity of disease during peak period")
    duration: Optional[str] = Field(None, description="Duration of peak period")


class ManagedBy(BaseModel):
    """Links a disease to its management or treatment."""

    efficacy: Optional[float] = Field(None, description="Treatment efficacy percentage")
    application_timing: Optional[str] = Field(None, description="Optimal timing for treatment application")
    cost_per_hectare: Optional[float] = Field(None, description="Cost per hectare for treatment")


class PreventedBy(BaseModel):
    """Links a disease to preventive methods."""

    prevention_efficacy: Optional[float] = Field(None, description="Prevention efficacy percentage")
    implementation_cost: Optional[float] = Field(None, description="Cost of implementing prevention method")


class SpreadsVia(BaseModel):
    """Links a disease or pathogen to its spread method."""

    transmission_efficiency: Optional[float] = Field(None, description="Efficiency of transmission via this method")
    environmental_dependence: Optional[str] = Field(None, description="Dependence on environmental conditions")


class TriggeredBy(BaseModel):
    """Links a disease to risk factors that trigger or exacerbate it."""

    trigger_threshold: Optional[str] = Field(None, description="Threshold value that triggers disease")
    response_time: Optional[str] = Field(None, description="Time between trigger and disease manifestation")


class OccursIn(BaseModel):
    """Links a disease to locations where it occurs."""

    occurrence_frequency: Optional[str] = Field(None, description="Frequency of occurrence in this location")
    first_report_date: Optional[datetime] = Field(None, description="Date of first report in this location")
    prevalence: Optional[float] = Field(None, description="Prevalence percentage in this location")


# Entity type mappings for Graphiti
ENTITY_TYPES = {
    "Crop": Crop,
    "Variety": Variety,
    "Disease": Disease,
    "CropPart": CropPart,
    "Symptom": Symptom,
    "Pathogen": Pathogen,
    "Condition": Condition,
    "Seasonality": Seasonality,
    "Location": Location,
    "Treatment": Treatment,
    "PreventionMethod": PreventionMethod,
    "SpreadMethod": SpreadMethod,
    "RiskFactor": RiskFactor,
}

# Edge type mappings for Graphiti
EDGE_TYPES = {
    "HasVariety": HasVariety,
    "AffectedBy": AffectedBy,
    "SusceptibleTo": SusceptibleTo,
    "HasCropPart": HasCropPart,
    "HasSymptom": HasSymptom,
    "HasDisease": HasDisease,
    "CausedBy": CausedBy,
    "PeaksDuring": PeaksDuring,
    "ManagedBy": ManagedBy,
    "PreventedBy": PreventedBy,
    "SpreadsVia": SpreadsVia,
    "TriggeredBy": TriggeredBy,
    "OccursIn": OccursIn,
}

# Edge type mapping - defines which edge types can exist between specific entity type pairs
EDGE_TYPE_MAP = {
    ("Crop", "Variety"): ["HasVariety"],
    ("Crop", "Disease"): ["AffectedBy"],
    ("Variety", "Disease"): ["SusceptibleTo"],
    ("Crop", "CropPart"): ["HasCropPart"],
    ("CropPart", "Symptom"): ["HasSymptom"],
    ("CropPart", "Disease"): ["HasDisease"],
    ("Disease", "Pathogen"): ["CausedBy"],
    ("Disease", "Condition"): ["CausedBy"],
    ("Disease", "Seasonality"): ["PeaksDuring"],
    ("Disease", "Treatment"): ["ManagedBy"],
    ("Disease", "PreventionMethod"): ["PreventedBy"],
    ("Disease", "SpreadMethod"): ["SpreadsVia"],
    ("Disease", "RiskFactor"): ["TriggeredBy"],
    ("Disease", "Location"): ["OccursIn"],
    ("Pathogen", "SpreadMethod"): ["SpreadsVia"],
    # Fallback for any unexpected relationships
    # ("Entity", "Entity"): ["RELATES_TO"],
}