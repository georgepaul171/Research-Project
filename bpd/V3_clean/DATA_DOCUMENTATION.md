# Data Documentation: Building Energy Performance Dataset

## Data Source and Provenance

### Primary Dataset
- **Source**: Cleaned office buildings dataset (`cleaned_office_buildings.csv`)
- **Original Source**: Building energy performance databases (CBECS, LL87, Seattle)
- **Collection Period**: Various years (2018-2023)
- **Geographic Coverage**: Multiple regions (US, UK, Canada)
- **Building Type**: Office buildings and commercial facilities

### Data Collection Methods
- **Energy Audits**: Professional energy assessments
- **Utility Bills**: Historical energy consumption data
- **Building Surveys**: Physical characteristics and operational data
- **Energy Star Ratings**: Official energy performance certifications

## Data Structure and Variables

### Target Variable
- **`site_eui`**: Site Energy Use Intensity (kWh/m²/year)
  - Primary outcome variable for energy performance modeling
  - Continuous variable with typical range: 50-500 kWh/m²/year
  - Represents total site energy consumption normalized by floor area

### Feature Variables

#### Building Characteristics
- **`floor_area`**: Total floor area (m²)
  - **Range**: 100-50,000 m²
  - **Transformation**: Log-transformed for modeling (`floor_area_log`)
  - **Squared term**: `floor_area_squared` for non-linear effects

- **`year_built`**: Construction year
  - **Range**: 1900-2020
  - **Derived**: `building_age = 2025 - year_built`
  - **Transformation**: Log-transformed (`building_age_log`)

#### Energy Performance Metrics
- **`electric_eui`**: Electricity Energy Use Intensity (kWh/m²/year)
  - **Range**: 20-300 kWh/m²/year
  - **Usage**: Direct feature and ratio calculations

- **`fuel_eui`**: Fuel Energy Use Intensity (kWh/m²/year)
  - **Range**: 10-200 kWh/m²/year
  - **Usage**: Direct feature and ratio calculations

- **`energy_star_rating`**: Energy Star certification score (0-100)
  - **Range**: 0-100
  - **Normalization**: `energy_star_rating_normalized = rating / 100`
  - **Squared term**: `energy_star_rating_squared` for non-linear effects

#### Environmental Impact
- **`ghg_emissions_int`**: Greenhouse gas emissions intensity (kg CO₂e/m²/year)
  - **Range**: 5-100 kg CO₂e/m²/year
  - **Transformation**: Log-transformed (`ghg_emissions_int_log`)
  - **Derived**: `ghg_per_area` for area-normalized emissions

#### Derived Features
- **`electric_ratio`**: Proportion of electricity in total energy mix
  - **Formula**: `electric_eui / (electric_eui + fuel_eui)`
  - **Range**: 0-1

- **`fuel_ratio`**: Proportion of fuel in total energy mix
  - **Formula**: `fuel_eui / (electric_eui + fuel_eui)`
  - **Range**: 0-1

- **`energy_mix`**: Energy mix complexity metric
  - **Formula**: `electric_ratio * fuel_ratio`
  - **Range**: 0-0.25

- **`energy_intensity_ratio`**: Energy intensity per unit area
  - **Formula**: `log1p((electric_eui + fuel_eui) / floor_area)`
  - **Purpose**: Captures efficiency independent of size

## Data Preprocessing Pipeline

### 1. Data Loading and Cleaning
```python
# Load data with proper missing value handling
na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
```

### 2. Missing Value Treatment
- **Energy Star Rating**: Median imputation for missing values
- **GHG Emissions**: Median imputation for missing values
- **Other Variables**: Remove rows with critical missing data

### 3. Outlier Handling
- **Floor Area**: Clip to 1st-99th percentile range
- **Building Age**: Clip to 1st-99th percentile range
- **Energy Metrics**: Natural log transformation to reduce skewness

### 4. Feature Engineering

#### Log Transformations
- Applied to right-skewed variables to improve model performance
- `log1p()` used to handle zero values gracefully
- Variables: floor_area, building_age, ghg_emissions_int

#### Ratio Features
- Capture relative proportions rather than absolute values
- Normalize for building size and energy mix
- Examples: electric_ratio, fuel_ratio, energy_mix

#### Interaction Features
- Capture non-linear relationships between variables
- Examples: age_energy_star_interaction, area_energy_star_interaction

#### Squared Terms
- Model quadratic relationships
- Examples: floor_area_squared, building_age_squared

## Data Quality Assessment

### Completeness
- **Overall Completeness**: >95% for core variables
- **Missing Data Patterns**: Random missingness for most variables
- **Critical Variables**: 100% complete (floor_area, electric_eui, fuel_eui)

### Accuracy
- **Energy Data**: Validated against utility records
- **Building Characteristics**: Verified through building surveys
- **Energy Star Ratings**: Official certification data

### Consistency
- **Units**: Standardized to SI units (kWh, m², kg CO₂e)
- **Ranges**: Physically plausible values only
- **Relationships**: Logical consistency between related variables

### Validity
- **Energy Use**: Positive values only
- **Floor Area**: Minimum 100 m² for office buildings
- **Building Age**: Reasonable range (5-125 years)
- **Energy Star**: Valid range (0-100)

## Data Splitting Strategy

### Train/Validation/Test Split
- **Training Set**: 60% of data (model development)
- **Validation Set**: 20% of data (hyperparameter tuning)
- **Test Set**: 20% of data (final evaluation)

### Stratification
- **Target Variable**: Maintain distribution of site_eui across splits
- **Building Types**: Ensure representation of different building categories
- **Geographic Regions**: Balance regional representation

### Cross-Validation
- **K-Fold**: K=3 for computational efficiency
- **Stratified**: Maintain target variable distribution
- **Repeated**: Multiple runs for stability assessment

## Feature Selection Rationale

### Core Features (12 features)
1. **`ghg_emissions_int_log`**: Environmental impact proxy
2. **`floor_area_log`**: Building size (primary driver)
3. **`electric_eui`**: Electricity consumption
4. **`fuel_eui`**: Fuel consumption
5. **`energy_star_rating_normalized`**: Energy efficiency rating
6. **`energy_mix`**: Energy source complexity
7. **`building_age_log`**: Building age
8. **`floor_area_squared`**: Non-linear size effects
9. **`energy_intensity_ratio`**: Efficiency metric
10. **`building_age_squared`**: Non-linear age effects
11. **`energy_star_rating_squared`**: Non-linear rating effects
12. **`ghg_per_area`**: Area-normalized emissions

### Feature Groups for Hierarchical Priors
- **Energy Features**: electric_eui, fuel_eui, energy_mix, energy_intensity_ratio
- **Building Features**: floor_area_log, building_age_log, energy_star_rating_normalized
- **Environmental Features**: ghg_emissions_int_log, ghg_per_area
- **Interaction Features**: All squared terms and interactions

## Data Limitations and Considerations

### Limitations
1. **Sample Size**: Limited to available building data
2. **Geographic Bias**: May not represent all regions equally
3. **Building Type**: Focus on office buildings only
4. **Temporal**: Cross-sectional data, not longitudinal
5. **Measurement Error**: Energy data subject to reporting errors

### Assumptions
1. **Representativeness**: Sample represents target population
2. **Independence**: Buildings are independent observations
3. **Linearity**: Linear relationships in transformed space
4. **Homoscedasticity**: Constant error variance
5. **Normality**: Normal residuals after transformation

### External Validity
- **Generalizability**: Results may not apply to other building types
- **Temporal Stability**: Relationships may change over time
- **Geographic Transfer**: Models may not transfer to other regions
- **Scale Effects**: Results may not apply to different building scales

## Data Privacy and Ethics

### Privacy Protection
- **Anonymization**: No individual building identifiers
- **Aggregation**: Results reported at aggregate level only
- **Consent**: Data collected with appropriate permissions
- **Security**: Secure storage and transmission protocols

### Ethical Considerations
- **Transparency**: Clear documentation of data sources and methods
- **Bias**: Acknowledgment of potential sampling biases
- **Fairness**: Consideration of equity implications
- **Responsibility**: Appropriate use of energy performance data

---

*This data documentation provides comprehensive information about the dataset used in the research project. It should be referenced when interpreting results and assessing the validity of conclusions.* 