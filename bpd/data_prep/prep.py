import pandas as pd

df = pd.read_csv('/Users/georgepaul/Desktop/bpd_all.csv')

# Filter for New York City buildings
df_nyc = df[df['city'] == 'New York']

# Convert relevant columns to numeric
numeric_columns = ['floor_area', 'electric_eui', 'fuel_eui', 'site_eui', 'source_eui', 'ghg_emissions_int']
for col in numeric_columns:
    df_nyc[col] = pd.to_numeric(df_nyc[col], errors='coerce')

# Display summary statistics for key metrics
print("\nSummary Statistics for NYC Buildings:")
print("\nFloor Area (sq ft):")
print(df_nyc['floor_area'].describe())
print("\nSite EUI (kBtu/sq ft):")
print(df_nyc['site_eui'].describe())
print("\nSource EUI (kBtu/sq ft):")
print(df_nyc['source_eui'].describe())
print("\nGHG Emissions Intensity (kgCO2e/sq ft):")
print(df_nyc['ghg_emissions_int'].describe())

# Display count of buildings
print("\nNumber of NYC buildings in dataset:", len(df_nyc))

# Display distribution of building types
print("\nTop 10 Building Types:")
print(df_nyc['facility_type'].value_counts().head(10))

# Now filter for office buildings
df_office = df_nyc[df_nyc['facility_type'].str.contains('Office', case=False, na=False)]

print("\n\n=== OFFICE BUILDINGS STATISTICS ===")
print("\nNumber of Office Buildings:", len(df_office))
print("\nSummary Statistics for Office Buildings:")
print("\nFloor Area (sq ft):")
print(df_office['floor_area'].describe())
print("\nSite EUI (kBtu/sq ft):")
print(df_office['site_eui'].describe())
print("\nSource EUI (kBtu/sq ft):")
print(df_office['source_eui'].describe())
print("\nGHG Emissions Intensity (kgCO2e/sq ft):")
print(df_office['ghg_emissions_int'].describe())

print("\nFirst 5 Office Buildings:")
print(df_office.head())

# Check unique building classes for office buildings
print("\nUnique Building Classes for Office Buildings:")
print(df_office['building_class'].unique())

# Filter for commercial office buildings
df_commercial_office = df_office[df_office['building_class'].str.contains('Commercial', case=False, na=False)]

print("\n\n=== COMMERCIAL OFFICE BUILDINGS STATISTICS ===")
print("\nNumber of Commercial Office Buildings:", len(df_commercial_office))
print("\nSummary Statistics for Commercial Office Buildings:")
print("\nFloor Area (sq ft):")
print(df_commercial_office['floor_area'].describe())
print("\nSite EUI (kBtu/sq ft):")
print(df_commercial_office['site_eui'].describe())
print("\nSource EUI (kBtu/sq ft):")
print(df_commercial_office['source_eui'].describe())
print("\nGHG Emissions Intensity (kgCO2e/sq ft):")
print(df_commercial_office['ghg_emissions_int'].describe())

print("\nFirst 5 Commercial Office Buildings:")
print(df_commercial_office.head())