# Unified Metrics Framework (12 Categories)

1. Temperature  
2. Pressure  
3. Flow  
4. Vibration  
5. Current  
6. Voltage  
7. Speed / RPM  
8. Torque  
9. Displacement  
10. Acoustic  
11. Chemical Concentration  
12. Misc Process Variables (Levels, Valves, Setpoints, Flags, Modes)


# Master Unified Conversion Table (All Datasets)

| Unified Metric          | Included Sensors (Datasets)                                                                 | Unified Unit | Conversion                                |
|-------------------------|----------------------------------------------------------------------------------------------|--------------|---------------------------------------------|
| Temperature             | Pump, Azure, AI4I, Turbofan, TEP, Bearing, PRONOSTIA                                        | °C           | K→°C: x - 273.15; °F→°C: (x-32)*(5/9)       |
| Pressure                | Pump, Azure, Turbofan, TEP                                                                   | kPa          | psi→kPa: x * 6.89476; bar→kPa: x * 100       |
| Flow                    | Pump, Azure, TEP, Turbofan                                                                   | L/min        | m³/h → L/min: (x * 1000) / 60               |
| Vibration               | Pump, Bearings, PRONOSTIA                                                                    | g            | m/s² → g: x / 9.80665                        |
| Current                 | Pump, Azure                                                                                  | A            | mA → A: x / 1000                             |
| Voltage                 | Pump                                                                                         | V            | mV → V: x / 1000                             |
| Speed (RPM)             | Pump, Azure, AI4I, Turbofan, PRONOSTIA                                                      | RPM          | Hz → RPM: x * 60                             |
| Torque                  | Azure, AI4I                                                                                  | N·m          | lb-ft → N·m: x * 1.35582                     |
| Displacement            | AI4I, PRONOSTIA                                                                              | mm/min/etc   | n/a                                         |
| Acoustic                | Pump                                                                                         | dB           | n/a                                         |
| Chemical Concentration  | TEP                                                                                          | ppm          | fraction → ppm: x * 1,000,000                |
| Misc Process Variables  | TEP, Azure, PRONOSTIA (levels, valves, setpoints, categorical flags, operation modes)       | N/A          | n/a                                         |

# Dataset-by-Dataset Conversion Maps

## 1. Pump Sensor Dataset

| Column         | Description              | Raw Units | Unified Metric | Unified Units | Conversion                       |
|----------------|--------------------------|-----------|----------------|---------------|-----------------------------------|
| pump_temp      | Pump temperature         | °C        | Temperature    | °C            | none                              |
| pump_pressure  | Internal pressure        | kPa/psi   | Pressure       | kPa           | psi→kPa: x * 6.89476              |
| flow_rate      | Flow rate                | L/min/m³h | Flow           | L/min         | m³/h→L/min: (x * 1000) / 60       |
| vibration_x    | Vibration X-axis         | g/m/s²    | Vibration      | g             | m/s²→g: x / 9.80665               |
| vibration_y    | Vibration Y-axis         | g/m/s²    | Vibration      | g             | m/s²→g: x / 9.80665               |
| sound_db       | Acoustic noise           | dB        | Acoustic       | dB            | none                              |
| pump_current   | Motor current            | A         | Current        | A             | none                              |
| pump_voltage   | Motor voltage            | V         | Voltage        | V             | none                              |
| status_flag    | Pump state               | 0/1       | Misc           | N/A           | none                              |

## 2. Microsoft Azure Predictive Maintenance Dataset

| Column             | Description                | Raw Units | Unified Metric | Unified Units | Conversion                     |
|--------------------|----------------------------|-----------|----------------|---------------|----------------------------------|
| hydraulic_pressure | System pressure            | psi       | Pressure       | kPa           | psi→kPa                          |
| hydraulic_temp     | Hydraulic temperature      | °C        | Temperature    | °C            | none                             |
| cooler_temp        | Cooler outlet temperature  | °C        | Temperature    | °C            | none                             |
| vibration          | Machine vibration          | g/m/s²    | Vibration      | g             | m/s²→g                           |
| flow_rate          | Hydraulic flow             | L/min     | Flow           | L/min         | none                             |
| eff_factor         | Cooling efficiency (%)     | %         | Misc           | fraction      | %→fraction: x/100                |
| machine_speed      | Rotational speed           | RPM       | Speed          | RPM           | none                             |
| torque_load        | Load torque                | N·m       | Torque         | N·m           | none                             |
| motor_current      | Current                    | A         | Current        | A             | none                             |

## 3. AI4I 2020 Predictive Maintenance Dataset

| Column             | Description              | Raw Units | Unified Metric | Unified Units | Conversion              |
|--------------------|--------------------------|-----------|----------------|---------------|-------------------------|
| air_temperature    | Ambient temp             | K         | Temperature    | °C            | x - 273.15              |
| process_temperature| Internal temp            | K         | Temperature    | °C            | x - 273.15              |
| rotational_speed   | Rotational speed         | RPM       | Speed          | RPM           | none                    |
| torque             | Torque applied           | N·m       | Torque         | N·m           | none                    |
| tool_wear          | Wear (minutes)           | min       | Displacement   | minutes       | none                    |
| machine_failure    | Failure label            | 0/1       | Misc           | N/A           | none                    |
| type               | Product type             | category  | Misc           | N/A           | one-hot encode          |


## 4. NASA Turbofan Jet Engine (C-MAPSS)

| Sensor       | Description                     | Raw Units | Unified Metric | Unified Units | Conversion          |
|--------------|----------------------------------|-----------|----------------|---------------|----------------------|
| T2,T24,T30   | Turbine temperatures             | K/°R      | Temperature    | °C            | convert to °C        |
| P2,P15,P30   | Pressures                        | psi       | Pressure       | kPa           | psi→kPa              |
| Nf           | Fan speed                        | RPM       | Speed          | RPM           | none                 |
| Nc           | Core speed                       | RPM       | Speed          | RPM           | none                 |
| EGT          | Exhaust gas temperature          | K         | Temperature    | °C            | K→°C                 |
| Fuel_flow    | Fuel flow rate                   | lb/s      | Flow           | standard unit | convert for uniformity|
| HPC_speed    | Compressor speed                 | RPM       | Speed          | RPM           | none                 |
| Vibration    | (if included in variant)         | g         | Vibration      | g             | none                 |

## 5. TEP (Tennessee Eastman Process)

| Variable Group    | Example Variables       | Unified Metric         | Units   | Conversion        |
|-------------------|--------------------------|------------------------|---------|--------------------|
| Temperatures      | ReactorTemp, SepTemp     | Temperature            | °C      | none              |
| Pressures         | ReactorPressure, SepP    | Pressure               | kPa/Pa  | Pa→kPa: x/1000    |
| Flows             | FeedFlow, ProductFlow    | Flow                   | various | normalize→L/min   |
| Concentrations    | C1, C2, CompositionVars  | Chemical Concentration | fraction| fraction→ppm       |
| Levels            | Level1, Level2           | Misc Process           | % or 0-1 | %→fraction       |
| Valve positions   | Valve1,Valve2            | Misc Process           | %       | %→fraction        |
| Setpoints         | SP_Temp, SP_Pressure     | Misc Process           | various | leave as-is        |

## 6. NASA Bearing Dataset (IMS/Case Western)

| Column           | Description               | Raw Units | Unified Metric | Unified Units | Conversion              |
|------------------|----------------------------|-----------|----------------|---------------|--------------------------|
| accel_x          | Vibration (X-axis)         | m/s²/g/V  | Vibration      | g             | m/s²→g: x/9.80665        |
| accel_y          | Vibration (Y-axis)         | m/s²/g/V  | Vibration      | g             | m/s²→g                   |
| bearing_temp     | Bearing temperature        | °C        | Temperature    | °C            | none                     |
| rpm              | Shaft speed                | RPM       | Speed          | RPM           | none                     |
| load             | Applied load               | N/lb      | Misc Process   | N             | lb→N: x*4.44822          |

## 7. PRONOSTIA Ball Bearing Run-to-Failure Dataset

| Column          | Description                 | Raw Units | Unified Metric | Unified Units | Conversion              |
|-----------------|------------------------------|-----------|----------------|---------------|--------------------------|
| acc_x           | Vibration (X-axis)           | m/s²/g    | Vibration      | g             | m/s²→g: x/9.80665        |
| acc_y           | Vibration (Y-axis)           | m/s²/g    | Vibration      | g             | m/s²→g                   |
| rotation_speed  | Rotational speed             | RPM       | Speed          | RPM           | none                     |
| load            | Bearing load                 | N         | Misc Process   | N             | none                     |
| temperature     | Bearing temperature          | °C        | Temperature    | °C            | none                     |


# Unified Conversion Calculations
All datasets are standardized into 12 unified sensor metrics.
Below are the official conversion formulas for each metric type.

## 1. Temperature Conversions
### Accepted raw units:
- Kelvin (K)
- Celsius (°C)
- Fahrenheit (°F)

### Unified unit:
- **°C**

### Formulas:
- Kelvin → Celsius:
    ```
    °C = K - 273.15
    ```
- Fahrenheit → Celsius:
    ```
    °C = (°F - 32) * (5/9)
    ```
--- 

## 2. Pressure Conversions
### Accepted raw units:
- psi
- bar
- Pa
- kPa
- MPa

### Unified unit:
- **kPa**

### Formulas:
- psi → kPa:
    ```
    kPa = psi * 6.89476
    ```

- bar → kPa:
    ```
    kPa = bar * 100
    ```

- Pa → kPa:
    ```
    kPa = Pa / 1000
    ```


- MPa → kPa:
    ```
    kPa = MPa * 1000
    ```


---

## 3. Flow Conversions
### Accepted raw units:
- L/min
- m³/h
- GPM (US gallons per minute)
- kg/s or lb/s (mass flow)

### Unified unit:
- **L/min**

### Formulas:
- m³/h → L/min:
    ```
    L/min = (m³/h * 1000) / 60
    ```

- GPM → L/min:
    ```
    L/min = GPM * 3.78541
    ```

- lb/s or kg/s → L/min:
*(Only apply if converting volumetric equivalent)*
- Depends on fluid density. Example for water:
  ```
  L/min = (kg/s / 1.0) * 60
  ```

---

## 4. Vibration Conversions
### Accepted raw units:
- g (gravitational acceleration)
- m/s² (SI acceleration)
- raw voltage (accelerometer output in Case Western dataset)

### Unified unit:
- **g** (gravitational acceleration)

### Formulas:
- m/s² → g:
    ```
    g = (m/s²) / 9.80665
    ```

- raw voltage (Case Western dataset)::
>  No physical unit conversion is applied.
> The raw voltage signal is typically normalized or scaled prior to feature extraction
> (e.g., FFT features, RMS, kurtosis). The conversion into g cannot be performed
> without accelerometer sensitivity calibration data, which is not provided in this dataset.

---

## 5. Current Conversions
### Accepted raw units:
- A
- mA

### Unified unit:
- **A**

### Formulas:
- mA → A:
    ```
    A = mA / 1000
    ```

---

## 6. Voltage Conversions
### Accepted raw units:
- V
- mV

### Unified unit:
- **V**

### Formula:
- mV → V:
    ```
    V = mV / 1000
    ```

---

## 7. Speed (RPM) Conversions
### Accepted raw units:
- RPM
- Hz (cycles per second)

### Unified unit:
- **RPM**

### Formula:
- Hz → RPM:
    ```
    RPM = Hz * 60
    ```

---

## 8. Torque Conversions
### Accepted raw units:
- N·m
- lb-ft

### Unified unit:
- **N·m**

### Formula:
- lb-ft → N·m:
    ```
    N·m = lb-ft * 1.35582
    ```

---

## 9. Displacement Conversions
### Accepted raw units:
- mm
- cm
- inches
- micrometers

### Unified unit:
- **mm**

### Formulas:
- cm → mm:
    ```
    mm = cm * 10
    ```

- inches → mm:
    ```
    mm = inches * 25.4
    ```

- micrometers → mm:
    ```
    mm = micrometers / 1000
    ```

---

## 10. Acoustic (Sound Level)
### Accepted raw units:
- dB
- RMS voltage

### Unified unit:
- **dB**

### Formula:
- RMS amplitude → dB:
    ```
    dB = 20 * log10(RMS)
    ```

---

## 11. Chemical Concentration Conversions
### Accepted raw units:
- fraction (0–1)
- %
- ppm

### Unified unit:
- **ppm**

### Formulas:
- % → ppm:
    ```
    ppm = % * 10,000
    ```

- fraction (0–1) → ppm:
    ```
    ppm = fraction * 1,000,000
    ```

---

## 12. Misc Process Variables
### Includes:
- valve positions (0–100%)
- levels (0–1 or %)
- system states
- categorical sensors
- setpoints

### Unified unit:
- **As-is (normalized if needed)**

### Formula:
- % → fraction:
    ```
    fraction = % / 100
    ```

--- 

# EML Master Grouping Table (All Datasets)

## 1. Pump Sensor Dataset → EML Mapping

### Pump Dataset → Unified Metrics → EML Groups
| Column         | Unified Metric | EML Group |
|----------------|----------------|-----------|
| pump_temp      | Temperature    | temp      |
| pump_pressure  | Pressure       | pressure  |
| flow_rate      | Flow           | flow      |
| vibration_x    | Vibration      | vib_x     |
| vibration_y    | Vibration      | vib_y     |
| sound_db       | Acoustic       | acoustic  |
| pump_current   | Current        | current   |
| pump_voltage   | Voltage        | voltage   |
| status_flag    | Misc           | EXCLUDED (categorical flag) |

## 2. Microsoft Azure Predictive Maintenance Dataset → EML Mapping
### Azure PM Dataset → Unified Metrics → EML Groups
| Column             | Unified Metric | EML Group |
|--------------------|----------------|-----------|
| hydraulic_pressure | Pressure       | pressure  |
| hydraulic_temp     | Temperature    | temp      |
| cooler_temp        | Temperature    | temp      |
| vibration          | Vibration      | vibration |
| flow_rate          | Flow           | flow      |
| eff_factor (%)     | Misc Process   | misc      |
| machine_speed      | Speed/RPM      | rpm       |
| torque_load        | Torque         | torque    |
| motor_current      | Current        | current   |

## 3. AI4I 2020 → EML Mapping
### AI4I 2020 → Unified Metrics → EML Groups
| Column              | Unified Metric | EML Group |
|---------------------|----------------|-----------|
| air_temperature     | Temperature    | temp      |
| process_temperature | Temperature    | temp      |
| rotational_speed    | Speed/RPM      | rpm       |
| torque              | Torque         | torque    |
| tool_wear           | Displacement   | wear      |
| machine_failure     | EXCLUDED       | Label     |
| type                | EXCLUDED       | Category  |

## 4. NASA Turbofan Jet Engine Dataset → EML Mapping
### Turbofan (C-MAPSS) → Unified Metrics → EML Groups
| Sensor | Unified Metric | EML Group |
|--------|----------------|-----------|
| T2,T24,T30,T50,EGT | Temperature | temp |
| P2,P15,P30         | Pressure    | pressure |
| Nf, Nc             | Speed/RPM   | rpm |
| Fuel_flow          | Flow        | flow |
| HPC_speed          | Speed/RPM   | rpm |
| Vibration*         | Vibration   | vibration |
| Operational Settings| EXCLUDED   | metadata |

## 5. Tennessee Eastman Process (TEP) → EML Mapping
### TEP → Unified Metrics → EML Groups
| Variable Type      | Example Columns          | Unified Metric         | EML Group |
|--------------------|---------------------------|------------------------|-----------|
| Temperatures       | ReactorTemp, SepTemp      | Temperature            | temp      |
| Pressures          | ReactorPressure, FeedP    | Pressure               | pressure  |
| Flows              | FeedFlow, PurgeFlow       | Flow                   | flow      |
| Concentrations     | Component_x, Purity       | Chemical               | chemical  |
| Levels             | Level1, Level2            | Misc Process           | misc      |
| Valve Positions    | Valve1, Valve2            | Misc Process           | misc      |
| Setpoints          | SP_Temp, SP_Pressure      | Misc Process           | misc_set  |
| Fault Labels       | FaultNum                  | EXCLUDED (label)       | label     |


## 6. NASA Bearing Dataset (IMS / Case Western) → EML Mapping

### Bearing Dataset → Unified Metrics → EML Groups
| Column           | Unified Metric | EML Group |
|------------------|----------------|-----------|
| accel_x          | Vibration      | vib_x     |
| accel_y          | Vibration      | vib_y     |
| bearing_temp     | Temperature    | temp      |
| rpm              | Speed/RPM      | rpm       |
| load             | Misc Process   | misc      |


## 7. PRONOSTIA Ball Bearing Run-to-Failure → EML Mapping

### PRONOSTIA → Unified Metrics → EML Groups
| Column          | Unified Metric | EML Group |
|-----------------|----------------|-----------|
| acc_x           | Vibration      | vib_x     |
| acc_y           | Vibration      | vib_y     |
| rotation_speed  | Speed/RPM      | rpm       |
| load            | Misc Process   | misc      |
| temperature     | Temperature    | temp      |


