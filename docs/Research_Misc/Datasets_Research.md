# Datasets:

    ## Tennessee Eastman Process:
        https://www.kaggle.com/datasets/afrniomelo/tep-csv
        https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset

        Columns: 
            FaultNumber
            simulationRun
            sample 
            xmeas_1 - xmeas_41
            xmv_1-xmv_11


            Continuous Process Measurements

                Variable	Description	unit
                XMEAS(1)	A Feed (stream 1)	kscmh
                XMEAS(2)	D Feed (stream 2)	kg/hr
                XMEAS(3)	E Feed (stream 3)	kg/hr
                XMEAS(4)	A and C Feed (stream 4)	kscmh
                XMEAS(5)	Recycle Flow (stream 8)	kscmh
                XMEAS(6)	Reactor Feed Rate (stream 6)	kscmh
                XMEAS(7)	Reactor Pressure	kPa gauge
                XMEAS(8)	Reactor Level	%
                XMEAS(9)	Reactor Temperature	Deg C
                XMEAS(10)	Purge Rate (stream 9)	kscmh
                XMEAS(11)	Product Sep Temp	Deg C
                XMEAS(12)	Product Sep Level	%
                XMEAS(13)	Prod Sep Pressure	kPa gauge
                XMEAS(14)	Prod Sep Underflow (stream 10)	m3/hr
                XMEAS(15)	Stripper Level	%
                XMEAS(16)	Stripper Pressure	kPa gauge
                XMEAS(17)	Stripper Underflow (stream 11)	m3/hr
                XMEAS(18)	Stripper Temperature	Deg C
                XMEAS(19)	Stripper Steam Flow	kg/hr
                XMEAS(20)	Compressor Work	kW
                XMEAS(21)	Reactor Cooling Water Outlet Temp	Deg C
                XMEAS(22)	Separator Cooling Water Outlet Temp	Deg C

            Manipulated Variables

                Variable	Description
                XMV(1)	D Feed Flow (stream 2) (Corrected Order)
                XMV(2)	E Feed Flow (stream 3) (Corrected Order)
                XMV(3)	A Feed Flow (stream 1) (Corrected Order)
                XMV(4)	A and C Feed Flow (stream 4)
                XMV(5)	Compressor Recycle Valve
                XMV(6)	Purge Valve (stream 9)
                XMV(7)	Separator Pot Liquid Flow (stream 10)
                XMV(8)	Stripper Liquid Product Flow (stream 11)
                XMV(9)	Stripper Steam Valve
                XMV(10)	Reactor Cooling Water Flow
                XMV(11)	Condenser Cooling Water Flow
                XMV(12)	Agitator Speed


            Sampled Process Measurements¶
                Reactor Feed Analysis (Stream 6)
                    Sampling Frequency = 0.1 hr
                    Dead Time = 0.1 hr
                    Mole %
                    
                    Variable	Description
                    XMEAS(23)	Component A
                    XMEAS(24)	Component B
                    XMEAS(25)	Component C
                    XMEAS(26)	Component D
                    XMEAS(27)	Component E
                    XMEAS(28)	Component F

                Purge Gas Analysis (Stream 9)
                    Sampling Frequency = 0.1 hr
                    Dead Time = 0.1 hr
                    Mole %

                    Variable	Description
                    XMEAS(29)	Component A
                    XMEAS(30)	Component B
                    XMEAS(31)	Component C
                    XMEAS(32)	Component D
                    XMEAS(33)	Component E
                    XMEAS(34)	Component F
                    XMEAS(35)	Component G
                    XMEAS(36)	Component H


            Process Disturbances
                Variable	Description
                IDV(1)	A/C Feed Ratio, B Composition Constant (Stream 4) Step
                IDV(2)	B Composition, A/C Ratio Constant (Stream 4) Step
                IDV(3)	D Feed Temperature (Stream 2) Step
                IDV(4)	Reactor Cooling Water Inlet Temperature Step
                IDV(5)	Condenser Cooling Water Inlet Temperature Step
                IDV(6)	A Feed Loss (Stream 1) Step
                IDV(7)	C Header Pressure Loss - Reduced Availability (Stream 4) Step
                IDV(8)	A, B, C Feed Composition (Stream 4) Random Variation
                IDV(9)	D Feed Temperature (Stream 2) Random Variation
                IDV(10)	C Feed Temperature (Stream 4) Random Variation
                IDV(11)	Reactor Cooling Water Inlet Temperature Random Variation
                IDV(12)	Condenser Cooling Water Inlet Temperature Random Variation
                IDV(13)	Reaction Kinetics Slow Drift
                IDV(14)	Reactor Cooling Water Valve Sticking
                IDV(15)	Condenser Cooling Water Valve Sticking
                IDV(16)	Unknown
                IDV(17)	Unknown
                IDV(18)	Unknown
                IDV(19)	Unknown
                IDV(20)	Unknown


    ## pump_sensor_data
        https://www.kaggle.com/datasets/nphantawee/pump-sensor-data/data

        Columns:
            # 
            timestamp - datetime 
            sensor_00 - float - 0-2.55 - Motor Casing Vibration - mm/s, g, mils (IPS)
            sensor_01 - float - 0-56.7 - Motor Frenquency A - Hz
            sensor_02 - float - 33.2-56 - Motor Frequency B - Hz
            sensor_03 - float - 31.6-48.2 - Motor Frequency C - Hz
            sensor_04 - float - 2.8-800 - Motor Speed - RPM
            sensor_05 - float - 0-100 - Motor Current - Amps
            sensor_06 - float - 0.01-22.3 - Motor Active Power - kW
            sensor_07 - float - 0-23.6 - Motor Apparent Power - kVA
            sensor_08 - float - 0-24.3 - Motor Reactive Power  - kVAR
            sensor_09 - float - 0-25 - Motor Shaft Power - kW or HP
            sensor_10 - float - 0-76.1 - Motor Phase Current A - Amps
            sensor_11 - float - 0-60 - Motor Phase Current B - Amps
            sensor_12 - float - 0-45 - Motor Phrase Current C - Amps
            sensor_13 - float - 0-31.2 - Motor Coupling Vibration  - mm/s, g, mils (IPS) 
            sensor_14 - float - 32.4-500 - Motor Phase Voltate AB - Volts
            sensor_15 - float - null -  - 
            sensor_16 - float - 0-740 - Motor Phase Voltate BC - Volts
            sensor_17 - float - 0-600 - Motor Phase Voltate CA - Volts
            sensor_18 - float - 0-4.87 - Pump Casing Vibration - mm/s, g, mils (IPS)
            sensor_19 - float - 0-879 - Pump Stage 1 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_20 - float - 0-449 - Pump Stage 1 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_21 - float - 95.5-1110 - Pump Stage 1 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_22 - float - 0-594 - Pump Stage 1 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_23 - float - 0-1230 - Pump Stage 1 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_24 - float - 0-1000 - Pump Stage 1 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_25 - float - 0-840 - Pump Stage 2 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_26 - float - 43.2-1210 - Pump Stage 2 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_27 - float - 0-2000 - Pump Stage 2 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_28 - float - 4.32-1840 - Pump Stage 2 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_29 - float - 0.64-1470 - Pump Stage 2 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_30 - float - 0-1600 - Pump Stage 2 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_31 - float - 24-1800 - Pump Stage 2 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_32 - float - 0.24-1840 - Pump Stage 2 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_33 - float - 6.46-1580 - Pump Stage 2 Impeller Speed - RPM or Hz Multiples (Harmonic 1X, 2X, 3X)
            sensor_34 - float - 54.9-426 - Pump Inlet Flow - Gpm
            sensor_35 - float - 0-694 - Pump Discharge Flow - Gpm
            sensor_36 - float - 2.26-984 - Pump Unknown (Possibly a Pump Internal Derived Value (Likely “Internal Flow Index” or “Torque Load Estimate”)) (LOAD_ESTIMATION / INTERNAL_MODEL_VARIABLE) (Torque / Load / Derived Load Index)
            sensor_37 - float - 0-175 - Pump Lube Oil Overhead Reservoir Level - %
            sensor_38 - float - 24.5-418 - Pump Lube Oil Return Temp - Celcius
            sensor_39 - float - 29.3-548 - Pump Lube Oil Supply Temp - Celcius
            sensor_40 - float - 23.4-513 - Pump Thrust Bearing Active Temp - Celcius
            sensor_41 - float - 20.8-420 - Motor Non-Drive End Radial Bearing Temp 1 - Celcius
            sensor_42 - float - 22.1-374 - Motor Non-Drive End Radial Bearing Temp 2 - Celcius
            sensor_43 - float - 24.5-409 - Pump Thrust Bearing Inactive Temp  - Celcius
            sensor_44 - float - 25.8-1000 - Pump Drive End Radial Bearing Temp 1 - Celcius
            sensor_45 - float - 26.3-320 - Pump Non-Drive End Radial Bearing Temp 1 - Celcius
            sensor_46 - float - 26.3-370 - Pump Non-Drive End Radial Bearing Temp 2 - Celcius
            sensor_47 - float - 27.2-304 - Pump Drive End Radial Bearing Temp 2 - Celcius
            sensor_48 - float - 26.3-562 - Pump Inlet Pressure - PSI
            sensor_49 - float - 26.6-464 - Pump Temp / Pump Housing / Volute Temp - Celcius
            sensor_50 - float - 27.5-1000 - Pump Discharge Pressure 1 - PSI
            sensor_51 - float - 27.8-1000 - Pump Discharge Pressure 2 - PSI
            machine_status - int/binary/category - 0/1








    ## Microsoft Azure Predictive Maintenance
        https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance?select=PdM_telemetry.csv
        https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_telemetry.csv

        Columns:
            timestamp - Datetime
            machineID - Int - 1-100
            machineAge - Int - 0-20
            machinemodel - category - model1, model2, model3, model4
            volt - Float - 97.3 - 255
            rotate - Float - 138-695
            pressure - Float - 51.2-186
            vibration - Float - 14.9-76.8
            errorID - Category - error1, error2, error3, error4, error5
            failure - Category - comp1, comp2, comp3, comp4
            maintenance - Category - comp1, comp2, comp3, comp4



    ## NASA Bearing Dataset
        https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset/data

        Columns:
            4 to 8 columns with only vibration readings
        Special Note:
            Tere 9463 files spread over three test runs 
                - each file logs 10 minutes worth of sensor test data (except for test 1 for the first 43 files, these are saved every 5 minutes)
                - each row of sensor readings is a 1 second interval capture

    ## Ball Bearing Run-to-Failure Dataset
        https://www.kaggle.com/datasets/sujaykapadnis/ball-bearing-run-to-failure-dataset

        Columns:
            X-Axis vibration
            Y-Axis vibration
            Bearing Temperature (C)
            Atmospheric Temperature (C)

        Special Note:
            - Vibrations were sampled at 25.6 kHz for 78.125 seconds
            - Temperature data was sampled as 25.6 kHz for 78.125
            - There are a 131 files
            - Each File is an hour interval of the testing


    ## NASA Turbofan Jet Engine Data Set
        https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

        Columns: 
            1 - unit number 
            2 - time cycle 
            3-5 - operational settings 
            6-26 - sensors Measurements
                - Fan inlet temperature (◦R)
                - LPC outlet temperature (◦R)
                - HPC outlet temperature (◦R)
                - LPT outlet temperature (◦R)
                - Fan inlet Pressure (psia)
                - bypass-duct pressure (psia)
                - HPC outlet pressure (psia)
                - Physical fan speed (rpm)
                - Physical core speed  (rpm)
                - Engine pressure ratio (P50/P2)
                - HPC outlet Static pressure (psia)
                - Ratio of fuel flow to Ps30 (pps/psia)
                - Corrected fan speed (rpm)
                - Corrected core speed (rpm)
                - Bypass Ratio ()
                - Burner fuel-air ratio ()
                - Bleed Enthalpy ()
                - Required fan speed 
                - Required fan conversion speed
                - High-pressure turbines Cool air flow
                - Low-pressure turbines Cool air flow

        
1 
1 
20.0072 0.7000 100.0 

491.19 606.67 1481.04 1227.81 9.35 13.60 332.52 2323.67 8704.98 1.07 43.83 313.03 2387.78 8048.98 9.2229 0.02 362 2324 100.00 24.31 14.7007


    ## AI41 2020

        https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

        Columns:
            - UID: unique identifier ranging from 1 to 10000
            - productID: consisting of a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number
            - air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
            - process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
            - rotational speed [rpm]: calculated from powepower of 2860 W, overlaid with a normally distributed noise
            - torque [Nm]: torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.
            - tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. and a
            - machine failure: label that indicates, whether the machine has failed in this particular data point for any of the following failure modes are true.

        Columns:
            - UDI - Int - 1-1000
            - productID - Str - (M14860, L47181, L47182, L47183, L47184)
            - Type - Str/Category - H/M/L
            - Air Temperature - Float - 2 K to 300 K 
            - Process Temperature - Float - Std of 1 K, added to Air Tempature plus 10 K
            - Rotational Speed - Int - 0-2860
            - Torque - Float - Nm 0 - 80.0 
            - Tool Wear - Int - 0 - 300 (Minutes)
            - Target - Binary - 0/1 
            - Failure Type - Category - (No Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Tool Wear Failure)

