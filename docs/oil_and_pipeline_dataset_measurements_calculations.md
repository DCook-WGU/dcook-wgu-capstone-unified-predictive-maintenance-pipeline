
    ## Oil and Gas Pipeline:
        https://www.kaggle.com/datasets/muhammadwaqas023/predictive-maintenance-oil-and-gas-pipeline-data

    ## Scheme: 
        - 11 Columns: 
            - Features:
                - Pipe Size : mm - integer
                - Thickness: mm - float
                - Material: string/category - Fiberglass, Carbon Steel, Stainless Steel, PVC, HDPE
                - Grade: string/category - ASTM A333 Grade 6, ASTM A106 Grade B, API 5L X52, API 5L X42, API 5L X65
                - Maximum Pressure: psi - integer
                - Temperature: Celcius - float 
                - Corrosion Impact Percentage:  percentage - float
                - Thickness Loss:  mm - float
                - Material Loss Percentage:  percentage - float
                - Year Times:  number of years - integer
                
            - Target
                - Conditions: string/category - Critical, Moderate, Normal

            - Inputed Target:
                - Maintenance Required: Boolean / Binary label (1 = maintenance needed, 0 = no maintenance)
                    - The goal of this dataset originally was to perform binary classification to determine if the pipeline needed to be serviced or not based on the data information from the features and the level of the condition. 
                    - Easiest method to achive this would be to set the Normal/Moderate to equal 0 for maintenance required, and Critical to 1. 
                    - Better method would be to calculate it.
                        - 
                            risk_score = 0

                            if thickness_loss > 0.3: risk_score += 1
                            if material_loss > 20: risk_score += 1
                            if corrosion > 40: risk_score += 1
                            if max_pressure > (design_pressure * 0.9): risk_score += 1
                            if temperature > 80: risk_score += 1

                            Maintenance_Required = 1 if risk_score >= 2



            ASTM A333 Grade 6: 
                - Low Temp, Seamless Carbon Steel Pipe
                - Tenise strength, min is 415 MPa or 60,000 psi
                - Yield Strength, min is 240 MPa or 35,000 psi 
                - Wall thickness: 
                    - 8mm : 


            ASTM A106 Grade B
            
            
            API 5L X52
                >= 10.3 mm to < 13.7 mm :: :: >= 1.7 mm to <= 2.4 mm
                >= 13.7 mm to < 17.1 mm :: :: >= 2.2 mm to <= 3.0 mm 
                >= 17.1 mm to < 21.3 mm :: :: >= 2.3 mm to <= 3.2 mm 
                >= 21.3 mm to < 26.7 mm :: :: >= 2.1 mm to <= 7.5 mm 
                >= 26.7 mm to < 33.4 mm :: :: >= 2.1 mm to <= 7.8 mm 
                >= 33.4 mm to < 48.3 mm :: :: >= 2.1 mm to <= 10.0 mm 
                >= 48.3 mm to < 60.3 mm :: :: >= 2.1 mm to <= 12.5 mm
                >= 60.3 mm to < 73.0 mm :: :: >= 3.6 mm to 14.2 mm
                >= 73.0 mm to < 88.9 mm :: :: > 3.6 mm to <= 20.0 mm
                >= 88.9 mm to < 101.6 mm :: :: > 4.0 mm to <= 22.0 mm 
                >= 101.6 mm to < 168.3 mm :: :: > 4.0 mm to <= 25.0 mm
                >= 168.3 mm to < 219.1 mm :: :: > 4.0 mm to <= 40.0 mm
                >= 219.1 mm to < 273.1 mm :: :: > 4.0 mm to <= 40.0 mm
                >= 273.1 mm to < 323.9 mm :: :: > 5.2 mm to <= 45.0 mm
                >= 323.9 mm to < 355.6 mm :: :: > 5.6 mm to <= 45.0 mm  
                >= 355.6 mm to < 457 mm :: :: > 7.1 mm to <= 45.0 mm 
                >= 457 mm to < 559 mm :: :: > 7.1 mm to <= 45.0 mm  
                >= 559 mm to < 711 mm :: :: > 7.1 mm to <= 45.0 mm 
                >= 711 mm to < 864 mm :: :: > 7.1 mm to <= 52.0 mm 
                >= 864 mm to < 965 mm :: :: >= 5.6 mm to <= 52.0 mm
                >= 965 mm to < 422 mm :: :: >= 6.4 mm to <= 52.0 mm
                >= 1422 mm to < 1829 mm :: :: >= 9.5 mm to <= 52.0 mm
                >= 1829 mm to < 2134 mm :: :: >= 10.3 mm to <= 52.0 mm      
        
            API 5L X42
                >= 10.3 mm to < 13.7 mm :: :: >= 1.7 mm to <= 2.4 mm
                >= 13.7 mm to < 17.1 mm :: :: >= 2.2 mm to <= 3.0 mm 
                >= 17.1 mm to < 21.3 mm :: :: >= 2.3 mm to <= 3.2 mm 
                >= 21.3 mm to < 26.7 mm :: :: >= 2.1 mm to <= 7.5 mm 
                >= 26.7 mm to < 33.4 mm :: :: >= 2.1 mm to <= 7.8 mm 
                >= 33.4 mm to < 48.3 mm :: :: >= 2.1 mm to <= 10.0 mm 
                >= 48.3 mm to < 60.3 mm :: :: >= 2.1 mm to <= 12.5 mm
                >= 60.3 mm to < 73.0 mm :: :: >= 3.6 mm to 14.2 mm
                >= 73.0 mm to < 88.9 mm :: :: > 3.6 mm to <= 20.0 mm
                >= 88.9 mm to < 101.6 mm :: :: > 4.0 mm to <= 22.0 mm 
                >= 101.6 mm to < 168.3 mm :: :: > 4.0 mm to <= 25.0 mm
                >= 168.3 mm to < 219.1 mm :: :: > 4.0 mm to <= 40.0 mm
                >= 219.1 mm to < 273.1 mm :: :: > 4.0 mm to <= 40.0 mm
                >= 273.1 mm to < 323.9 mm :: :: > 5.2 mm to <= 45.0 mm
                >= 323.9 mm to < 355.6 mm :: :: > 5.6 mm to <= 45.0 mm  
                >= 355.6 mm to < 457 mm :: :: > 7.1 mm to <= 45.0 mm 
                >= 457 mm to < 559 mm :: :: > 7.1 mm to <= 45.0 mm  
                >= 559 mm to < 711 mm :: :: > 7.1 mm to <= 45.0 mm 
                >= 711 mm to < 864 mm :: :: > 7.1 mm to <= 52.0 mm 
                >= 864 mm to < 965 mm :: :: >= 5.6 mm to <= 52.0 mm
                >= 965 mm to < 422 mm :: :: >= 6.4 mm to <= 52.0 mm
                >= 1422 mm to < 1829 mm :: :: >= 9.5 mm to <= 52.0 mm
                >= 1829 mm to < 2134 mm :: :: >= 10.3 mm to <= 52.0 mm           
            
            API 5L X65
                >= 10.3 mm to < 13.7 mm :: :: >= 1.7 mm to <= 2.4 mm
                >= 13.7 mm to < 17.1 mm :: :: >= 2.2 mm to <= 3.0 mm 
                >= 17.1 mm to < 21.3 mm :: :: >= 2.3 mm to <= 3.2 mm 
                >= 21.3 mm to < 26.7 mm :: :: >= 2.1 mm to <= 7.5 mm 
                >= 26.7 mm to < 33.4 mm :: :: >= 2.1 mm to <= 7.8 mm 
                >= 33.4 mm to < 48.3 mm :: :: >= 2.1 mm to <= 10.0 mm 
                >= 48.3 mm to < 60.3 mm :: :: >= 2.1 mm to <= 12.5 mm
                >= 60.3 mm to < 73.0 mm :: :: >= 3.6 mm to 14.2 mm
                >= 73.0 mm to < 88.9 mm :: :: > 3.6 mm to <= 20.0 mm
                >= 88.9 mm to < 101.6 mm :: :: > 4.0 mm to <= 22.0 mm 
                >= 101.6 mm to < 168.3 mm : :: > 4.0 mm to <= 25.0 mm
                >= 168.3 mm to < 219.1 mm :: :: > 4.0 mm to <= 40.0 mm
                >= 219.1 mm to < 273.1 mm :: :: > 4.0 mm to <= 40.0 mm
                >= 273.1 mm to < 323.9 mm :: :: > 5.2 mm to <= 45.0 mm
                >= 323.9 mm to < 355.6 mm :: :: > 5.6 mm to <= 45.0 mm  
                >= 355.6 mm to < 457 mm :: :: > 7.1 mm to <= 45.0 mm 
                >= 457 mm to < 559 mm :: :: > 7.1 mm to <= 45.0 mm  
                >= 559 mm to < 711 mm :: :: > 7.1 mm to <= 45.0 mm 
                >= 711 mm to < 864 mm :: :: > 7.1 mm to <= 52.0 mm 
                >= 864 mm to < 965 mm :: :: >= 5.6 mm to <= 52.0 mm
                >= 965 mm to < 422 mm :: :: >= 6.4 mm to <= 52.0 mm
                >= 1422 mm to < 1829 mm :: :: >= 9.5 mm to <= 52.0 mm
                >= 1829 mm to < 2134 mm :: :: >= 10.3 mm to <= 52.0 mm                  