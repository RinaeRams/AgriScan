# AgriScan System - Domain Class Diagram

## Overview
The domain class diagram represents the key entities (things) in the AgriScan crop disease detection system. These classes represent the core business concepts and their relationships.

```mermaid
classDiagram
    class Farmer {
        +farmerId: String
        +name: String
        +phoneNumber: String
        +location: Location
        +cropTypes: List<Crop>
        +registrationDate: Date
        +scanHistory: List<Scan>
        +uploadPlantImage()
        +viewDiagnosis()
    }

    class Location {
        +locationId: String
        +city: String
        +province: String
        +country: String
        +latitude: Float
        +longitude: Float
        +climateZone: String
    }

    class Scan {
        +scanId: String
        +farmerId: String
        +imagePath: String
        +timestamp: DateTime
        +disease: Disease
        +confidence: Float
        +analyzeImage()
        +getDiagnosis()
    }

    class PlantImage {
        +imageId: String
        +farmerId: String
        +filePath: String
        +fileSize: Integer
        +format: String
        +uploadDate: DateTime
        +analyzeImage()
        +getDiagnosis()
    }

    class Disease {
        +diseaseId: String
        +name: String
        +scientificName: String
        +affectedCrops: List<Crop>
        +symptoms: List<String>
        +causes: String
        +severity: SeverityLevel
        +treatment: List<Treatment>
        +prevention: List<String>
        +getTreatmentPlan()
        +getPreventionTips()
    }

    class Treatment {
        +treatmentId: String
        +diseaseId: String
        +description: String
        +method: TreatmentMethod
        +effectiveness: Float
        +duration: Integer
        +cost: Float
        +materials: List<String>
        +applyTreatment()
    }

    class Crop {
        +cropId: String
        +name: String
        +scientificName: String
        +season: Season
        +waterRequirement: WaterLevel
        +soilType: SoilType
        +commonDiseases: List<Disease>
        +marketPrice: Float
        +growthPeriod: Integer
        +getOptimalConditions()
        +getDiseaseRisk()
    }

    class SystemLog {
        +logId: String
        +timestamp: DateTime
        +userId: String
        +action: String
        +result: String
        +errorMessage: String
        +responseTime: Float
        +logActivity()
        +generateReports()
    }

    %% Relationships
    Farmer "1" --> "*" Scan : makes
    Farmer "1" --> "1" Location : located in
    Farmer "1" --> "*" PlantImage : uploads

    Scan "1" --> "1" PlantImage : contains
    Scan "1" --> "1" Disease : diagnosed as

    PlantImage "1" --> "1" Disease : diagnosed as

    Disease "1" --> "*" Treatment : has
    Disease "*" --> "*" Crop : affects

    Crop "1" --> "*" Disease : has

    SystemLog "*" --> "1" Farmer : logs activity for
    SystemLog "*" --> "1" Scan : logs

    %% Enumerations
    class SeverityLevel {
        <<enumeration>>
        LOW
        MEDIUM
        HIGH
        CRITICAL
    }

    class TreatmentMethod {
        <<enumeration>>
        CHEMICAL
        ORGANIC
        CULTURAL
        BIOLOGICAL
        INTEGRATED
    }

    class Season {
        <<enumeration>>
        SPRING
        SUMMER
        AUTUMN
        WINTER
        YEAR_ROUND
    }

    class WaterLevel {
        <<enumeration>>
        LOW
        MEDIUM
        HIGH
        VERY_HIGH
    }

    class SoilType {
        <<enumeration>>
        CLAY
        SANDY
        LOAM
        SILT
        CHALK
    }
```

## Class Descriptions

### Core Entities

1. **Farmer**: Represents the primary user of the system
   - Contains personal and farm information
   - Uploads plant images for diagnosis
   - Views scan history and results

2. **Location**: Geographic information for regional context
   - Supports location-based services

3. **Scan**: Represents a plant disease diagnosis request
   - Links image input to diagnosis results
   - Contains timestamp and confidence data

### Agricultural Domain

4. **PlantImage**: Images for disease diagnosis
   - Image processing and analysis
   - Diagnosis results

5. **Disease**: Plant diseases and their management
   - Comprehensive disease information
   - Treatment and prevention strategies

6. **Treatment**: Disease treatment methods
   - Multiple treatment approaches
   - Effectiveness and cost information

7. **Crop**: Agricultural products and their characteristics
   - Links diseases and treatments
   - Contains growing requirements

### System Support

8. **SystemLog**: System activity tracking
   - Usage analytics and reporting
   - Performance monitoring

## Key Relationships

- **Farmer** uploads **PlantImage** which becomes a **Scan**
- **Scan** is diagnosed as a **Disease**
- **Disease** and **Treatment** form the knowledge base for plant health
- **Crop** connects agricultural products to diseases

## Design Principles

1. **Single Responsibility**: Each class has a clear, focused purpose
2. **High Cohesion**: Related attributes and methods are grouped together
3. **Low Coupling**: Classes are loosely connected through well-defined relationships
4. **Domain-Driven Design**: Classes reflect real-world agricultural concepts
5. **Extensibility**: Easy to add new crops, diseases, and treatments
