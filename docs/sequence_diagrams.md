# AgriScan System - Sequence Diagrams

## Overview
This document contains simplified sequence diagrams for key use cases in the AgriScan crop disease detection system. Each diagram shows the interaction flow between actors and system components.

## 1. Plant Disease Diagnosis

```mermaid
sequenceDiagram
    participant F as Farmer
    participant UI as Web Interface
    participant PS as Plant Scanner Service
    participant IP as Image Processor
    participant ML as ML Disease Model
    participant DB as Disease Database
    participant TE as Treatment Engine

    F->>UI: Upload/capture plant image
    UI->>PS: Send image file
    PS->>IP: Preprocess image
    IP-->>PS: Return processed image
    PS->>ML: Analyze for diseases
    ML->>DB: Query disease patterns
    DB-->>ML: Return disease data
    ML-->>PS: Return diagnosis results
    PS->>TE: Get treatment plan
    TE->>DB: Query treatments
    DB-->>TE: Return treatment options
    TE-->>PS: Return treatment plan
    PS-->>UI: Return diagnosis & treatment
    UI-->>F: Display results & recommendations
    PS->>DB: Log diagnosis
```

## 2. System Monitoring and Analytics

```mermaid
sequenceDiagram
    participant EO as Extension Officer
    participant UI as Admin Interface
    participant AS as Analytics Service
    participant DB as Database
    participant LOG as System Logs

    EO->>UI: Login to admin panel
    UI->>AS: Request dashboard data
    AS->>DB: Query usage statistics
    DB-->>AS: Return query counts
    AS->>LOG: Get system performance
    LOG-->>AS: Return performance metrics
    AS->>DB: Get farmer demographics
    DB-->>AS: Return user data
    AS-->>UI: Return analytics data
    UI-->>EO: Display dashboard
```

## Common Patterns

### Error Handling Pattern
```mermaid
sequenceDiagram
    participant U as User
    participant S as Service
    participant API as External API
    participant FB as Fallback Service

    U->>S: Make request
    S->>API: Call external service
    API-->>S: Error response
    S->>FB: Try fallback
    FB-->>S: Fallback response
    S-->>U: Return fallback data
    S->>LOG: Log error
```

### Authentication Pattern
```mermaid
sequenceDiagram
    participant U as User
    participant UI as Interface
    participant AUTH as Auth Service
    participant DB as User Database

    U->>UI: Login request
    UI->>AUTH: Validate credentials
    AUTH->>DB: Query user data
    DB-->>AUTH: Return user info
    AUTH->>AUTH: Verify password
    AUTH-->>UI: Return auth token
    UI-->>U: Login successful
    U->>UI: Access protected resource
    UI->>AUTH: Validate token
    AUTH-->>UI: Token valid
    UI-->>U: Grant access
```

## Performance Considerations

1. **Caching**: Analysis results cached for repeat scans
2. **Async Processing**: Image analysis runs asynchronously
3. **Database Indexing**: Query logs indexed by timestamp and user
4. **CDN**: Static assets served via CDN

## Security Measures

1. **Input Validation**: All user inputs validated
2. **Rate Limiting**: API calls limited per user
3. **Encryption**: Sensitive data encrypted in transit and at rest
4. **Audit Logging**: All system interactions logged
