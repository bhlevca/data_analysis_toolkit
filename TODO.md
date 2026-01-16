# TODO - Future Enhancements

This document tracks lower priority features and enhancements planned for future releases.

## Low Priority Features

### 1. LaTeX Table Export
**Priority:** Low  
**Estimated Effort:** Medium

Export analysis results directly to LaTeX table format for academic papers.

- [ ] Add `export_latex()` method to ReportGenerator
- [ ] Support for booktabs style tables
- [ ] APA/AMA format options
- [ ] Automatic significance marking (*, **, ***)

### 2. Interactive Dashboard Builder
**Priority:** Low  
**Estimated Effort:** High

GUI tool for building custom analysis dashboards without coding.

- [ ] Drag-and-drop widget placement
- [ ] Widget types: charts, tables, metrics, filters
- [ ] Dashboard templates (statistical, ML, time series)
- [ ] Export dashboards as standalone HTML

### 3. Database Connectivity
**Priority:** Low  
**Estimated Effort:** Medium

Direct connections to SQL databases.

- [ ] Support PostgreSQL, MySQL, SQLite
- [ ] Query builder interface
- [ ] Save queries for reuse
- [ ] Incremental data loading

### 4. Cloud Integration
**Priority:** Low  
**Estimated Effort:** High

Integration with cloud storage and compute.

- [ ] AWS S3 data loading
- [ ] Google Cloud Storage support
- [ ] Azure Blob Storage
- [ ] Cloud-based model training (optional)

### 5. API Mode
**Priority:** Low  
**Estimated Effort:** Medium

RESTful API for programmatic access.

- [ ] FastAPI backend
- [ ] Authentication/authorization
- [ ] Batch processing endpoints
- [ ] Swagger documentation

### 6. Multilingual Support
**Priority:** Low  
**Estimated Effort:** Medium

Internationalization of the UI.

- [ ] Translatable strings extraction
- [ ] Spanish, French, German translations
- [ ] Right-to-left language support

---

## Medium Priority Features (Backlog)

### 7. Mixed-Effects Models
**Priority:** Medium (moved to backlog)  
**Estimated Effort:** High

Hierarchical/multilevel models for nested data.

- [ ] Linear mixed-effects models
- [ ] Generalized linear mixed models
- [ ] Random slopes and intercepts
- [ ] Model comparison (AIC/BIC)

**Dependencies:** statsmodels mixed linear models

### 8. Bayesian Model Comparison
**Priority:** Medium  
**Estimated Effort:** Medium

Enhanced Bayesian analysis tools.

- [ ] Bayes factors calculation
- [ ] ROPE analysis
- [ ] Posterior predictive checks
- [ ] Model averaging

### 9. Geographic/Spatial Analysis
**Priority:** Medium  
**Estimated Effort:** High

Full geospatial analysis capabilities.

- [ ] Shapefile/GeoJSON loading
- [ ] Choropleth maps
- [ ] Kriging interpolation
- [ ] Spatial regression

**Dependencies:** geopandas, folium

### 10. Real-Time Data Streaming
**Priority:** Medium  
**Estimated Effort:** High

Process streaming data sources.

- [ ] Kafka consumer
- [ ] WebSocket data ingestion
- [ ] Real-time plotting
- [ ] Anomaly detection alerts

---

## Completed Features (v4.0)

- ✅ Effect Sizes Module (Cohen's d, η², Cramér's V, etc.)
- ✅ Model Validation Module (cross-validation, learning curves)
- ✅ Report Generator (HTML/Markdown reports, provenance tracking)
- ✅ Model Interpretability (SHAP, LIME, permutation importance)
- ✅ Survival Analysis (Kaplan-Meier, Cox regression)
- ✅ Data Quality Module (missing data analysis, imputation)
- ✅ Feature Selection (RFE, Boruta, SHAP-based)
- ✅ Advanced Time Series (Prophet, changepoint detection, DTW)
- ✅ Domain-Specific Analysis (ecology, climate, biostatistics)
- ✅ Multiple Testing Correction (Bonferroni, FDR)
- ✅ Variance Inflation Factor (VIF)
- ✅ Robust Statistics

---

## Contributing

To contribute to any of these features:

1. Check if the feature has an open issue
2. Discuss implementation approach in the issue
3. Follow the contribution guidelines in CONTRIBUTING.md
4. Submit a pull request with tests

## Feature Requests

Have a feature idea not listed here? Open an issue with:
- Use case description
- Expected functionality
- Any relevant references or examples
