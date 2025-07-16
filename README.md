# TechNova Partners HR Analytics Project

**Status: COMPLETED** - Employee Turnover Prediction & Retention Strategy

## Project Overview

This repository contains a comprehensive HR analytics solution for predicting employee turnover and developing retention strategies. The project achieved a **269% improvement** in turnover prediction accuracy with significant business impact.

### Key Achievements
- **64% of departures** now predictable (vs. 8% baseline)
- **$500K+ annual savings** potential through early intervention
- **269% improvement** in F1-score (0.517 vs. 0.192 baseline)
- **Production-ready model** with comprehensive monitoring
- **600% ROI** with 2.4-month payback period

## Quick Start

### Prerequisites
- Python 3.8+
- Poetry (for dependency management)

### Installation
```bash
# Clone the repository
git clone https://github.com/Septimus4/Automated-classification.git
cd Automated-classification

# Install dependencies
poetry install
```

### Running the Project
```bash
# Verify project completion and dependencies
poetry run python main.py verify

# Run all 6 phases of analysis
poetry run python main.py run-all

# Run a specific phase (1-6)
poetry run python main.py run-single 1

# Get help
poetry run python main.py help
```

## 📚 Project Structure

### 📓 Notebooks (6 Phases)
1. **Data Wrangling & EDA** - `1_data_wrangling_eda.ipynb`
2. **Feature Engineering** - `2_feature_engineering.ipynb`
3. **Baseline Modeling** - `3_baseline_modeling.ipynb`
4. **Class Imbalance Handling** - `4_class_imbalance_handling.ipynb`
5. **Hyperparameter Tuning & Interpretability** - `5_hyperparameter_tuning.ipynb`
6. **Executive Presentation** - `6_executive_presentation.ipynb`

### Key Files
- `hr_analytics_utils.py` - Custom utility functions and classes
- `main.py` - Project launcher and main entry point
- `run_all_notebooks.py` - Automated notebook runner
- `verify_project_completion.py` - Project verification script
- `PROJECT_SUMMARY.md` - Comprehensive project summary
- `FINAL_REPORT.md` - Executive final report

###  Data Files
- `extrait_sirh.csv` - HRIS employee data (1,470 employees)
- `extrait_eval.csv` - Performance evaluation data
- `extrait_sondage.csv` - Employee survey data with turnover flags

### Results
- `results/` - Model artifacts, predictions, and analysis outputs
- `results/technova_hr.db` - SQLite database with all results

## Technical Details

### Model Performance
- **Best Model**: Threshold Optimized Random Forest
- **F1-Score**: 0.517 (269% improvement over baseline)
- **Recall**: 64% (ability to identify departures) 
- **Precision**: 43% (accuracy of predictions)
- **Overall Accuracy**: 87%

### Feature Engineering
- **1,551 features** engineered from 57 base variables
- **Satisfaction aggregations** - strongest predictors
- **Interaction terms** - department-specific patterns
- **Domain transformations** - HR-specific insights

### Business Impact
- **At-Risk Employees Identified**: 131 additional employees annually
- **Intervention Success Rate**: 60% expected
- **Prevented Departures**: 79 employees per year
- **Cost Savings**: $500K+ annually ($25K per prevented departure)

##  Key Insights

### Top 5 Turnover Predictors
1. **Employee Satisfaction** (CRITICAL) - Strongest predictor
2. **Overtime Patterns** (HIGH) - Excessive overtime = risk
3. **Compensation Gaps** (HIGH) - Salary disparities matter
4. **Department Variations** (MEDIUM) - Commercial/IT highest risk
5. **New Hire Integration** (MEDIUM) - First-year employee patterns

### Actionable Recommendations
1. **Quarterly satisfaction surveys** with action plans
2. **Overtime monitoring** with workload balancing
3. **Compensation benchmarking** and gap analysis
4. **Department-specific strategies** for high-risk areas
5. **Enhanced onboarding** for new employees

## 🛠️ Technology Stack

- **Python 3.12** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning models and evaluation
- **SHAP** - Model interpretability and feature importance
- **Matplotlib & Seaborn** - Data visualization
- **Plotly** - Interactive visualizations
- **SQLite** - Database for results storage
- **Jupyter** - Interactive notebook environment
- **Poetry** - Dependency management

## Usage Examples

### Basic Verification
```bash
# Check if project is ready to run
poetry run python main.py verify
```

### Running Analysis
```bash
# Run complete 6-phase analysis
poetry run python main.py run-all

# Run with error tolerance
poetry run python main.py run-all --skip-errors

# Run specific phase
poetry run python main.py run-single 4  # Class imbalance handling
```

### Advanced Usage
```python
# Using the HR Analytics utilities
from hr_analytics_utils import HRDataProcessor, DatabaseManager

# Load and process data
processor = HRDataProcessor(X, y)
imbalance_stats = processor.get_class_imbalance_stats()

# Database operations
db = DatabaseManager()
results = db.load_model_results('baseline_model_results')
```

##  Results Summary

The project successfully delivered:

### Technical Deliverables
- Production-ready ML model with 269% performance improvement
- Comprehensive feature engineering pipeline (1,551 features)
- SHAP-based model interpretability analysis
- SQLite database with all results and predictions
- 6 complete Jupyter notebooks with full analysis

### Business Deliverables  
- Executive presentation with ROI analysis
- Implementation roadmap with timeline
- Cost-benefit analysis ($500K+ savings potential)
- Actionable recommendations for HR team
- Success metrics and monitoring framework

## Implementation Roadmap

### Month 1: Foundation & Quick Wins
- Deploy predictive model in production
- Set up monthly risk scoring process
- Train HR team on model interpretation
- Implement satisfaction survey system

### Month 2: Process Integration
- Launch proactive intervention program
- Implement overtime monitoring alerts
- Begin compensation benchmarking
- Start manager training program

### Month 3: Optimization & Measurement
- Complete manager training rollout
- Implement compensation adjustments
- Launch department-specific strategies
- Establish success measurement framework

## Support & Contact

For questions about this project or implementation support:

- **Technical Issues**: Check the verification script output
- **Business Questions**: Review the executive presentation (Phase 6)
- **Implementation**: Follow the roadmap in `PROJECT_SUMMARY.md`

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

---

**Project Status**: COMPLETED  
**Last Updated**: July 2025  
**Ready for**: Executive presentation and production deployment

**TechNova Partners HR Analytics Project - Successfully Completed!**
