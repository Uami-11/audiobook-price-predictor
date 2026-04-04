# Project Title

## Team Information

### Team Mars
### Team Members
1. *Nirwan Maharjan ([Uami-11](https://github.com/Uami-11))*
2. *Ugen Basnet ([UgenBasnet19](https://github.com/UgenBasnet19))*
3. *Bishesh Chapagain ([Bishesh001](https://github.com/Bishesh001))*

### Task Division
#### Nirwan Maharjan
 
**Primary Responsibilities:**
- Overall project coordination and deadline tracking
- Data loading, cleaning, and preprocessing
  - Stripping `Writtenby:` / `Narratedby:` prefixes
  - Parsing `stars` column into `star_score` and `num_ratings`
  - Converting `time` to `duration_minutes`
  - Handling `price` column (Free → 0.0, type casting)
  - Deduplication logic
- Exploratory Data Analysis (EDA)
  - All descriptive statistics
  - Visualizations: histograms, scatter plots, box plots, bar charts, correlation heatmap
  - Written insights for each visualization
- Feature engineering
  - Creating `is_top_narrator`, `is_top_author` flags
  - One-hot encoding for `language`
  - Creating binary `high_rating` target column
- Final report compilation and editing
- Coordination of Week 8 presentation
 
**Deliverables Owned:**
- `data_cleaning.ipynb`
- `eda.ipynb`
- Final report (compiled)
- Presentation slides (final version)


#### Ugen Basnet
 
**Primary Responsibilities:**
- Building and training both statistical models
  - Linear Regression: predict `price`
  - Logistic Regression: classify `high_rating`
- Train/test split (80/20, random_state=42)
- Model evaluation and diagnostics
  - Linear: R², Adjusted R², RMSE, residual plots, Q-Q plot, VIF
  - Logistic: confusion matrix, accuracy, precision, recall, F1, ROC-AUC curve
- Hypothesis testing and interpretation of results
  - Checking p-values of coefficients
  - Stating and evaluating H₀ / H₁ for each feature
- Week 4 slide deck: model justification and feature selection
- Saving trained models using `joblib`
 
**Deliverables Owned:**
- `linear_regression_model.ipynb`
- `logistic_regression_model.ipynb`
- `price_model.pkl` and `rating_model.pkl` (saved models)
- Week 4 justification slide deck


#### Bishesh Chapagain
 
**Primary Responsibilities:**
- Conducting the literature review (Week 2)
  - Finding and summarizing 3 relevant academic papers
  - Themes: pricing regression in digital goods, rating prediction classification, narrator/author influence on media performance
  - Writing structured reviews: Citation → Problem → Method → Dataset → Finding → Relevance
- Building the Python application (Week 7)
  - Streamlit or CLI app with:
    - **Price Predictor** (uses saved linear regression model)
    - **Rating Classifier** (uses saved logistic regression model)
  - Loading saved models with `joblib`
  - User input handling and output formatting
- Supporting the final presentation (content slides 1–2)
- Peer evaluations (Week 8)
 
**Deliverables Owned:**
- `literature_review.md` / `literature_review.pdf`
- `app.py` (runnable Python application)
- `requirements.txt` (for app dependencies)
