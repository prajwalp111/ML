# Python Data Science & Machine Learning Toolkit

## 1. Data Handling & Manipulation

**pandas** (`pd`)
- Powerful library for data handling and analysis.
- Reading and writing data: `read_csv`, `to_csv`, `read_excel`
- Data cleaning: `dropna`, `fillna`, `replace`
- Data manipulation: `groupby`, `merge`, `pivot_table`

**numpy** (`np`)
- Core library for numerical operations and matrix computations.
- Numerical operations: `np.sqrt`, `np.exp`, `np.log`
- Array manipulations: `np.reshape`, `np.concatenate`
- Random number generation: `np.random.rand`, `np.random.normal`

**scipy.stats** (`stats`)
- Statistical functions and distributions.
- Useful for tests and probabilities: `norm.pdf`, `ttest_ind`, `shapiro`

**math**
- Basic mathematical functions in Python.
- Includes: `sqrt`, `log`, `ceil`, `floor`

---

## 2. Data Preprocessing

**sklearn.preprocessing**
- Contains functions for preparing your data before modeling.
- `LabelEncoder`: Converts text labels into numbers.
- `StandardScaler`: Scales features so they have zero mean and unit variance.
- `MinMaxScaler`: Normalizes data to a given range (0 to 1 by default).
- `LabelBinarizer`, `scale`: Help in encoding categorical variables.

**sklearn.metrics.pairwise**
- Used to calculate distance/similarity between samples.
- `cosine_similarity`: Measures angle-based similarity between vectors.

---

## 3. Data Visualization

**matplotlib.pyplot** (`plt`)
- Basic plotting library for 2D graphs.
- Common functions: `plot`, `scatter`, `hist`, `bar`
- Customize with: `xlabel`, `ylabel`, `title`, `legend`

**seaborn** (`sns`)
- Built on matplotlib with more attractive and informative plots.
- Examples: `heatmap`, `pairplot`, `boxplot`, `barplot`, `violinplot`

**plotly** (`plotly.express`, `plotly.graph_objects`)
- Used for interactive plots.
- Simple interface: `px.scatter`, `px.bar`, `px.line`
- Custom dashboards: `go.Figure`, `update_layout`

**Basemap**
- Extension for matplotlib to plot geographic data on 2D maps.

---

## 4. Statistical Modeling

**statsmodels.api** (`sm`)
- Library for statistical analysis and modeling.
- Regression: `OLS`, `GLM`
- Time series tools: `tsa`

---

## 5. Machine Learning

**scikit-learn (`sklearn`)**

*sklearn contains many tools used to build machine learning models easily. It supports classification, regression, clustering, dimensionality reduction, and model evaluation.*

- **Regression Models:**
  - `LinearRegression`: Fits a line to predict continuous values.
  - `DecisionTreeRegressor`, `RandomForestRegressor`: Tree-based regressors.
  - `GradientBoostingRegressor`, `BaggingRegressor`: Combine multiple weak learners.
  - `KNeighborsRegressor`: Predicts using closest data points.
  - `XGBRegressor`: Gradient boosting from XGBoost, fast and powerful.

- **Classification Models:**
  - `LogisticRegression`: Good for binary/multiclass classification.
  - `DecisionTreeClassifier`, `RandomForestClassifier`: Tree-based classifiers.
  - `GradientBoostingClassifier`, `BaggingClassifier`: Boosting/Bagging ensembles.
  - `KNeighborsClassifier`: Classifies based on neighbors.
  - `XGBClassifier`: High-performance classifier from XGBoost.

- **Clustering:**
  - `KMeans`: Groups similar data points.
  - `DBSCAN`: Detects clusters of varying density.

- **Dimensionality Reduction:**
  - `PCA`: Reduces features while keeping important ones.
  - `TSNE`: Helps visualize high-dimensional data.

- **Model Tuning:**
  - `RandomizedSearchCV`: Searches best hyperparameters randomly.
  - `cross_val_score`: Validates model using cross-validation.

- **Performance Metrics:**
  - `roc_auc_score`, `accuracy_score`, `confusion_matrix`: Evaluate model quality.

- **Utilities:**
  - `train_test_split`: Splits data into training/testing sets.

---

## 6. Deep Learning

**tensorflow** (`tf`)
- Framework for building and training deep learning models.
- Key tools: `tf.keras.models`, `model.fit`, `model.evaluate`
- Layers like: `Dense`, `Dropout`, `LSTM`

**pytorch** (`torch`)
- Flexible deep learning framework.
- Core modules: `torch.nn`, `torch.optim`, `torch.cuda`

**keras**
- High-level API for TensorFlow.
- Easy model building: `Sequential`, `Dense`, `Dropout`, `LSTM`

**sklearn.neural_network**
- Traditional neural networks for small problems.
- Models: `MLPClassifier`, `MLPRegressor`

---

## 7. Natural Language Processing (NLP)

**nltk**
- Toolkit for text processing and analysis.
- Basic NLP: `word_tokenize`, `stopwords`, `WordNetLemmatizer`, `pos_tag`, `ne_chunk`

**spacy**
- Fast and efficient NLP pipeline.
- Named entities and noun phrases: `doc.ents`, `doc.noun_chunks`

**re, string**
- Built-in modules for regular expressions and text manipulation.

**sklearn.feature_extraction.text**
- Text to numeric features: `CountVectorizer`, `TfidfVectorizer`

---

## 8. Web Scraping

**beautifulsoup4** (`bs4`)
- Parses HTML for web scraping.
- Useful methods: `BeautifulSoup`, `find`, `find_all`

**scrapy**
- Powerful web scraping framework.
- Used for crawling and automation: `Spider`, `Request`

---

## 9. APIs & HTTP

**requests**
- Makes HTTP requests easily.
- Methods: `get`, `post`
- Responses: `response.status_code`, `response.json()`

---

## 10. Databases & Storage

**sqlalchemy**
- Python SQL toolkit.
- Used for ORM and queries: `create_engine`, `sessionmaker`, `execute`

**sqlite3**
- Built-in support for SQLite databases.
- Commands: `connect`, `cursor`, `execute`

---
These libraries form a robust Python toolkit for data science, from data processing and modeling to visualization and deployment.
