# Implementation Plan: Remove Dataset Limits

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dataset limits (500 feature cap, high cardinality drops, EDA column limits) from the main branch to revert to the initial "no-limits" implementation, verify with local tests and linting.

**Architecture:** We will modify `preprocessor.py` to remove the logic that drops high-cardinality columns and truncates features to 500. We will also modify `eda.py` to remove the max column limits (8 for histograms/boxplots, 15 for heatmaps) and `dataset.py` if there are any artificial chunk limits (though chunked upload is generally good, the user implies removing "limits we introduced", which likely refers to the feature caps, but we will leave the 1MB chunked upload intact as it's a structural safety feature, unless specifically targeting the data bounds). We will then run backend tests and linting (ruff/black) to ensure correctness. Finally, we'll update the documentation (`ARCHITECTURE.md`, `PLAN.md`) to reflect the removal of these limits.

**Tech Stack:** Python, Pandas, Scikit-Learn, Pytest, Ruff, Black

---

### Task 1: Remove Feature Limits in Preprocessor

**Files:**
- Modify: `backend/app/ml/preprocessor.py`

- [ ] **Step 1: Remove High-Cardinality Drop Logic**
Remove the block that drops columns with > 20 unique values and ID columns.
```python
    # ── 1. Separate features and target ──
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    steps_log.append(f"Separated target column '{target_column}' from {X.shape[1]} features.")

    # ── 2. Identify column types ──
```

- [ ] **Step 2: Remove Feature Explosion Cap**
Remove the block that truncates the dataset to 500 features.
```python
    # ── 4. Encode categorical features ──
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
        steps_log.append(
            f"One-Hot Encoded {len(categorical_cols)} categorical columns → "
            f"{X.shape[1]} total features."
        )

    # ── 5. Encode target (for classification) ──
```

### Task 2: Remove EDA Column Limits

**Files:**
- Modify: `backend/app/ml/eda.py`

- [ ] **Step 1: Remove max_cols limit in generate_histograms**
```python
def generate_histograms(df, target_column):
    """Generate histograms for numeric columns."""
    _setup_style()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if not numeric_cols:
        return None
```

- [ ] **Step 2: Remove max_cols limit in generate_box_plots**
```python
def generate_box_plots(df, target_column):
    """Generate box plots for numeric columns."""
    _setup_style()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if not numeric_cols:
        return None
```

- [ ] **Step 3: Remove 15-column limit in generate_correlation_heatmap**
```python
def generate_correlation_heatmap(df, target_column):
    """Generate a correlation heatmap for numeric columns."""
    _setup_style()
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
```

- [ ] **Step 4: Update generate_all_eda signature calls (if any exist in `eda.py`)**
Ensure `generate_all_eda` calls the updated functions without `max_cols` if it was passing them (it doesn't currently, it just uses defaults, so changing the signatures above is sufficient).

### Task 3: Update Documentation to Remove Limit References

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `PLAN.md`

- [ ] **Step 1: Remove limits from ARCHITECTURE.md**
Find and remove references to:
- "Step 2: Drop high-cardinality columns (>20 unique categorical, ID columns)" -> Remove Step 2 from 5.2 and renumber.
- "Step 6: Guard against feature explosion (cap at 500 columns)" -> Remove Step 6 from 5.2 and renumber.
- Table entries in "9. Memory Safety Design":
  - High-cardinality drop
  - ID column detection
  - Feature explosion cap
  - EDA column limit
  - Correlation heatmap limit

- [ ] **Step 2: Remove limits from PLAN.md**
- Remove "Cardinality limits, feature cap at 500" from the Risk Analysis table (keep chunked upload).
- Remove "memory guards" from Phase 7.
- Remove "14 | Memory-safe preprocessing with cardinality guards | ✅ |" from Section 3. Objectives and renumber 15.

### Task 4: Verify and Lint

- [ ] **Step 1: Run Pytest locally**
```bash
pytest backend/tests/
```
Expected: PASS

- [ ] **Step 2: Run Ruff (Linting)**
```bash
ruff check backend/
```
Expected: PASS (or auto-fix with `ruff check --fix backend/`)

- [ ] **Step 3: Run Black (Formatting)**
```bash
black backend/
```
Expected: Reformatted files if necessary.

- [ ] **Step 4: Commit Changes (Do not push)**
```bash
git add backend/app/ml/preprocessor.py backend/app/ml/eda.py ARCHITECTURE.md PLAN.md
git commit -m "feat: remove dataset limits and EDA constraints"
```