# Enhanced Tutorial: Signal Analysis & Dimensionality Reduction

## 📡 Signal Analysis (New Comprehensive Guide)

### Why Use Signal Analysis?

**Signal analysis helps you understand patterns hidden in time-series or frequency data:**

- 🎵 **Audio/Music**: Identify dominant frequencies, remove noise
- 📊 **Sensor Data**: Detect oscillations, periodicities, or anomalies
- 🏥 **Biomedical**: Find heartbeat patterns, brain wave frequencies
- 💰 **Finance**: Detect cyclical market patterns
- 🌊 **Physics/Engineering**: Analyze vibrations, waves, oscillations
- 🔌 **Electronics**: Characterize signal components

**Real-world example**: You have a signal that's a combination of 3 Hz and 15 Hz oscillations mixed with noise. Signal analysis helps you:
- Identify which frequencies are present (FFT)
- See how frequencies change over time (CWT)
- Extract different frequency components (DWT)

---

## 🎯 When to Use Each Signal Analysis Method

### 1. **FFT (Fast Fourier Transform)** 
#### ❓ When & Why
- **Use when**: You want to know WHICH frequencies are present in your signal
- **Best for**: Stationary signals (frequencies don't change over time)
- **Problems it solves**:
  - "What are the main oscillation frequencies?"
  - "Is there 50/60 Hz electrical noise in my data?"
  - "What's the dominant frequency of this vibration?"

#### 📋 How to Use
1. Go to **Signal Analysis** tab
2. Load your time-series data
3. Set **Sampling Rate** (Hz) - must match your data collection rate
   - Example: If you recorded at 200 samples/second → 200 Hz
4. Click **Compute FFT**
5. View the **FFT Magnitude Spectrum** plot

#### 📊 Interpreting Results

**The FFT plot shows:**
- **X-axis**: Frequency (Hz)
- **Y-axis**: Magnitude (power/intensity)
- **Peaks**: Dominant frequencies in your signal

**Example results:**
```
✓ Dominant Frequency: 3.00 Hz
✓ Peak Power: 3.92e+04

Top 5 frequencies found:
  1. 3.00 Hz - Power: 3.92e+04  ← Main oscillation
  2. 15.00 Hz - Power: 1.46e+04 ← Secondary oscillation
  3. 60.00 Hz - Power: 2.39e+01 ← Electrical noise (50/60 Hz artifact)
```

**What it means:**
- Peaks at 3 Hz and 15 Hz → Signal contains these frequencies
- 60 Hz peak → Electrical noise interference (common in AC mains power)
- Frequency values tell you the period: Period = 1 / Frequency
  - 3 Hz → Period = 0.33 seconds (oscillates 3 times/second)
  - 15 Hz → Period = 0.067 seconds (oscillates 15 times/second)

---

### 2. **PSD (Power Spectral Density)**
#### ❓ When & Why
- **Use when**: You want smooth frequency representation for noise analysis
- **Best for**: Identifying noise floors, comparing frequency content
- **Problems it solves**:
  - "How much power is in each frequency range?"
  - "Where is the noise concentrated?"
  - "Is my signal above the noise floor?"

#### 📋 How to Use
1. Same starting point as FFT
2. Click **Compute PSD**
3. View the **Power Spectral Density** plot (Welch method)

#### 📊 Interpreting Results

**PSD advantages over FFT:**
- Smoother estimate (better for noisy data)
- Better for comparing signal power across frequencies
- Shows which frequency ranges contain energy

**Example interpretation:**
```
Dominant Frequency (PSD): 2.00 Hz
Total Power: 4.03e-01

What to look for:
- Higher values → More power at that frequency
- Flat regions → Noise floor
- Sharp peaks → Signal components
- Slope pattern → Pink noise (1/f) suggests correlation structure
```

---

### 3. **CWT (Continuous Wavelet Transform)**
#### ❓ When & Why
- **Use when**: Frequencies CHANGE over time (non-stationary signals)
- **Best for**: Time-frequency analysis, detecting transients
- **Problems it solves**:
  - "Does the frequency change during the recording?"
  - "When do frequency components appear/disappear?"
  - "Where are the high-power events?"
  - "Are there frequency sweeps or modulations?"

#### 📋 How to Use
1. Go to **Signal Analysis** tab
2. Set sampling rate
3. Click **Compute CWT**
4. View the **Time-Frequency Heatmap**:
   - **X-axis**: Time (seconds)
   - **Y-axis**: Frequency/Period (Hz)
   - **Color**: Power (red = high, blue = low)

#### 📊 Interpreting Results

**Reading the CWT heatmap:**
```
Power vs Time-Frequency:

Frequency (Hz)
     │     Bright red region
     │     (high power)
  15 │  ╔════════════════╗
     │  ║                ║
  10 │  ║    CWT Power   ║
     │  ║    Heatmap     ║
   5 │  ║╔══════════════╗║
     │  ║║ Cone of      ║║
   2 │  ║║ Influence    ║║
     │  ╚╩══════════════╩╝
   0 └──┴──┴──┴──┴──┴──┴──┴──
     0  0.5  1.0  1.5  2.0  Time(s)
```

**What the colors mean:**
- 🔴 **Red/Bright**: High power at that time-frequency
- 🟡 **Yellow**: Medium power
- 🔵 **Blue/Dark**: Low power (background/noise)

**Cone of Influence (COI)** - gray shaded region:
- Area where edge effects are significant
- Results OUTSIDE COI are more reliable
- Results INSIDE COI (especially edges) should be interpreted cautiously

**Example analysis:**
```
If CWT shows:
✓ Horizontal red stripe at 3 Hz → 3 Hz is present throughout signal
✓ Red region appearing mid-signal → Transient at that time
✓ Red line sweeping from low→high → Chirp (frequency sweep)
✗ Only blue below COI → No real signal, just noise
```

---

### 4. **DWT (Discrete Wavelet Transform)**
#### ❓ When & Why
- **Use when**: You need to decompose signal into detail/approximation components
- **Best for**: Denoising, feature extraction, compression
- **Problems it solves**:
  - "How do I remove noise while preserving signal?"
  - "What are the multi-scale features in my signal?"
  - "Can I compress this signal?"
  - "What's the high-frequency vs low-frequency content?"

#### 📋 How to Use
1. Go to **Signal Analysis** tab
2. Set sampling rate
3. Click **Compute DWT**
4. View the **Wavelet Decomposition** plot showing multiple levels

#### 📊 Interpreting Results

**DWT breakdown:**
```
Original Signal
    │
    ├─► Approximation (A1)   ← Low-frequency/smooth part
    │       │
    │       ├─► A2
    │       │    └─► A3, etc.
    │
    └─► Details (D1, D2, D3) ← High-frequency/detail parts
            │
            └─ Each level shows finer details
```

**Levels explained:**
- **Level 1**: Highest frequency resolution (finest details)
- **Level 2-N**: Progressively coarser (lower frequencies)
- **Approximation**: Overall trend/low-frequency envelope
- **Details**: Deviations from trend at each scale

**Example interpretation:**
```
DWT with 5 levels:

Level 1 Details (D1): High-freq noise, rapid oscillations
Level 2 Details (D2): Medium-freq ripples
Level 3 Details (D3): Slower modulations
Level 4 Details (D4): Even slower changes
Level 5 Details (D5): Very slow variations
Approximation (A5): Overall trend/baseline

For denoising: Zero out D1 (noise level) → Reconstruct
For compression: Keep only large-magnitude coefficients
```

---

## 📊 Sampling Rate - CRITICAL Setting

**What is sampling rate?**
- How many times per second you collected data points
- Must match your actual data collection

**Common sampling rates:**
- Audio: 44,100 Hz (CD quality), 22,050 Hz
- Biomedical: 100-1000 Hz
- Sensors: 10-100 Hz
- Financial data: 1 Hz (once per second) or less

**Nyquist Frequency** (important limit):
- Max detectable frequency = Sampling Rate / 2
- At 200 Hz sampling → Can only detect up to 100 Hz frequencies
- Missing this causes aliasing (false frequency detection)

**Example with 200 Hz sampling:**
```
✓ Can detect 3 Hz ✓ Can detect 15 Hz ✓ Can detect 90 Hz
✗ Cannot detect 150 Hz ✗ Cannot detect 200 Hz
  (will get false frequencies)
```

---

## 🔍 Complete Signal Analysis Workflow

### Workflow for Unknown Signal

```
Step 1: Load signal, set correct sampling rate
   ↓
Step 2: Plot raw signal (sanity check)
   ↓
Step 3: Compute FFT → See which frequencies exist
   ↓
Step 4: Compute CWT → See if frequencies change over time
   ↓
Step 5: Interpret:
   - Static frequencies? (FFT enough)
   - Changing frequencies? (Use CWT)
   - Need denoising? (Use DWT)
   ↓
Step 6: Apply appropriate transformation
```

### Workflow for Denoising

```
Step 1: Compute DWT (5-10 levels)
   ↓
Step 2: Identify noise level:
   - Look at highest-frequency details (D1, D2)
   - Identify noise threshold
   ↓
Step 3: Zero out small coefficients
   - Keep only |coeff| > threshold
   ↓
Step 4: Reconstruct signal
   ↓
Step 5: Compare original vs denoised
```

---

## 🎨 PCA (Principal Component Analysis) - Enhanced with Vector Visualization

### Why Use PCA?

**PCA helps you:**
- Reduce data dimensions (100 columns → 2-3)
- Visualize high-dimensional data
- Find patterns in the data
- Remove noise/less important variations
- Speed up machine learning models

**Real-world problems it solves:**
- 📸 **Images**: Compress from 1000s of pixels to 10-50 components
- 🧬 **Genomics**: Reduce 20,000 genes to key patterns
- 📊 **Finance**: Find key market drivers from 100+ stocks
- 🎵 **Music**: Identify main patterns in audio features

---

## 📍 When to Use PCA

**Use PCA when:**
- ✅ You have many columns (features)
- ✅ You want to visualize data
- ✅ You suspect correlations exist
- ✅ You want to denoise data
- ✅ You need faster model training

**Don't use PCA when:**
- ❌ You need feature interpretability (PCA makes it hard)
- ❌ Your data is low-dimensional (<10 variables)
- ❌ You care about individual feature importance

---

## 🎯 How to Use PCA with Vector Visualization

### Step 1: Prepare Your Data
1. Go to **Non-Linear Analysis** tab
2. Load your dataset
3. Select numeric features (at least 3)
4. **Auto-scale checkbox** ✓ (recommended)
   - Ensures equal contribution regardless of units

### Step 2: Choose Number of Components
1. Select **Number of Components**: Start with 2-3
   - 2 components → 2D scatter plot
   - 3 components → 3D scatter plot
2. Click **Compute PCA**

### Step 3: Interpret Results

**First, look at explained variance:**
```
Explained Variance:
  PC1: 45.2% ← First component explains 45% of variation
  PC2: 28.5% ← Second component explains 28.5%
  PC3: 15.1% ← Third component explains 15.1%
  ─────────────
  Total: 88.8% ← 3 components capture 88.8% of info

Rule of thumb:
  80-90% is usually good enough
  <70% means you lost important info, use more components
```

---

## 📈 **NEW: Cartesian Plot with Vector Indicators**

### Understanding the Vector Plot

The enhanced PCA visualization now shows:

```
         PC2 (28.5% var)
         │
         │      Var1 vector
         │        ↗ (angle shows contribution)
         │       /
         │      /  Var2 vector
         │     /      ↗
      ───┼────────────────  PC1 (45.2% var)
         │
         │
    Original variables as vectors in PC space:
    - Vector direction: How the original variable aligns with PCs
    - Vector length: How much that variable contributes
    - Vector angle: Shows variable relationships
```

### Reading the Vector Plot

**Vector meanings:**
- 🔴 **Long vector → PC1** direction: Variable strongly defines PC1
- 🔵 **Long vector → PC2** direction: Variable strongly defines PC2
- 📏 **Vector length**: Magnitude of contribution (longer = stronger)
- 🔄 **Vector angle between two variables**: Shows their correlation
  - **Parallel vectors** (small angle) → Variables correlated
  - **Perpendicular vectors** (90° angle) → Variables independent
  - **Opposite vectors** (180° angle) → Variables negatively correlated

**Example:**
```
Vectors in PCA plot:

     PC2
      │     Age
      │    ↗│← Long vector pointing at PC2
      │   / │  (Age varies mostly in PC2 direction)
   ───┼──/──────────  PC1
      │/    Salary (long → explains PC1)
      │

Interpretation:
✓ Age and Salary are somewhat perpendicular → Weakly related
✓ Salary has long projection on PC1 → PC1 is salary-driven
✓ Age has long projection on PC2 → PC2 is age-driven
```

---

## 🎨 Detailed Example: Iris Flower Dataset

### Dataset Overview
```
4 variables: Sepal Length, Sepal Width, Petal Length, Petal Width
3 classes: Setosa, Versicolor, Virginica
```

### Step 1: Run PCA with 2 components

Results:
```
PC1: 72.9% variance explained
PC2: 22.8% variance explained
Total: 95.7% (excellent!)
```

### Step 2: View Cartesian Plot with Vectors

```
             PC2 (22.8%)
                │
                │ PetalWidth↗
                │        ↗ PetalLength
                │      ↗
         ───────┼─────────────────── PC1 (72.9%)
                │    ↗ SepalLength
              ↙ │ ↗
         SepalWidth │

Data points (colored by class):
         │     ✳️ Setosa (small circles)
         │   ✳️✳️✳️
         │✳️✳️
         ├────────┼── ⭐ Virginica (large stars)
         │        ⭐⭐⭐
         │          ⭐⭐
```

### Step 3: Interpret Vectors

**What the vectors tell us:**
1. **PetalLength & PetalWidth** vectors point upper-right
   - Strong contribution to PC1
   - Highly correlated (parallel)
   - These distinguish Virginica from others

2. **SepalLength** points right
   - Strong PC1 contribution
   - Partially aligns with petal measurements

3. **SepalWidth** points slightly left-down
   - Weak PC1 contribution (almost perpendicular)
   - Independent of petal measurements

**Biological meaning:**
- PC1 captures "overall flower size" (petal & sepal length)
- PC2 captures "petal vs sepal balance"
- Virginica flowers are larger (right side)
- Setosa flowers have proportionally larger sepals (left/down)

---

## 💡 Interpretation Tips for Vectors

### Biplot Reading Rules

| Feature | Interpretation |
|---------|-----------------|
| Vector parallel to PC1 | Strongly relates to PC1 |
| Vector parallel to PC2 | Strongly relates to PC2 |
| Vectors pointing same way | Positively correlated |
| Vectors opposite | Negatively correlated |
| Vectors perpendicular | Uncorrelated/Independent |
| Long vector | Important contributor |
| Short vector | Minor contributor |

### Common Patterns

**Pattern 1: Two vectors close together**
```
Output: Performance ↗
Input:  Effort      ↗
        Quality     ↗↑ (slight angle)

Interpretation: All increase together (positive correlation)
Action: Can probably predict one from others
```

**Pattern 2: Perpendicular vectors**
```
Age ↑
    │
    │ Income →

Interpretation: Age and income independent
Action: Independently vary
```

**Pattern 3: Opposite vectors**
```
Complexity ←─── Price
            180°

Interpretation: Negative relationship
Action: Higher complexity → Lower price (or vice versa)
```

---

## 🔍 PCA + Vector Plot Workflow

### Complete Analysis Workflow

```
Step 1: Load data
   ↓
Step 2: Select features
   ↓
Step 3: Auto-scale data ✓
   ↓
Step 4: Compute PCA (2-3 components)
   ↓
Step 5: Check explained variance
   - Is it > 80%? If not, use more components
   ↓
Step 6: Examine Cartesian plot with vectors
   ├─ Where are clusters?
   ├─ Which variables drive PC1, PC2?
   └─ What are the correlations?
   ↓
Step 7: Interpret vectors
   - Parallel → Correlated
   - Perpendicular → Independent
   - Opposite → Negatively correlated
   ↓
Step 8: Draw conclusions about data structure
```

---

## 📋 PCA Result Interpretation Checklist

When looking at PCA results, answer:

- [ ] Is explained variance > 80%?
- [ ] Are clusters visible in the plot?
- [ ] Which variables have the longest vectors?
- [ ] Are any variables nearly perpendicular?
- [ ] Do the results match domain knowledge?
- [ ] Are there any outliers?
- [ ] Can you name what each PC represents?

---

## 🎓 Complete Example: Patient Health Data

**Dataset**: 10 health measurements for 200 patients
- Age, Blood Pressure, Cholesterol, BMI, Exercise, Sleep, Stress, Heart Rate, Glucose, Weight

### Analysis Results

```
PCA with 3 components:
  PC1: 52.3% "Overall Health Status"
       (strongly defined by: BP, Cholesterol, Glucose)
       
  PC2: 28.1% "Lifestyle Factors"  
       (strongly defined by: Exercise, Sleep, Stress)
       
  PC3: 12.6% "Anthropometric Features"
       (strongly defined by: Weight, BMI, Age)
```

### Vector Plot Insights

```
        PC2 (Lifestyle)
         │   Exercise
         │    ↑│
         │   / │ Sleep↖
         ├─ /──┼────────────── PC1 (Health)
         │/    │    ↗ BP
         │ Stress↗ Chol
         │         Glucose

Interpretation:
✓ Sedentary patients (low Exercise, low Sleep, high Stress) 
  → bottom-left of plot
  
✓ Active healthy patients (high Exercise, good Sleep, low Stress,
  low BP, low Cholesterol)
  → top-right of plot
  
✓ Can easily segment patients into health categories
```

---

## 🎯 Key Takeaways

### Signal Analysis
- **FFT**: Static frequency analysis
- **PSD**: Smooth frequency representation  
- **CWT**: Time-frequency changes
- **DWT**: Multi-scale decomposition

### PCA with Vectors
- **Vectors** show how original features contribute to PCs
- **Vector angles** reveal feature correlations
- **Vector lengths** show importance
- **Clusters** in plot show natural groupings

### Always Remember
✓ Set correct sampling rate for signal analysis  
✓ Explain variance > 80% is good for PCA  
✓ Use vectors to interpret feature relationships  
✓ Compare results with domain knowledge
