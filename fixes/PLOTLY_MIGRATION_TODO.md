# 🔄 Plotly Migration - Signal Analysis Tab

**Date:** December 9, 2024  
**Status:** Pending - Sleep first! 😴

---

## 🎯 QUICK FIX APPLIED TONIGHT

✅ **FFT Subplot Fixed** (Line ~1906):
- Removed `sharey=ax_cwt` (was causing scaling issues)
- Added magnitude filtering (threshold = 1% of max) to remove noise floor
- Added fill_between for better visualization
- Should now show clear 3 Hz and 15 Hz peaks

---

## 📋 TOMORROW'S TASKS: Full Matplotlib → Plotly Migration

### Current State:
- **FFT standalone**: ✅ Already using Plotly (line 1977-1986) - WORKING PERFECTLY
- **PSD standalone**: ✅ Already using Plotly (line 2012-2020) - WORKING PERFECTLY  
- **CWT subplot**: ❌ Uses Matplotlib (line ~1890-1900)
- **PSD subplot**: ❌ Uses Matplotlib (line ~1901-1905)
- **FFT subplot**: ❌ Uses Matplotlib (line ~1906-1920) - JUST FIXED
- **DWT display**: ❌ Uses Matplotlib (line ~2054)

### Migration Priority:

#### 🔴 HIGH PRIORITY (Fix display issues)

1. **CWT Display** (Line ~1890-1900)
   - Replace `ax_cwt.imshow()` with Plotly Heatmap
   - Use `go.Heatmap()` for interactive color-coded spectrogram
   - Add hover info showing time/frequency/magnitude
   - File: `src/data_toolkit/streamlit_app.py`

2. **FFT + PSD + CWT Combined View** (Line ~1885-1916)
   - Replace entire GridSpec layout with Plotly subplots
   - Use `make_subplots()` with 1 row, 2-3 columns
   - Reuse standalone Plotly FFT code (already working)
   - Add subplot for CWT heatmap

3. **DWT Display** (Line ~2049-2058)
   - Replace `plot_discrete_wavelet()` matplotlib function
   - Create Plotly version with interactive multi-level plots
   - Use `make_subplots()` for stacked coefficient plots
   - File: `src/data_toolkit/signal_analysis.py` (line 317)

#### 🟢 NICE TO HAVE

4. **Enhance Interactive Features**
   - Add frequency range selection slider
   - Add zoom/pan controls  
   - Add export to PNG/SVG buttons
   - Add annotations for dominant frequencies

---

## 🔧 IMPLEMENTATION GUIDE

### Template: CWT Heatmap Conversion

**Current Matplotlib Code** (Line ~1893):
```python
cwt_magnitude = cwt_res.get('magnitude')
scales = cwt_res.get('scales')
times = cwt_res.get('time')
ax_cwt.imshow(cwt_magnitude, extent=[times[0], times[-1], scales[0], scales[-1]], 
              cmap='viridis', aspect='auto', origin='lower')
ax_cwt.set_xlabel('Time')
ax_cwt.set_ylabel('Frequency')
ax_cwt.set_title('Continuous Wavelet Transform')
```

**Target Plotly Code**:
```python
cwt_magnitude = cwt_res.get('magnitude')
scales = cwt_res.get('scales')
times = cwt_res.get('time')

fig_cwt = go.Figure(data=go.Heatmap(
    z=cwt_magnitude,
    x=times,
    y=scales,
    colorscale='Viridis',
    hovertemplate='Time: %{x:.3f}s<br>Frequency: %{y:.2f}<br>Magnitude: %{z:.2e}<extra></extra>'
))

fig_cwt.update_layout(
    title='Continuous Wavelet Transform',
    xaxis_title='Time (s)',
    yaxis_title='Frequency Scale',
    template='plotly_white',
    height=500,
    hovermode='closest'
)

st.plotly_chart(fig_cwt, use_container_width=True)
```

### Template: Combined View with Subplots

```python
from plotly.subplots import make_subplots

# Create figure with 1 row, 3 columns (CWT, FFT, PSD)
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('CWT', 'FFT', 'PSD'),
    specs=[[{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'scatter'}]]
)

# Add CWT heatmap
fig.add_trace(go.Heatmap(
    z=cwt_magnitude,
    x=times,
    y=scales,
    colorscale='Viridis',
    name='CWT'
), row=1, col=1)

# Add FFT line
fig.add_trace(go.Scatter(
    x=fft_res['positive_frequencies'],
    y=fft_res['magnitude'],
    mode='lines',
    name='FFT',
    line=dict(color='steelblue')
), row=1, col=2)

# Add PSD line
fig.add_trace(go.Scatter(
    x=psd_res['frequencies'],
    y=psd_res['power_spectral_density'],
    mode='lines',
    name='PSD',
    line=dict(color='coral')
), row=1, col=3)

fig.update_layout(height=400, template='plotly_white')
st.plotly_chart(fig, use_container_width=True)
```

---

## 🧪 TESTING CHECKLIST

After migration, test with synthetic signal:
- [ ] Load test data: `1.0*sin(2π*3t) + 0.6*sin(2π*15t) + noise`
- [ ] FFT shows clear peaks at 3 Hz and 15 Hz
- [ ] CWT heatmap shows two horizontal bands at 3 Hz and 15 Hz
- [ ] PSD shows power concentration at 3 Hz and 15 Hz
- [ ] DWT shows decomposition across 3-5 levels
- [ ] All plots are interactive (zoom, pan, hover)
- [ ] Plots render correctly at different screen sizes

---

## 📚 KEY FILES TO MODIFY

1. **src/data_toolkit/streamlit_app.py**
   - Lines 1885-1916: Combined subplot view
   - Lines 1970-2010: Standalone displays (reference for working code)
   - Line 2049-2058: DWT display section

2. **src/data_toolkit/signal_analysis.py**
   - Lines 159-220: `plot_wavelet_torrence()` - Convert to Plotly
   - Lines 317-340: `plot_discrete_wavelet()` - Convert to Plotly
   - Keep data computation functions unchanged

3. **src/data_toolkit/timeseries_analysis.py**
   - Lines 486-507: Wrapper functions - Update to call Plotly versions

---

## 🎨 PLOTLY BENEFITS

- ✅ Interactive zoom/pan
- ✅ Hover tooltips with exact values
- ✅ Export to PNG/SVG built-in
- ✅ Responsive to container width
- ✅ Better color scales
- ✅ Consistent styling across all plots
- ✅ No pyplot memory leaks
- ✅ Subplots with shared axes

---

## 💡 NOTES

- **DWT straight lines**: Normal for constant/low-variation data at some levels
  - Test with more complex signal to see variations
  - Consider adding example synthetic signals with multiple components
  
- **FFT tonight's fix**: Should work better but full Plotly will be best
  - Removed axis sharing that was causing scaling issues
  - Added noise floor filtering to highlight peaks

---

## 🚀 ESTIMATED TIME

- CWT Plotly conversion: ~15 minutes
- FFT/PSD/CWT combined subplot: ~20 minutes
- DWT Plotly conversion: ~25 minutes
- Testing and refinement: ~20 minutes
- **Total: ~80 minutes**

---

**Sleep well! The FFT quick fix should help tonight. Full migration tomorrow. 🌙**
