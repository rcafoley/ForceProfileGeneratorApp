# Force Profile Generator for Neuromuscular Fatigue Modeling

**Version:** 2.0 (Streamlit Implementation)  
**Author:** Ryan C. A. Foley, PhD Candidate, CSEP Clinical Exercise Physiologist  
**Laboratory:** Occupational Neuromechanics and Ergonomics (ONE) Laboratory  
**Institution:** Ontario Tech University, Oshawa, Ontario, Canada  

## 🚀 Live Demo

Access the web application: [https://force-profile-generator.streamlit.app](https://force-profile-generator.streamlit.app)

## 📋 Purpose

This application generates force-time profiles for use in neuromuscular fatigue modeling. It allows researchers and clinicians to:
- Create complex force profiles with multiple subtask types
- Visualize force patterns over time
- Combine multiple profiles for complex work simulations
- Export data for further analysis

## ✨ Features

- **Multiple Subtask Types:** Constant force, linear slopes, ramps, precision work, and rest periods
- **Visual Profile Builder:** Intuitive interface with real-time preview
- **Profile Management:** Save, load, and combine multiple profiles
- **Data Import/Export:** Excel file support for data exchange
- **Real-time Visualization:** Interactive Plotly charts
- **Configurable Sample Rate:** 1-1000 Hz support

## 🔬 Precision Force Assumptions

The **Precision** subtask type estimates shoulder muscle activation during precision manual work tasks, accounting for postural demands of maintaining shoulder flexion during fine motor activities.

Force values for different shoulder flexion angles are derived from:
1. **Published Research:** Brookham, R. L., Wong, J. M., & Dickerson, C. R. (2010). Upper limb posture and submaximal hand tasks influence shoulder muscle activity. *International Journal of Industrial Ergonomics*, *40*(3), 337-344.
2. **Unpublished Data:** Overhead work EMG data from the ONE Laboratory (Ontario Tech University)

Implemented values represent anterior deltoid activation as percentage of maximum voluntary contraction (%MVC):
- **0-45° shoulder flexion:** 5% MVC
- **45-90° shoulder flexion:** 12% MVC
- **>90° shoulder flexion:** 13.5% MVC

## 🛠️ Installation for Local Use

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/rcafoley/force-profile-generator.git
cd force-profile-generator
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit-force-profile-generator-v7.py
```

4. Open your browser to `http://localhost:8501`

## 📖 Usage Instructions

1. **Profile Builder:** Start by adding subtasks to create your force profile
   - Select subtask type (Force, Slope, Ramp, Precision, Rest)
   - Set duration and force parameters
   - Add descriptions for each subtask

2. **Visualize & Store:** View your profile and save it for later use
   - Real-time visualization of force patterns
   - Save profiles for reuse
   - Export options for data and templates

3. **Import Data:** Load existing force profiles from Excel files
   - Auto-detection of data columns
   - Support for various Excel formats

4. **Combine Profiles:** Merge multiple profiles for complex scenarios
   - Sequential combination of saved profiles
   - Visual preview of combined output

5. **Export Data:** Export combined profiles for analysis
   - Multi-sheet Excel output with metadata
   - Preserves all timing and force data

## 📊 Technical Details

- **Sample Rate:** Configurable from 1-1000 Hz
- **Force Units:** Percentage of Maximum Voluntary Contraction (%MVC)
- **Time Units:** Seconds
- **Export Format:** Excel (.xlsx) with metadata, subtask details, and force profile data

## 📧 Contact

- **Website:** [rcafoley.github.io](https://rcafoley.github.io)
- **GitHub:** [github.com/rcafoley](https://github.com/rcafoley)
- **Academic Email:** ryan.foley@ontariotechu.ca
- **Research Email:** ryan@afferentresearch.com

## 📄 License

© 2023-2025 Ryan C. A. Foley. All rights reserved.

## 🙏 Acknowledgments

Developed at the Occupational Neuromechanics and Ergonomics (ONE) Laboratory, Ontario Tech University.

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{foley2025forceprofiling,
  author = {Foley, Ryan C. A.},
  title = {Force Profile Generator for Neuromuscular Fatigue Modeling},
  year = {2025},
  version = {2.0},
  institution = {Ontario Tech University},
  url = {https://github.com/rcafoley/force-profile-generator}
}
```