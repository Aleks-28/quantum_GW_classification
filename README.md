# Gravitational Wave Signal Classification Using Projected Quantum Kernels

## 📊 Project Overview

This repository implements a two-level feature extraction method for gravitational wave (GW) signal classification, combining classical convolutional autoencoders with Projected Quantum Kernel (PQK) techniques. The primary focus is evaluating the applicability of quantum computing methods for binary black hole (BBH) merger detection.

### Why This Matters

Gravitational waves (GW) are one of the most exciting discoveries of
modern astrophysics. Significant amount of information on the source of
GW signals can be extracted in order to describe astrophysical objects.
For example, the analysis and classification of these signals can offer
a powerful detection tool for binary black hole coalescences. GW signals
are noise-dominated times series, making their detection a difficult
task. In this work we present a method of a two-level feature extraction
from GW signals that uses classical convolutional auto-encoders followed
by Projected Quantum Kernel-based feature extractor. The goal of this
study is to assess if there is hope for application of these particular
methods for BBH mergers detection. We use the dataset, recently
published on Kaggle and produced by the G2-NET collaboration.
We follow the approach published in Huang et al. Nat. Comm. 12:2631 (2021).

## 🚀 Features

- Complete pipeline for gravitational wave signal preprocessing and classification
- Implementation of convolutional autoencoders for dimensionality reduction
- Projected Quantum Kernel implementation in PennyLane
- Alternative implementation using TensorFlow Quantum based on methodolthe notebook from [Huang et al. Nature Communications 12:2631 (2021)](https://www.nature.com/articles/s41467-021-22847-0)

## 📋 Pipeline Overview

Our approach follows this workflow:

1. **Data Preprocessing**: Extract and prepare GW signals from the G2NET dataset
2. **Dimensionality Reduction**: Apply convolutional autoencoders to compress signal data
3. **Quantum Feature Extraction**: Process reduced data through Projected Quantum Kernels
4. **Classification**: Train and evaluate models on the quantum-enhanced features

## 📊 Dataset

This project uses the G2NET gravitational wave detection dataset from Kaggle:

1. Download the dataset from the [G2NET Kaggle competition](https://www.kaggle.com/competitions/g2net-gravitational-wave-detection/overview)
2. Extract the dataset to the `data/` directory (or modify the configuration to point to your data location)

## 🔬 Usage

### Data Preprocessing

```bash
# Run the preprocessing script
python utils/preprocessing.py
```

### Autoencoder Training

```bash
# Run the main script for autoencoder training
python src/main.py
```

### Feature Extraction

```bash
# Extract features from preprocessed signals
python src/features_extraction.py
```

### Quantum Processing

Choose one of the quantum processing methods:

- **PennyLane Implementation**:
  ```bash
  python quantum/PQK_logic.py
  ```

- **TensorFlow Quantum Implementation**:
  Run the Jupyter notebook at `quantum/quantum_data_GW.ipynb`

## 📝 Configuration

Modify configuration parameters in `config.yml` to adjust:
- Data paths
- Model hyperparameters
- Training settings
- Quantum circuit design

## 📊 Results and Visualization

After running the pipeline, you can visualize results using:

```bash
python src/visualize_results.py
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{huang2021power,
  title={Power of data in quantum machine learning},
  author={Huang, Hsin-Yuan and Broughton, Michael and Mohseni, Masoud and Babbush, Ryan and Boixo, Sergio and Neven, Hartmut and McClean, Jarrod R},
  journal={Nature communications},
  volume={12},
  number={1},
  pages={2631},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or support, please open an issue or contact [your-email@example.com].