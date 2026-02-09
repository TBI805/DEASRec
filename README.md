# DEASRec

**Purify and Generalize: Efficient Dual-End Adapters for Sequential Recommendation**

This repository contains implementation of DEASRec built on the RecBole framework.

## Features

- **Built on RecBole**: Uses the powerful and extensible RecBole recommendation framework
- **Easy to Use**: Simple configuration and training interface

### Requirements

- Python >= 3.7
- PyTorch >= 1.10.0
- See `requirements.txt` for complete dependencies

- Download datasets from "[Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI)". Put the files in ./dataset/

## Quick Start

### Dataset

Download the processed datasets from Recbole Library "[Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI)". Put the files in ./dataset/

### Basic Usage

```python

# Run DEASRec with default configuration
run_recbole(model='DEASRec', dataset='ml-100k')
```

Then run:

```bash
python run_recbole.py
```

## Acknowledgments

This implementation is based on the [RecBole](https://github.com/RUCAIBox/RecBole) recommendation library. We appreciate their outstanding work.
