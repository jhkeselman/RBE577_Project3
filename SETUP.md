# Setup Instructions

## Prerequisites

- Python 3.6+
- CUDA-compatible GPU (recommended)


## Preferred Pre-requisites

- Python 3.8
- CUDA 12.1 


## Installation

1. Setup the environment using the environment.yml file

```bash
conda env create -f environment.yml --name rl-assignment
```

2. Activate the environment

```bash
conda activate rl-assignment
```

3. Install the dependencies

```bash
pip install -r requirements.txt
```

4. Run the main.py file

```bash
python main.py
```


## Usage

1. Run the main.py file

```bash
python main.py
```

2. Run the main.py file with the desired algorithm

```bash
python main.py --algorithm a3c
```

