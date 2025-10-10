# GeophysicalTools
 Teaching material and exercises

## ⚙️ Setup Instructions
To ensure reproducibility and modularity, three separate conda environments are used — one for each main module.


### 1. Clone the repository
```bash
git clone https://github.com/AlbCa/GeophysicalTools.git
cd GeophysicalTools
```
### 2. Create the environments
The requirements files are stored in the main folder (GeophysicalTools/).  
Run the following commands to create the environments:
```bash
conda create --name seismic --file seismic_requirements.txt
conda create --name electro --file electro_requirements.txt
conda create --name gpr --file gpr_requirements.txt
```
### 3. Activate the environment
Before running a notebook, activate the corresponding environment:  
```bash
conda activate seismic    # for seismic methods
conda activate electro    # for electro-magnetic methods
conda activate gpr  # for gpr
```
### 4. Launch Jupyter
Once the desired environment is active, start Jupyter:
```bash
jupyter lab
```

---
## 📘 Basic Python & Jupyter Instructions
For dummy users, a quick guide is available in ` python&jupyter.pdf` 
