# GeophysicalTools
 Teaching material and exercises

## ⚙️ Setup Instructions
To ensure reproducibility and modularity, three separate conda environments are used — one for each main module.


### 1. Clone the repository
Open your terminal/Anaconda prompt and run this:
```bash
git clone https://github.com/AlbCa/GeophysicalTools.git
cd GeophysicalTools
```
### 2. Create the environments
The requirements are stored in the main folder (GeophysicalTools/) as `.yml` files, used to describe an environment — that is, all the packages, dependencies, Python version, and sometimes even the channels (repositories) used to install them. 
Run the following commands to create the environments:
```bash
conda env create -f electro.yml		# for electro-magnetic methods
```
```bash
conda env create -f seismic.yml		# for seismic methods
```
```bash
conda env create -f gpr.yml			# for gpr
```

### 3. Activate the environment
Before running a notebook, activate the corresponding environment:  
```bash
conda activate electro
```
```bash
conda activate seismic
```
```bash
conda activate gpr
```

### 4. Launch Jupyter
Once the desired environment is active, start Jupyter:
```bash
jupyter lab
```

---
## 📘 Basic Python & Jupyter Instructions
For dummy users, a quick guide is available here:  
[📄 Python & Jupyter Guide](PythonJupyter.pdf)
