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
The requirements are stored in the main folder (GeophysicalTools/) as `.yml` files, used to describe an environment — that is, all the packages and dependencies used to install them.  
You can simply run the following commands:
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

### --- 💻 Solving errors
If you encounter errors when creating the environments, you can manually create and install them as follows:

#### 1. electro
```bash
conda create -n electro -c gimli -c conda-forge "pygimli>=1.5.0"
conda activate electro
conda install jupyterlab
pip install resipy emagpy
```
#### 2. seismic
```bash
conda create -n seismic -c gimli -c conda-forge "pygimli>=1.5.0"
conda activate seismic
conda install jupyterlab
pip install scipy obspy disba evodcinv tqdm 
```
#### 3. gpr
```bash
conda create -n gpr python=3.11
conda activate gpr
conda install jupyterlab
pip install gprpy
```
If any errors occur due to missing dependencies or libraries, you can always use `pip` which lets you download, install, upgrade, and remove packages:
```bash
pip install packa_gename
```
### --- 


### 4. Launch Jupyter
Once the desired environment is active, start Jupyter:
```bash
jupyter lab
```

---
## 📘 Basic Python & Jupyter Instructions
For dummy users, a quick guide is available here:  
[📄 Python & Jupyter Guide](PythonJupyter.pdf)
