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
### 2. 🔧 Create the environments
The requirements are stored in the main folder (GeophysicalTools/) as `.yml` files, used to describe an environment — that is, all the packages and dependencies used to install them. 

On *Windows* and *Linux*, you can simply run the following commands:
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

### --- 💻 Note for macOS users 
Some dependencies used in the `.yml` files may not be available for macOS or could require compilation tools that are not pre-installed.  
If you encounter errors when creating the environments, you can manually create and install them as follows:

#### 1. Create a new conda environment
```bash
conda create -n electro -c gimli -c conda-forge "pygimli>=1.5.0"
conda activate electro
```
#### 2. Install dependencies manually
```bash
pip install -r electro_mac.txt
```
Repeat the same procedure for the other environments (`seismic`, `gpr`), changing name and file accordingly. There's no need to install `pyGIMLi` anymore, thus you'll just run:
```bash
conda create -n envname python=3.11
conda activate envname
pip install -r envname_mac.txt
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
