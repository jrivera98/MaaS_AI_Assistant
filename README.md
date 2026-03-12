Setup Step:

1. Setup venv & dependencies  *run each service in ther oun venv
	python -m venv .venv
    .venv\Scripts\activate.bat
    	
    pip install -r requirements.txt

    pip install --upgrade pip;
    pip install setuptools; wheel;
    pip install numpy scipy scikit-learn pandas matplotlib seaborn;
    pip install python-dotenv requests urllib3 chardet;
    pip install aiohttp asyncio-contextmanager;
    
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org{CUDA}.html


2. Configure .env

3. Start Services
    ollama serve

    ollama run *foundation_model*
    ollama run qwen2.5:14b (16GB)
    ollama run qwen2.5:7b (8GB) *recommended*

    
    python *agent_name*.py

    #to kill task at port 
    taskkill /PID {PID} /F

4. create jupyter notebook
    cntrl+shift+p
