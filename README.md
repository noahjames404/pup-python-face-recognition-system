# Face Recognition System
Developed for Elementary Schools and communities alike to monitor & track their attendance by their respective guardians/parents

## Prerequisites
- [git](https://git-scm.com/downloads)
- [Python 3.9](https://www.python.org/downloads/release/python-390/)


## Installation
- Setup local environment inside the project. Ensure you're using python 3.9, any versions would result to incompatibilities.  
```
    <path/to/python 3.9> -m venv local_env
```

- Install required dependencies
```
    pip install "./setup.sh"
```

- duplicate `env.py.example` file inside the project, and rename it to `env.py`. Setup the appropriate API by navigating to your Supabase Database, Project Settings, API Settings. 

## Execution
```
    python main.py
```