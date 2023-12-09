# Flight_Predictions_FDS-Project" 


## Quick Guide for correct Version Control

We are using *jupytext* for version control to better manage git Merge conflicts.

### Setup:
- download jupytext: pip install jupytext
 ```bash
pip install jupytext
  ```
- link your local jupyter Notebook with a .py script: 

```bash
jupytext --set-format ipynb,py notebook.ipynb
```

### Using jupytext for Version Control:
- don't push the notebook.ipynb at all
- after pulling -> your notebook should automatically sync with the updates of the .py file
- if not do it manually:
```bash
jupytext -s notebook.ipynb
```
- before pushing the .py with your new input -> check if it synced
- if not do it manually:

```bash
jupytext -s notebook.py
```
- both commands should update 1 of the files accordingly to the changes of the file last updated and saved
- be careful not to override your work ...