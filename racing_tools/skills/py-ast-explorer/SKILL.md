---
name: py-ast-explorer
description: Use when analyzing Python code structure, understanding call hierarchies, or mapping function/class relationships across the codebase
---

# Py AST Explorer

## Overview

Parses all Python files and prints a tree of classes/functions/variables with their call hierarchies using Python's `ast` module and `rich` for beautiful output.

## Usage

```bash
python skills/py-ast-explorer/ast_explorer.py [path]
```

Default path: current directory (`.`)

## Example Output

```
overlay.py
├── VAR: REPO_ROOT
│     -> [Path, resolve]
├── CLASS: PredictiveLapModel
│     -> 
│   ├── DEF: __init__
│   │     -> 
│   └── DEF: get_time
│         -> [float, interp]
├── DEF: load_session
│     -> [load]
└── ...

camera/model.py
└── CLASS: CameraModel
      -> [FileNotFoundError, append, array, ...]
    ├── DEF: __init__
    │     -> 
    ├── DEF: load
    │     -> [FileNotFoundError, append, array, ...]
    └── DEF: __repr__
          -> 
```

## Features

- **VAR** - global variables
- **CLASS** - class definitions
- **DEF** - function definitions
- **ASYNC** - async function definitions
- **->** - list of function calls within the scope
- Tree structure with proper indentation
- Color output (blue: paths, cyan: types, yellow: names)
- Auto-excludes `.venv` and `venv` directories
- Auto-excludes the script itself

## Script Location

`skills/py-ast-explorer/ast_explorer.py`
