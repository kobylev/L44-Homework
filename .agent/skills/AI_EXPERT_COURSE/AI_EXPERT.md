---
name: AI_EXPERT_COURSE
description: Applies strict course guidelines for modular project structure, reproducible virtual environments, and highly detailed README documentation.
---

# Course Project Standards Skill

This skill ensures that all course projects adhere to the strict guidelines demonstrated in the L42 reference repository. Whenever you are tasked with creating, structuring, or finalizing a project for this course, you MUST implement the following standards.

## 1. Project Structure
Projects must be modular. Avoid monolithic scripts. Use a structure similar to:
- `config.py` - Hyperparameters, paths, and device configuration.
- `datasets.py` - Data loading, dataset splitting, and preprocessing.
- `model.py` - Neural network architectures.
- `train.py` - Training loops, loss calculations, and checkpointing.
- `evaluate.py` - Evaluation metrics, testing, and output generation.
- `main.py` - Orchestration of the full pipeline.
- `requirements.txt` - All Python dependencies.
- `docs/` or `assets/` - Folder dedicated to README images, diagrams, and result 
- `code/`  use seperated folder to place the code insidevisualizations.

## 2. Environment & Requirements
- Always include a `requirements.txt` file with appropriately scoped versions.
- Provide explicit setup instructions in the README using `venv`. Include commands for both macOS/Linux and Windows.

## 3. Comprehensive README.md
Every project MUST include a highly detailed `README.md` containing all of the following sections:
- **Title and Overview**: A clear title and 1-2 sentence description.
- **The Core Idea**: A high-level explanation, including a structured text breakdown or intuition of the problem being solved.
- **Project Structure**: An ASCII tree representation of the codebase.
- **Data Flow / Architecture**: Diagrams (text-based ASCII or Mermaid) showing how data moves through the system or detailing the network architecture (layers, dimensions, transformations).
- **Results**: Visual proof of the project's output, utilizing images stored in the `docs/` or `assets/` directory. Include loss curves, metrics, and example inferences.
- **Honest Assessment**: A critical evaluation section detailing what worked, what didn't produce meaningful results, and *why* (e.g., underfitting, loss function choices, data complexity). Avoid generic praise; be rigorously analytical.
- **What Needs to Be Done (Next Steps)**: A table or structured list of solutions to the issues identified in the Honest Assessment.
- **Setup & Usage**: Step-by-step terminal commands to create the virtual environment, install requirements, and execute the main script.
- **Dataset**: Attribution, licensing, and reference to the source of the data used.

## 4. Execution Workflow
When activated to apply these guidelines to an existing or new project:
1. Review the current project files and user objectives.
2. Structure or refactor the code into the modular files defined above (`config.py`, `model.py`, etc.).
3. Generate or update `requirements.txt`.
4. Draft the comprehensive `README.md` ensuring NO required section is missing. If results are pending, explicitly create placeholders for images.
5. Emulate the critical tone of the "Honest Assessment" based on the actual training/evaluation results of the project.
