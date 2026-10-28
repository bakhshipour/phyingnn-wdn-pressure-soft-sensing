# Physics-Infused GNN for Pressure Soft Sensing in Water Distribution Networks

Supplementary code and data for the manuscript:

> **Fusing Physics into Graph Structures for GNN-Based Pressure Monitoring in Water Distribution Networks**

This repository provides the notebooks, network models (EPANET `.inp`), and metadata used to generate training/evaluation datasets and to train the physics-infused GNN (PhyInGNN) reported in the paper.

## Contents

```
codes/
  steady_data_generator_physics_infused.ipynb   # Physics-infused graph + dataset generation
  steady_data_generator_nophysic.ipynb          # Baseline (no physics-infused weights) generator
  gnn.py                                       # GNN model utilities
  requirements.txt
data/
  data/
    Alperovits/ Anytown/ Baghmalek/ Bak/ Hanoi/ Kadu/ Modena/ Zhijiang/
      *.inp              # EPANET network models
      *_metadata.json    # Dataset metadata
      *_stats.json       # Normalization / summary statistics
```

Large PyTorch dataset binaries (`.pt`) are **not** tracked in Git (GitHub file-size limits). They can be regenerated with the notebooks above, or requested from the corresponding author.

## Hydraulic edge weights (Hazen–Williams)

Pipe resistances used for shortest-path / message-passing weights follow the flow-independent Hazen–Williams form used in the manuscript:

\[
r_e = \frac{L}{f^{1.852}\, D^{4.87}}
\]

where \(f\) is the Hazen–Williams roughness coefficient, \(L\) is length, and \(D\) is diameter. Resistances are then max-normalized within each network. Absolute units cancel under normalization; values are taken consistently from the EPANET model attributes.

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r codes/requirements.txt
jupyter notebook codes/steady_data_generator_physics_infused.ipynb
```

Update the `inp_file` path and sensor / demand settings in the notebook for the target network.

## Citation

If you use this code or data, please cite the associated journal article (Discover Applied Sciences).

## License

Research code released for reproducibility of the published results. See the manuscript for authorship and funding details.

## Contact

Corresponding author: Amin E. Bakhshipour (`amin.bakhshipour@rptu.de`)
