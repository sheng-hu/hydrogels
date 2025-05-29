### Descriptions
- This is the data mining part for reproducing the submission "Data-Driven De Novo Design of Super-Adhesive Hydrogels."
### Datasets
- The raw datasets are all gp formats in `sequence/`.
For example:
```
sequence/adhesive proteins sequence.gp
```

### Usage
The main implementation for protein analysis is in:
- `gb_analysis_200.ipynb`: It reads the protein raw data and saves the top 200 species data into `fa_files/`.
- `fa_files/run_clustalo.sh`: Use clustalo to do multi-sequence alignment (MSA) for each species. Alignment files and consensus files are stored in `aln_files/` and `con_files/`.
- `histogram_visualize_200.ipynb`: It reads the consensus files and conducts pair-wise analysis for further visualizations. The resulted 180 proportions (formulas) are used for wet experiments to construct the initial hydrogel datasets.

