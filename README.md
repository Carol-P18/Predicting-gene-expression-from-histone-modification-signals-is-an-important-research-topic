# Predicting-gene-expression-from-histone-modification-signals

Histone modifications play an important role in affecting gene regulation. Predicting gene expression from histone modification signals is an important research topic. The dataset is on primary T CD8+ naive cells from peripheral blood cell type from the Roadmap Epigenomics Mapping Consortium database. For each gene, there are 100 bins with five core histone modification marks. The 10,000 base pair (bp) DNA region (+/-5000bp) around the transcription start site of each gene was divided into bins of length 100 bp. Reads of 100 bp in each bin were counted. The signal of each gene has a shape of 100x5.
 
Use histone_modifications_dataset.zip. It should contain 2 files: features.csv and output.csv. The goal is to develop a model that can accurately predict gene expression level. High gene expression level corresponds to target label = 1, and low gene expression corresponds to target label = 0. The inputs are 100x5 matrices and target is the probability of gene activity. 
 
References: 
1.	https://en.wikipedia.org/wiki/Histone
2.	https://en.wikipedia.org/wiki/Histone_acetylation_and_deacetylation
3.	https://www.whatisepigenetics.com/histone-modifications/
4.	http://www.roadmapepigenomics.org
5.	Kundaje, A. et al. Integrative analysis of 111 reference human epigenomes. Nature, 518, 317–330, 2015.
Note: Code written using Matlab only. 
