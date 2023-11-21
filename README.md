# SELECTOR: Heterogeneous graph network with convolutional masked autoencoder for multi-modal robust prediction of cancer survival

## Abstract

Accurately predicting the survival rate of cancer patients is very important to help clinicians plan appropriate treatment, saving cancer patients' related medical expenses and improving their quality of life to a large extent. Multi-modality prediction of cancer patient survival can provide more comprehensive and accurate results. However, existing methods still need to fully solve the problems of missing multi-modal data and information interaction within modalities. This paper proposes a heterogeneous graph-aware network based on convolutional mask encoders for multi-modal robust prediction of cancer patient survival, named SELECTOR. It mainly consists of feature edge reconstruction, convolutional mask encoder, feature cross-fusion and multi-modal survival prediction modules. First, we construct a multi-modal heterogeneous graph and use the meta-path method to perform edge reconstruction of features to fully account for the graph edges' feature information and the nodes' effective embedding. To prevent the impact of missing features within the modality on prediction accuracy, we designed a convolutional masked autoencoder (CMAE) to process the heterogeneous graph after feature reconstruction. Secondly, the feature cross-fusion module establishes communication between multi-modalities so that the output features include all features of the modality as well as relevant information about other modalities. Extensive experiments and analysis on six cancer cohorts from TCGA show that our method significantly outperforms state-of-the-art methods in both modality-missing and intra-modality information-confirmed cases.

![Uploading 111111.jpg…]()



## Table of Contents

- Installation
- Quick start
- Contributing
- Cite
- Contacts
- Licence

## Installation

```markdown
Python 3.9.16
torch 1.13.1+cu116
numpy 1.22.4
Please see piplist to install other basic python 
```

## Data description

```markdown
TCGA :
1.Diagnosing WSI
2.RNA data
3.Clinical data
```

## Train

```python
python ./HG-FCMAE/trainour.py
```

## Notice

All codes will be updated after paper accepted.

## Contacts

If you have any questions or comments, please feel free to email: [panlr@hnu.edu.cn].

## License

MIT © Richard McRichface.
