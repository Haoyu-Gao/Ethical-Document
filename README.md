# Ethical-Document

This is the replication package for "Documenting Ethical Considerations in Open Source AI Models", under submission for ESEM'24.

## Folder Walkthrough
- `qualitative codes` contains our thematic analysis codes for documents from three data sources.
- `mine_repositories` folder contains code for mining repositories on both GitHub and Hugging Face.
- `duplicate_detection` section contains the code we used to detect document reuse.
- `Curated documents` folder contains documents we mined on three data sources. You can cross reference these raw documents with our codes under `qualitative codes` folder.

We will provide a more detailed folder explaination below. 

## qualitative codes
`categories_mindmap.pdf` contains the mindmap describing the hierarchical relationship for how base codes are synthesised into concepts and then categories.

`codes` contains the code number and its description.

`sample_hf_mc` contains the document name, keypoints, codes for the data source of `HF_CARD`.

`sample_gh_rm` contains the document name, keypoints, codes for the data source of `GH_README`.

`sample_gh_mc` contains the document name, keypoints, codes for the data source of `GH_CARD`.


## mine_repositories
This folder contains the mining procedure as well as the keyword expansion procedure. The `keyword-base.txt` is used for initial keyword filtering. After `keyword_filter_expansion.py`, which is used to expand the keyword set based on the process we described in the paper, we get `extra_keywords_paragraph.txt`.


## duplicate_detection
This folder contains the code for detecting duplicate documents. Please refer to section 3.3 in our paper for details. `duplicate_detection.py` for is responsible for generating similarity matrix and performing clustering.

## Curated documents
This folder contains our collected documents. There are three subfolders in it, `github_model_card_documents`, `github_readme_documents`, and `huggingface_model_card_documents` corresponds to terms `GH_CARD`, `GH_README`, and `HF_CARD` respectively in our paper. Please note that this is the raw data before any filters.