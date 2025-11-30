# RALFS
Retrieval Augmented Long-Form Summarization

**Under Developement**

## Project Structure

```

RALFS/\
├── pyproject.toml\
├── README.md\
├── configs/\
│ ├── train/\
│ │ ├── fid_base.yaml\
│ │ ├── retriever.yaml\
│ │ └── reranker.yaml\
│ └── eval/\
│ └── eval.yaml\
├── src/\
│ ├── data/\
│ │ ├── download.py\
│ │ ├── preprocess.py\
│ │ ├── build_index.py\
│ │ └── dataset.py\
│ ├── retriever/\
│ │ ├── encoder.py\
│ │ ├── contriever.py\
│ │ ├── train_retriever.py\
│ │ └── utils.py\
│ ├── reranker/\
│ │ ├── cross_encoder.py\
│ │ └── train_reranker.py\
│ ├── generator/\
│ │ ├── fid_model.py\
│ │ ├── adaptive_fid.py\
│ │ ├── train_generator.py\
│ │ └── inference.py\
│ ├── faithfulness/\
│ │ ├── egf_checker.py\
│ │ ├── qa_checker.py\
│ │ └── loss.py\
│ ├── eval/\
│ │ ├── metrics.py\
│ │ ├── comprehensive_metrics.py\
│ │ └── human_eval_tools.py\
│ ├── utils/\
│ │ ├── logging.py\
│ │ ├── io.py\
│ │ └── config.py\
│ └── cli.py\
├── scripts/\
│ ├── run_train_retriever.sh\
│ ├── run_train_generator.sh\
│ └── run_eval.sh\
├── tests/\
│ ├── test_chunker.py\
│ ├── test_retriever.py\
│ └── test_generator.py\
├── notebooks/\
│ ├── exploration.ipynb\
│ └── ablation_analysis.ipynb\
├── examples/\
│ └── demo_inference.py\
└── docker/\
└── Dockerfile
```
