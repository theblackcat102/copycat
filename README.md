# Copycat

```
pip install colossalai==0.1.12+torch1.12cu11.3 -f https://release.colossalai.org
pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers
```

Test running colossalai code

```bash
colossalai run --nproc_per_node 1 colossalai_test.py --use_trainer
```

# Run T5 version of the model

You will need to install the version from here:

[PhungVanDuy/trlx](https://github.com/PhungVanDuy/trlx/tree/add_t5)

# Notes

[WebGPT arxiv](https://arxiv.org/pdf/2112.09332.pdf)

    - [blog](https://openai.com/blog/webgpt/)

[Instruct GPT](https://arxiv.org/pdf/2203.02155.pdf)

[RLHF - Summary](https://arxiv.org/pdf/2009.01325.pdf)

[Longformer](https://arxiv.org/pdf/2004.05150.pdf)

[Recross](https://arxiv.org/pdf/2204.07937.pdf)

[RETRO](https://arxiv.org/pdf/2112.04426.pdf)
