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

