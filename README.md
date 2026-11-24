# Demo app: Adapting Without Forgetting

Self-contained (21 MB of data in `data/`, no GPU, no dataset needed). Everything shown is precomputed from our runs.

    pip install -r requirements.txt
    streamlit run app.py

Tabs: 1 try the models on real test images · 2 change method / scope / memory size / budget / order and see the result ·
3 what the memory stores at each image size · 4 why replay works (BatchNorm diagnostics).

Regenerate `data/` on the cluster with `python src/export_app_data.py`.
