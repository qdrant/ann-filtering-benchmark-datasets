
#!/usr/bin/env bash


rsync -avP \
   --exclude='data' \
   --exclude='models' \
   --exclude='lightning_logs' \
   --exclude='venv' \
   --exclude='__pycache__' \
   --exclude='frontend' \
   --exclude='.idea' \
   . $1:./projects/ann-filtering-benchmark-datasets/
