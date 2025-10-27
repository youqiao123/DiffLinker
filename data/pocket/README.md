# How to prepare Pockets dataset 

Download Binding MOAD:
```
wget http://www.bindingmoad.org/files/biou/every_part_a.zip
wget http://www.bindingmoad.org/files/biou/every_part_b.zip
unzip every_part_a.zip
unzip every_part_b.zip
```

Clean and split raw PL-complexes:
```
export PROCESSED_DIR='/home/qianyouqiao/protac/pdbbind_processed'
export RAW_PDBBIND='/home/qianyouqiao/protac/sc_complexes'

python -W ignore clean_and_split.py --in-dir $RAW_PDBBIND --proteins-dir $PROCESSED_DIR/proteins --ligands-dir $PROCESSED_DIR/ligands
```

Create fragments and conformers:
```
python -W ignore generate_fragmentation_and_conformers.py \
                 --in-ligands $PROCESSED_DIR/ligands \
                 --out-fragmentations $PROCESSED_DIR/generated_splits.csv \
                 --out-conformers $PROCESSED_DIR/generated_conformers.sdf
```

Prepare dataset:
```
python -W ignore prepare_dataset.py \
                 --table $PROCESSED_DIR/generated_splits.csv \
                 --sdf $PROCESSED_DIR/generated_conformers.sdf \
                 --proteins $PROCESSED_DIR/proteins \
                 --out-mol-sdf $PROCESSED_DIR/pdbbind_mol.sdf \
                 --out-frag-sdf $PROCESSED_DIR/pdbbind_frag.sdf \
                 --out-link-sdf $PROCESSED_DIR/pdbbind_link.sdf \
                 --out-pockets-pkl $PROCESSED_DIR/pdbbind_pockets.pkl \
                 --out-table $PROCESSED_DIR/pdbbind_table.csv
```

Final filtering and train/val/test split:
```
python -W ignore filter_and_train_test_split.py \
                 --mol-sdf $PROCESSED_DIR/pdbbind_mol.sdf \
                 --frag-sdf $PROCESSED_DIR/pdbbind_frag.sdf \
                 --link-sdf $PROCESSED_DIR/pdbbind_link.sdf \
                 --pockets-pkl $PROCESSED_DIR/pdbbind_pockets.pkl \
                 --table $PROCESSED_DIR/pdbbind_table.csv
```