# datasets="
# ../../Datasets/Set5
# ../../Datasets/Set14
# ../../Datasets/BSDS100
# ../../Datasets/BSDS200
# ../../Datasets/Kodak
# ../../Datasets/urban100
# ../../Datasets/manga109
# ../../Datasets/lfw/lfw_valid
# ../../Datasets/DIV2K/DIV2K_valid_HR
# "
datasets="
../../Datasets/lfw/lfw_train
"
datasets=$(echo $datasets | tac)
for dataset in $datasets; do
    quality=50
    estimate="mod_spd"
    dataset_path="${dataset}"
    org_imgs_path="${dataset}_org_${quality}_${estimate}"
    std_imgs_path="${dataset}_std_${quality}_${estimate}"
    ehc_imgs_path="${dataset}_ehc_${quality}_${estimate}"
    rec_imgs_path="${dataset}_rec_${quality}_${estimate}"
    python3 ndcjpeg.py --dataset_path $dataset_path --org_imgs_path $org_imgs_path --std_imgs_path $std_imgs_path --ehc_imgs_path $ehc_imgs_path --rec_imgs_path $rec_imgs_path --quality $quality --estimate $estimate --output --grayscale
done
# # dataset="../../Datasets/lfw/lfw_valid"
# # dataset="../../Datasets/urban100"
# dataset="../../Datasets/DIV2K/DIV2K_train_HR"
# weight=256
# height=256
# quality=50
# estimate="mod_spd"
# dataset_path="${dataset}"
# org_imgs_path="${dataset}_org_${quality}_${estimate}"
# std_imgs_path="${dataset}_std_${quality}_${estimate}"
# ehc_imgs_path="${dataset}_ehc_${quality}_${estimate}"
# rec_imgs_path="${dataset}_rec_${quality}_${estimate}"
# python3 ndcjpeg.py --dataset_path $dataset_path --org_imgs_path $org_imgs_path --std_imgs_path $std_imgs_path --ehc_imgs_path $ehc_imgs_path --rec_imgs_path $rec_imgs_path --weight $weight --height $height --quality $quality --estimate $estimate --output --grayscale #--resize # --multiprocess