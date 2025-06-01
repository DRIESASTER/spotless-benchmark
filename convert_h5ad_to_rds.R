# 필요한 라이브러리 로드
library(Seurat)
library(anndataR)
# 파일 경로 설정
input_dir <- "C:/Users/hyoju/OneDrive/Desktop/convert_h5ad_to_rds/"
output_dir <- "C:/Users/hyoju/OneDrive/Desktop/convert_h5ad_to_rds/"


h5ad_files <- "C:/Users/hyoju/OneDrive/Desktop/convert_h5ad_to_rds/liver_mouseStSt_noEC_9celltypes_annot_cd45.h5ad"

for (file in h5ad_files) {
  
  file_name <- tools::file_path_sans_ext(basename(file))
  
  seurat_obj <- read_h5ad(file, to = "Seurat")
  
  # .rds 파일로 저장
  saveRDS(seurat_obj, file = paste0(output_dir, file_name, ".rds"))
}

