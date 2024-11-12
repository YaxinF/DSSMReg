library(AUCell)
library(Seurat)
library(dplyr)
library(GSEABase)
#读取scRNA-seq和meta
rt=read.table("HSC.csv",sep=",",header=T,check.names=F)
rt=as.matrix(rt)
rownames(rt)=rt[,1]
exp=rt[,2:ncol(rt)]
dimnames=list(rownames(exp),colnames(exp))
data=matrix(as.numeric(as.matrix(exp)),nrow=nrow(exp),dimnames=dimnames)

#将矩阵转换为Seurat对象，并对数据进行过滤
pbmc <- CreateSeuratObject(counts = data,project = "seurat", min.cells = 3, min.features = 200, names.delim = "_",)
pbmc[["RNA"]] <- as(object = pbmc[["RNA"]], Class = "Assay")

#aucell 
cells_rankings <- AUCell_buildRankings(pbmc@assays$RNA@data) 
cells_rankings
folder_path <- "E:\\DSSMReg\\regulon\\regulon_HSC"  # 替换为实际的文件夹路径
file_names <- list.files(folder_path, pattern = "\\.txt$", full.names = TRUE)
extraGeneSets <- list()
for (file_name in file_names) {
  gene_list <- read.table(file_name, header=FALSE, sep="\t", stringsAsFactors=FALSE)
  gene_list <- gene_list$V1
  set_name <- basename(file_name)
  gene_set <- GeneSet(gene_list, setName = set_name)
  extraGeneSets <- c(extraGeneSets, gene_set)
}
geneSets <- GeneSetCollection(c(extraGeneSets))
names(geneSets)
cells_AUC <- AUCell_calcAUC(geneSets, cells_rankings)
cells_AUC
set.seed(123)
par(mfrow=c(3,3)) 
lapply(1:9, function(i) print(hist(as.numeric(getAUC(cells_AUC)[ names(geneSets)[[i]], ]) )))
a<-as.data.frame(cells_AUC@assays@data@listData[["AUC"]])

#计算所有细胞的平均值
a$mean <- apply(a, 1, mean)
# 根据平均值排序，获取排序后的行名调控子名
prioritize <- rownames(a)[order(a$mean)]
a <- cbind(a, mean = a$mean, prioritize = prioritize)

write.table(a,"E:\\DSSMReg\\regulon\\aucell.csv",row.names=TRUE,col.names=NA,sep=",") 
