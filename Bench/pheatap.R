# Load libraries
library(pheatmap)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(pheatmap)
library(viridis)  # for colorblind-friendly palettes

# Helper: build named colors for whatever levels are present
mk_anno_colors <- function(levels, defaults) {
  # Keep defaults that are actually present
  got <- defaults[names(defaults) %in% levels]
  # For any missing levels, assign new distinct colors
  missing <- setdiff(levels, names(got))
  if (length(missing)) {
    extra <- viridis(length(missing))
    names(extra) <- missing
    got <- c(got, extra)
  }
  got
}

make_heatmap <- function(csv_path, cohort, out_dir = ".") {
  # 1) Load
  df <- read.csv(csv_path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!all(c("Feature","FeatureType") %in% names(df))) {
    stop("CSV must contain 'Feature' and 'FeatureType' columns.")
  }
  # Normalize OmicsLevel column name
  if ("OmicsLevels" %in% names(df)) {
    df$OmicsLevel <- df$OmicsLevels
  } else if (!"OmicsLevel" %in% names(df)) {
    df$OmicsLevel <- "Unknown"
  }
  
  # 2) Numeric matrix (no scaling)
  num_cols <- names(df)[sapply(df, is.numeric)]
  num_cols <- setdiff(num_cols, c("Feature","FeatureType","OmicsLevel","OmicsLevels"))
  if (!length(num_cols)) stop("No numeric columns found for heatmap.")
  mat <- as.matrix(df[, num_cols, drop = FALSE])
  rownames(mat) <- df$Feature
  
  # 3) Row annotations
  annotation_row <- data.frame(
    FeatureType = df$FeatureType,
    OmicsLevels = df$OmicsLevel,
    row.names = df$Feature,
    check.names = FALSE
  )
  
  # 4) Annotation colors (defaults include Metab & Prot)
  feature_type_defaults <- c(
    mRNA = "#56B4E9",  # sky blue
    miRNA = "#CC79A7", # reddish purple
    Meth = "#E69F00",  # orange
    Metab = "#009E73", # bluish green
    Prot = "#0072B2"   # blue
  )
  omics_level_defaults <- c(
    "Single+Dual+Triple" = "#6A3D9A",
    "Dual+Triple"        = "#FF7F00",
    "Single+Dual"        = "#B15928",
    "Single"             = "#1B9E77",
    "Unknown"            = "#999999"
  )
  
  anno_colors <- list(
    FeatureType = mk_anno_colors(unique(annotation_row$FeatureType), feature_type_defaults),
    OmicsLevels = mk_anno_colors(unique(annotation_row$OmicsLevels), omics_level_defaults)
  )
  
  # 5) Integer cell text (keep blanks for NA)
  mat_int <- round(mat)
  text_mat <- matrix("", nrow = nrow(mat_int), ncol = ncol(mat_int), dimnames = dimnames(mat_int))
  mask <- !is.na(mat_int)
  text_mat[mask] <- sprintf("%d", mat_int[mask])
  
  # 6) Output paths
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  png_path <- file.path(out_dir, paste0(cohort, "-RankHeatmap.png"))
  pdf_path <- file.path(out_dir, paste0(cohort, "-RankHeatmap.pdf"))
  
  # 7) Save PNG
  pheatmap(
    mat,
    scale = "none",
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    annotation_row = annotation_row,
    annotation_colors = anno_colors,
    show_rownames = TRUE,
    display_numbers = text_mat,
    number_color = "black",
    fontsize_number = 8,
    angle_col = 45,
    border_color = "grey80",
    color = viridis(100),
    filename = png_path,
    width = 12, height = 8, units = "in", dpi = 300
  )
  
  # 8) Save PDF
  pheatmap(
    mat,
    scale = "none",
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    annotation_row = annotation_row,
    annotation_colors = anno_colors,
    show_rownames = TRUE,
    display_numbers = text_mat,
    number_color = "black",
    fontsize_number = 8,
    angle_col = 45,
    border_color = "grey80",
    color = viridis(100),
    filename = pdf_path,
    width = 12, height = 8
  )
  
  message("Saved: ", png_path, " and ", pdf_path)
}

# --- Run for each cohort (adjust paths as needed) ---
make_heatmap("plot_data_ROSMAP.csv",     "ROSMAP",     out_dir = ".")
make_heatmap("plot_data_MayoRNASeq.csv", "MayoRNASeq", out_dir = ".")  # handles Metab/Prot/mRNA
make_heatmap("plot_data_BRCA.csv",       "BRCA",       out_dir = ".")
