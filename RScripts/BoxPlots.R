# --- Plotting and visualization ---
library(ggplot2)      # base plotting system
library(ggpubr)       # publication-ready plots
library(cowplot)      # combine multiple ggplots
library(patchwork)    # alternative for combining ggplots
library(gridExtra)    # arranging multiple plots
library(pheatmap)     # heatmaps
library(ComplexUpset)       # upset plots 
library(ggrepel) 
library(stringr)
library(colorspace)


# --- Color palettes ---
library(viridis)      # colorblind-friendly palettes
library(RColorBrewer) # additional palettes

# --- Data wrangling ---
library(dplyr)        # data manipulation
library(tidyr)        # data reshaping

# Set working directory
setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))

df <- read.csv("BENCHMARKING/Cross_Validation_Results.csv")  
df$Cohort[df$Cohort == "MayoRNASeq"] <- "PSP"

# Make a named vector of replacements
model_names_map <- c(
  "AdaBoost Classifier"    = "AdaBoost",
  "CatBoosting Classifier" = "CatBoost",
  "Decision Tree"          = "D.Tree",
  "Gradient Boosting"      = "Gradient Boost",
  "Logistic Regression"    = "L.Regression",
  "MLPClassifier"          = "MLP",
  "MOGONET"                = "MOGONET",
  "MORE"                   = "MORE",
  "Random Forest"          = "Random Forest",
  "SVC"                    = "SVM",
  "XGBClassifier"          = "XGBoost"
)

# Canonical label map (raw -> pretty)
label_map <- c(
  "NONE"             = "None",
  "MORE:Ranker"      = "MORE-Ranker",
  "MOGONET:Ranker"   = "MOGONET-Ranker",
  "shap"             = "SHAP",
  "lime"             = "LIME",
  "t_test"           = "T-Test",
  "RF-FI"            = "RF-FI",
  "XGB-FI"           = "XGB-FI",
  "RF-PFI"           = "RF-PFI",
  "XGB-PFI"          = "XGB-PFI",
  "lasso"            = "LASSO",
  "ridge"            = "Ridge",
  "elasticnet"       = "Elastic Net",
  "boruta"           = "Boruta",
  "mannwhitneyu"     = "Mann-Whitney U",  # en dash
  "svm_rfe"          = "SVM-RFE",
  "mean_rank"        = "Mean rank",
  "min_rank"         = "Min rank",
  "median_rank"      = "Median rank",
  "rra_rank"         = "RRA",
  "geom.mean_rank"   = "Geomean rank",
  "stuart_rank"      = "Stuart",
  "mra_rank"         = "MRA",
  "mean_weight"      = "Mean weight",
  "max_weight"       = "Max weight",
  "median_weight"    = "Median weight",
  "geom.mean_weight" = "Geomean weight",
  "ta_weight"        = "TA"
)

metatable <- data.frame(
  featureSelector = unname(label_map),
  SelectorType = c(
    # NONE
    "Single",
    # MORE/MOGONET rankers
    "Single","Single",
    # shap, lime
    "Single","Single",
    # t_test
    "Single",
    # RF-FI, XGB-FI, RF-PFI, XGB-PFI
    "Single","Single","Single","Single",
    # lasso, ridge, elasticnet, boruta
    "Single","Single","Single","Single",
    # mannwhitneyu, svm_rfe
    "Single","Single",
    # rank-based ensembles
    "Ensemble","Ensemble","Ensemble","Ensemble","Ensemble","Ensemble","Ensemble",
    # weight-based ensembles
    "Ensemble","Ensemble","Ensemble","Ensemble","Ensemble"
  ),
  basis = c(
    "Baseline",                 # NONE
    "Model","Model",            # MORE/MOGONET rankers
    "Weight","Weight",          # SHAP/LIME
    "Weight",                   # t-test
    "Weight","Weight","Weight","Weight",   # RF/XGB importances & PFI
    "Model","Model","Model","Weight",      # LASSO/Ridge/EN, Boruta
    "Weight","Rank",           # Mann–Whitney U, SVM-RFE
    # rank-based ensembles
    "Rank","Rank","Rank","Rank","Rank","Rank","Rank",
    # weight-based ensembles
    "Weight","Weight","Weight","Weight","Weight"
  ),
  category = c(
    "Baseline",
    "Deep ranker","Deep ranker",
    "Explainer","Explainer",
    "Univariate test",
    "Tree importance","Tree importance","Permutation importance","Permutation importance",
    "Linear model (embedded)","Linear model (embedded)","Linear model (embedded)","Wrapper (RF/Boruta)",
    "Univariate test","Wrapper (RFE)",
    "Rank aggregation","Rank aggregation","Rank aggregation","Rank aggregation","Rank aggregation","Rank aggregation","Rank aggregation",
    "Weight aggregation","Weight aggregation","Weight aggregation","Weight aggregation","Weight aggregation"
  ),
  stringsAsFactors = FALSE
)

# optional: ordered factors for plotting
metatable$SelectorType   <- factor(metatable$SelectorType, levels = c("Single","Ensemble"))
metatable$basis    <- factor(metatable$basis, levels = c("Baseline","Rank","Weight","Model"))
metatable$category <- factor(metatable$category, levels = c(
  "Baseline",
  "Univariate test",
  "Linear model (embedded)",
  "Tree importance",
  "Permutation importance",
  "Wrapper (RFE)",
  "Wrapper (RF/Boruta)",
  "Explainer",
  "Deep ranker",
  "Rank aggregation",
  "Weight aggregation"
))


# Apply mapping
df <- df %>%
  mutate(modelName = recode(modelName, !!!model_names_map)) %>%
  mutate(featureSelector = recode(featureSelector, !!!label_map)) 

# Ensure OmicsLevel follows the logical order
df$OmicsLevel <- factor(df$OmicsLevel, 
                        levels = c("SingleOmics", "DualOmics", "TripleOmics", "Unknown"))

gg_default_cols <- scales::hue_pal()(length(unique(df$featureSelector))) 
# Shuffle the whole named vector
gg_default_cols <- gg_default_cols[sample(seq_along(gg_default_cols))]

gg_default_cols <- c(
  '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
  '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
  '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
  '#000075', '#808080', '#ffffff', '#000000',
  '#4B0082', '#00CED1', '#FF1493', '#A1CAF1', '#B3446C', '#2B3D26', '#654522'
)

################################################################################
#### Feature Selection vs No Feature Selection##################################
################################################################################

# Add the new column
df$FeatureSelection <- ifelse(df$featureSelector != "None", "YES", "NO")

# Boxplot with a color-blind friendly palette (Dark2)
feature_selection <- ggboxplot(df, x = "Cohort", y = "MeanF1",
          color = "FeatureSelection",
          palette = c("#e6194b", "#00CED1")) + 
  labs(
    x = "",  
    y = "Mean F1 Score"           # optional: new y-axis label
  ) +
  theme_pubclean() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 
ggsave("BENCHMARKING/FIGURES/feature_selection_vs_no_feature_selection_boxplots.png", feature_selection, width = 8, height = 6, dpi = 300)
ggsave("BENCHMARKING/FIGURES/feature_selection_vs_no_feature_selection_boxplot.pdf", feature_selection, width = 8, height = 6, dpi = 300)
feature_selection

################################################################################
#############################Comparison Between Omics Levels####################
################################################################################
omics_level_comparison <- ggboxplot(df, x = "Cohort", y = "MeanF1",
               color = "OmicsLevel", palette = c("#e6194b", "#00CED1", "#f58231"))+
  labs(
    x = "",  
    y = "Mean F1 Score"           # optional: new y-axis label
  )+
  theme_pubclean() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 
ggsave("BENCHMARKING/FIGURES/omics_level_comparison_boxplots.png", omics_level_comparison, width = 8, height = 6, dpi = 300)
ggsave("BENCHMARKING/FIGURES/omics_level_comparison_boxplot.pdf", omics_level_comparison, width = 8, height = 6, dpi = 300)
omics_level_comparison

################################################################################
#############################Comparison Between Omics Types ####################
################################################################################
omics_type_comparison <- ggplot(df, aes(x = OmicsType, y = MeanF1, fill = OmicsType)) +
  geom_violin(trim = TRUE, alpha = 0.6) +
  geom_boxplot(width = 0.12, outlier.size = 1, coef = 1.5) +
  stat_summary(fun = median, geom = "point", size = 2, color = "black") + 
  scale_fill_manual(values = gg_default_cols) +
  labs(
       x = "",
       y = "Mean F1 Score") +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) 
ggsave("BENCHMARKING/FIGURES/omics_type_comparison_violinplots.png", omics_type_comparison, width = 8, height = 6, dpi = 300)
ggsave("BENCHMARKING/FIGURES/omics_type_comparison_violinplot.pdf", omics_type_comparison, width = 8, height = 6, dpi = 300)
omics_type_comparison


################################################################################
############################# Model Comparison Overall #########################
################################################################################
model_comparison_violin <- ggplot(df, aes(x = modelName, y = MeanF1, fill = modelName)) +
  geom_violin(trim = TRUE, alpha = 0.6) +
  geom_boxplot(width = 0.12, outlier.size = 1, coef = 1.5) +
  stat_summary(fun = median, geom = "point", size = 2, color = "black") + 
  scale_fill_manual(values = gg_default_cols) +
  labs(
    x = "",
    y = "Mean F1 Score") +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) 
ggsave("BENCHMARKING/FIGURES/model_comparison_violin_violinplots.png", model_comparison_violin, width = 8, height = 6, dpi = 300)
ggsave("BENCHMARKING/FIGURES/model_comparison_violin_violinplot.pdf", model_comparison_violin, width = 8, height = 6, dpi = 300)
model_comparison_violin

################################################################################
############################# Feature Selector Comparison Overall ##############
################################################################################
selector_comparison_violin <- ggplot(df, aes(x = featureSelector, y = MeanF1, fill = featureSelector)) +
  geom_violin(trim = TRUE, alpha = 0.6) +
  geom_boxplot(width = 0.12, outlier.size = 1, coef = 1.5) +
  stat_summary(fun = median, geom = "point", size = 2, color = "black") + 
  scale_fill_manual(values = gg_default_cols) +
  labs(
    x = "",
    y = "Mean F1 Score") +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) 
ggsave("BENCHMARKING/FIGURES/selector_comparison_violin_violinplots.png", selector_comparison_violin, width = 8, height = 6, dpi = 300)
ggsave("BENCHMARKING/FIGURES/selector_comparison_violin_violinplot.pdf", selector_comparison_violin, width = 8, height = 6, dpi = 300)
selector_comparison_violin

 
################################################################################
############################# Figure 1 Plot ####################################
################################################################################
Figure1 <- feature_selection + 
  omics_type_comparison + 
  omics_level_comparison + 
  model_comparison_violin +
  selector_comparison_violin + 
  plot_layout(
    design = "
ABB
CDD
EEE
", 
    widths = c(1,1,1), # relative widths of columns
    heights = c(2,1,1)   # relative heights of rows
  ) +
  plot_annotation(
    tag_levels = 'a', 
    tag_prefix = '(', tag_suffix = ')',
    theme = theme(plot.margin = unit(c(1.5,1.5,1.5,1.5),"lines"),
                  )  # margin around everything
  ) 
ggsave("BENCHMARKING/FIGURES/Figure1.png", Figure1, width = 12, height = 12, dpi = 300)
ggsave("BENCHMARKING/FIGURES/Figure1.pdf", Figure1, width = 12, height = 12, dpi = 300)










################################################################################
#############################  Performance Metric At Each Omics Type############
################################################################################
# Step 1: group by Cohort and OmicsType, and select the row where F1 is max
df_max <- df %>%
  group_by(Cohort, OmicsType) %>%
  slice_max(order_by = MeanF1, n = 1, with_ties = FALSE) %>%
  ungroup()

# Step 2: pivot longer & wider
df_long <- df_max %>%
  pivot_longer(
    cols = c(MeanF1, MeanAccuracy, MeanPrecision, MeanRecall, MeanAUC, 
             MeanSpecificity, MeanNPV, StdF1, StdAccuracy, StdPrecision, StdRecall, 
             StdAUC, StdSpecificity, StdNPV),
    names_to = c("Type", "Metric"),
    names_pattern = "(Mean|Std)(.*)",
    values_to = "Value"
  ) %>%
  pivot_wider(names_from = Type, values_from = Value)

performance_metrics_at_omics_types <- ggplot(df_long, aes(x = Metric, y = Mean, fill = OmicsType)) +
  geom_bar(stat = "identity",
           position = position_dodge(width = 0.8),
           width = 0.7) +
  geom_errorbar(aes(ymin = Mean - Std, ymax = Mean + Std),
                position = position_dodge(width = 0.8), width = 0.3) +
  facet_wrap(~ Cohort, ncol = 1) +
  theme_bw(base_size = 14) +
  theme(
    strip.text = element_text(size=14, face="bold"),
    axis.text.x = element_text(angle=45, hjust=1),
    legend.position = "top"
  ) +
  labs(y = "Value", x = "") +
  scale_fill_manual(values = gg_default_cols)  # <-- apply palette

ggsave("BENCHMARKING/FIGURES/performance_metrics_at_omics_types.pdf", performance_metrics_at_omics_types, width = 12, height = 12, dpi = 300)
ggsave("BENCHMARKING/FIGURES/performance_metrics_at_omics_types.png", performance_metrics_at_omics_types, width = 12, height = 12, dpi = 300)
performance_metrics_at_omics_types
 

################################################################################
#############################  Plot Model Performance by PanelSize #############
################################################################################
# 1. group by Cohort, numFeatures and modelName, and select the row where F1 is max
df_max <- df %>%
  group_by(Cohort, numFeatures, modelName, OmicsLevel) %>%
  slice_max(order_by = MeanF1, n = 1, with_ties = FALSE) %>%
  ungroup()  

df_summary <- df_max %>%
  group_by(Cohort, numFeatures, modelName) %>%
  summarise(
    across(
      c(MeanF1, MeanAccuracy, MeanPrecision, MeanRecall, MeanAUC, 
        MeanSpecificity, MeanNPV, StdF1, StdAccuracy, StdPrecision, StdRecall, 
        StdAUC, StdSpecificity, StdNPV),
      mean,
      na.rm = TRUE
    ),
    .groups = "drop"
  )


# 2. Pivot metrics to long format 
df_long <-df_summary %>%
  pivot_longer(
    cols = c(MeanF1, MeanAccuracy, MeanPrecision, MeanAUC, MeanRecall, MeanSpecificity, MeanNPV),  # add your metrics
    names_to = "Metric",
    values_to = "Value"
  )

# 3. Filter out numFeatures > 100
df_long <- df_long %>% filter(numFeatures <= 100)   

performace_by_panelsize <- ggplot(df_long, aes(x = numFeatures, y = Value,
                                               color = Metric, group = Metric)) +
  geom_line() +
  geom_point(size = 1) +
  facet_grid(rows = vars(Cohort), cols = vars(modelName)) +
  theme_bw(base_size = 12) +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  labs(x = "Number of Features", y = "Score") +
  scale_color_manual(values = gg_default_cols)   # 👈 Apply custom palette here

# Save figures
ggsave("BENCHMARKING/FIGURES/performace_by_panelsize.pdf", performace_by_panelsize,
       width = 12, height = 6, dpi = 300)
ggsave("BENCHMARKING/FIGURES/performace_by_panelsize.png", performace_by_panelsize,
       width = 12, height = 6, dpi = 300)
performace_by_panelsize


################################################################################
############################# Figure 2 Plot ####################################
################################################################################
Figure2 <- performance_metrics_at_omics_types + performace_by_panelsize + 
  plot_layout(
    design = "
A
B
", 
    widths = c(1), # relative widths of columns
    heights = c(3,1)   # relative heights of rows
  ) +
  plot_annotation(
    tag_levels = 'a', 
    tag_prefix = '(', tag_suffix = ')',
    theme = theme(plot.margin = unit(c(1.5,1.5,1.5,1.5),"lines"),
    )  # margin around everything
  ) 
ggsave("BENCHMARKING/FIGURES/Figure2.png", Figure2, width = 12, height = 16, dpi = 300)
ggsave("BENCHMARKING/FIGURES/Figure2.pdf", Figure2, width = 12, height = 16, dpi = 300)
 










################################################################################
#############################  Precision vs Recall Plots #######################
################################################################################
# Step 1: Filter to TripleOmics first
df_plot <- df %>%
  filter(OmicsLevel == "TripleOmics") %>%
  group_by(featureSelector, Cohort, modelName) %>%
  # pick row with max MeanF1
  slice_max(order_by = MeanF1, n = 1, with_ties = FALSE) %>%
  ungroup()
# Step 2: Plot Precision vs Recall
precision_vs_recall1 <- ggplot(df_plot,
       aes(x = MeanRecall,
           y = MeanPrecision,
           fill = Cohort)) +
  geom_point(
    shape = 21,
    size = 4,
    stroke = 1,
    color = "black"
  ) + 
  scale_fill_manual(values = c("#4363d8","#FE6D8C" ,"#3cb44b")) +
  facet_wrap(~ featureSelector, ncol = 6) +
  theme_bw(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    # place legend relative to panel region: (1.05, 0.5) = just outside right edge
    legend.position = c(0.8,0.03),
    legend.justification = c("left", "center")
  ) +
  labs(
    x = "Sensitivity (Mean Recall)",
    y = "Precision (Mean Precision)",
    fill = "Cohort"
  )
ggsave("BENCHMARKING/FIGURES/precision_vs_recall1.pdf", precision_vs_recall1, width = 12, height = 6, dpi = 300)
ggsave("BENCHMARKING/FIGURES/precision_vs_recall1.png", precision_vs_recall1, width = 12, height = 6, dpi = 300)
precision_vs_recall1


precision_vs_recall2 <- ggplot(df_plot,
   aes(x = MeanRecall,
       y = MeanPrecision,
       fill = Cohort)) +
  geom_point(
    shape = 21,
    size = 4,
    stroke = 1,
    color = "black"
  ) + 
  scale_fill_manual(values = c("#4363d8","#FE6D8C" ,"#3cb44b")) +
  facet_wrap(~ modelName, ncol = 6) +
  theme_bw(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    # place legend relative to panel region: (1.05, 0.5) = just outside right edge
    legend.position = c(0.85,0.15),
    legend.justification = c("left", "center")
  ) +
  labs(
    x = "Sensitivity (Mean Recall)",
    y = "Precision (Mean Precision)",
    fill = "Cohort"
  )
ggsave("BENCHMARKING/FIGURES/precision_vs_recall2.pdf", precision_vs_recall2, width = 12, height = 6, dpi = 300)
ggsave("BENCHMARKING/FIGURES/precision_vs_recall2.png", precision_vs_recall2, width = 12, height = 6, dpi = 300)
precision_vs_recall2

################################################################################
#############################  Model F1 Score Barplots #########################
################################################################################


# --- 1. Prepare df_plot ---
# Filter TripleOmics and summarise if needed:
df_plot_formodel <- df %>%
  filter(OmicsLevel == "TripleOmics") %>%
  filter(numFeatures <= 100) %>%
  # pick highest MeanF1 per group (or whatever you want)
  group_by(featureSelector, Cohort, modelName) %>%
  slice_max(order_by = MeanF1, n = 1, with_ties = FALSE) %>%
  ungroup()

# --- 2. Plot grouped bar chart with facets ---
modelmaxf1_barplot_by_model <- ggplot(df_plot_formodel,
       aes(x = Cohort,                # groups like TruSeq etc
           y = MeanF1,                # F-score
           fill = modelName)) +       # fill = Bowtie2 / BWA / NovoAlign
  geom_bar(stat = "identity",
           position = position_dodge(width = 0.8),
           width = 0.7) +  
  scale_fill_manual(values = gg_default_cols) +
  facet_wrap(~ featureSelector, ncol=6) +  # one panel per caller/tool
  theme_bw(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  labs(
    x = "",
    y = "F-score",
    fill = "Classifier",
    title = ""
  ) 
ggsave("BENCHMARKING/FIGURES/modelmaxf1_barplot_by_model.pdf", modelmaxf1_barplot_by_model, width = 12, height = 6, dpi = 300)
ggsave("BENCHMARKING/FIGURES/modelmaxf1_barplot_by_model.png", modelmaxf1_barplot_by_model, width = 12, height = 6, dpi = 300)
modelmaxf1_barplot_by_model

df_plot_forpanesize <- df %>%
  filter(OmicsLevel == "TripleOmics") %>%
  filter(numFeatures <= 100) %>%
  # pick highest MeanF1 per group (or whatever you want)
  group_by(featureSelector, Cohort, numFeatures) %>%
  slice_max(order_by = MeanF1, n = 1, with_ties = FALSE) %>%
  ungroup()

df_plot_forpanesize <- df_plot_forpanesize %>%
  mutate(numFeatures = factor(numFeatures))
modelmaxf1_barplot_by_panelsize<- ggplot(df_plot_forpanesize,
       aes(x = Cohort,                # groups like TruSeq etc
           y = MeanF1,                # F-score
           fill = numFeatures)) +       # fill = Bowtie2 / BWA / NovoAlign
  geom_bar(stat = "identity",
           position = position_dodge(width = 0.8),
           width = 0.7) + 
  scale_fill_manual(values = gg_default_cols) +
  facet_wrap(~ featureSelector, ncol=6) +  # one panel per caller/tool
  theme_bw(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  labs(
    x = "",
    y = "F-score",
    fill = "Panel Size",
    title = ""
  ) 
ggsave("BENCHMARKING/FIGURES/modelmaxf1_barplot_by_panelsize.pdf", modelmaxf1_barplot_by_panelsize, width = 12, height = 6, dpi = 300)
ggsave("BENCHMARKING/FIGURES/modelmaxf1_barplot_by_panelsize.png", modelmaxf1_barplot_by_panelsize, width = 12, height = 6, dpi = 300)
modelmaxf1_barplot_by_panelsize


################################################################################
############################# Figure 3 Plot ####################################
################################################################################
Figure3 <- precision_vs_recall1 + precision_vs_recall2 + modelmaxf1_barplot_by_model +
  plot_layout(
    design = "
A
B
C
", 
    widths = c(1), # relative widths of columns
    heights = c(2,1,2)   # relative heights of rows
  ) +
  plot_annotation(
    tag_levels = 'a', 
    tag_prefix = '(', tag_suffix = ')',
    theme = theme(plot.margin = unit(c(1.5,1.5,1.5,1.5),"lines"),
    )  # margin around everything
  ) 
ggsave("BENCHMARKING/FIGURES/Figure3.png", Figure3, width = 12, height = 18, dpi = 300)
ggsave("BENCHMARKING/FIGURES/Figure3.pdf", Figure3, width = 12, height = 18, dpi = 300)
 


Figure31 <- precision_vs_recall1 + precision_vs_recall2 +
  plot_layout(
    design = "
A
B
", 
    widths = c(1), # relative widths of columns
    heights = c(2,1)   # relative heights of rows
  ) +
  plot_annotation(
    tag_levels = 'a', 
    tag_prefix = '(', tag_suffix = ')',
    theme = theme(plot.margin = unit(c(1.5,1.5,1.5,1.5),"lines"),
    )  # margin around everything
  ) 
ggsave("BENCHMARKING/FIGURES/Figure3_1.png", Figure31, width = 12, height = 14, dpi = 300)
ggsave("BENCHMARKING/FIGURES/Figure3_1.pdf", Figure31, width = 12, height = 14, dpi = 300)


Figure32 <- modelmaxf1_barplot_by_model + modelmaxf1_barplot_by_panelsize +
  plot_layout(
    design = "
A
B
", 
    widths = c(1), # relative widths of columns
    heights = c(1,1)   # relative heights of rows
  ) +
  plot_annotation(
    tag_levels = 'a', 
    tag_prefix = '(', tag_suffix = ')',
    theme = theme(plot.margin = unit(c(1.5,1.5,1.5,1.5),"lines"),
    )  # margin around everything
  ) 
ggsave("BENCHMARKING/FIGURES/Figure3_2.png", Figure32, width = 12, height = 14, dpi = 300)
ggsave("BENCHMARKING/FIGURES/Figure3_2.pdf", Figure32, width = 12, height = 14, dpi = 300)



################################################################################
###################### Single vs Ensemble Selectors (Figure4) ##################
################################################################################
selected_scoring_feature <- "MeanF1"
selected_hue_feature <- "SelectorType"
selected_principal_feature <- "featureSelector"

data_subset <- df %>%
  select(-any_of("SelectorType")) %>%                     # drop old column if present
  left_join(metatable %>% distinct(featureSelector, SelectorType),
            by = "featureSelector")

best_rows <- data_subset %>%
  group_by(featureSelector, modelName, SelectorType, Cohort) %>%
  slice_max(order_by = !!sym(selected_scoring_feature), with_ties = FALSE) %>%
  ungroup() %>%
  mutate(Hue = paste(Cohort, SelectorType, sep = "_"))

best_rows <- best_rows %>%
  filter(numFeatures <= 100)
best_rows <- best_rows[order(best_rows$SelectorType, decreasing = TRUE), ]

Single_vs_ensemble_selectors <- ggplot(best_rows, aes_string(
  x = selected_principal_feature,
  y = selected_scoring_feature,
  color = "SelectorType"
)) +
  geom_boxplot(outlier.shape = NA, position = position_dodge(width = 0.8)) +
  geom_jitter(position = position_jitterdodge(jitter.width = 0.3, dodge.width = 0.8),
              alpha = 0.6, size = 1.5) +
  scale_fill_manual(values = gg_default_cols) +
  facet_wrap(~ Cohort) +
  theme_bw(base_size = 14) + 
  coord_flip() 
ggsave("BENCHMARKING/FIGURES/Single_vs_ensemble_selectors.png", Single_vs_ensemble_selectors, width = 12, height = 6, dpi = 300)
ggsave("BENCHMARKING/FIGURES/Single_vs_ensemble_selectors.pdf", Single_vs_ensemble_selectors, width = 12, height = 6, dpi = 300)
Single_vs_ensemble_selectors


Figure4 <- modelmaxf1_barplot_by_panelsize + Single_vs_ensemble_selectors + 
  plot_layout(
    design = "
A
B
", 
    widths = c(1), # relative widths of columns
    heights = c(1,2)   # relative heights of rows
  ) +
  plot_annotation(
    tag_levels = 'a', 
    tag_prefix = '(', tag_suffix = ')',
    theme = theme(plot.margin = unit(c(1.5,1.5,1.5,1.5),"lines"),
    )  # margin around everything
  ) 
ggsave("BENCHMARKING/FIGURES/Figure4.png", Figure4, width = 12, height = 16, dpi = 300)
ggsave("BENCHMARKING/FIGURES/Figure4.pdf", Figure4, width = 12, height = 16, dpi = 300)












################################################################################
######################### Model vs Selector Heatmap ############################
################################################################################ 

normalize_matrix <- function(x, margin = c("row", "col")) {
  margin <- match.arg(margin)
  
  if (margin == "row") {
    # divide each row by its max
    return(t(apply(x, 1, function(r) r / max(r, na.rm = TRUE))))
  } else {
    # divide each column by its max
    return(apply(x, 2, function(c) c / max(c, na.rm = TRUE)))
  }
}


# ---- 1) Filter and annotate ----
df2 <- df %>%
  filter(featureSelector != "None") %>%
  select(-any_of("SelectorType")) %>%     # ensure a clean overwrite
  left_join(
    metatable %>% distinct(featureSelector, SelectorType),
    by = "featureSelector"
  )

# ---- Parameters ----
selected_scoring_feature   <- "MeanF1"
selected_secondary_feature <- "modelName"
agg_fun_name               <- "max"
agg_fun <- match.fun(agg_fun_name)

# ---- 2) Aggregate metric keeping Cohort ----
agg_df <- df2 %>%
  group_by(Cohort, !!sym(selected_secondary_feature), featureSelector) %>%
  summarise(value = agg_fun(.data[[selected_scoring_feature]], na.rm = TRUE), .groups = "drop")

# ---- 3) If numFeatures, coerce numeric and create labels ----
if (selected_secondary_feature == "numFeatures") {
  agg_df <- agg_df %>%
    mutate(numFeatures_num = as.numeric(as.character(!!sym(selected_secondary_feature)))) %>%
    arrange(Cohort, numFeatures_num) %>%
    mutate(!!selected_secondary_feature := paste0("K: ", numFeatures_num)) %>%
    select(-numFeatures_num)
}

# ---- 4) Pivot: rows = Cohort+numFeatures, columns = featureSelector ----
agg_df <- agg_df %>%
  mutate(row_id = paste(Cohort, !!sym(selected_secondary_feature), sep = "_"))

heatwide <- agg_df %>%
  select(row_id, featureSelector, value) %>%
  pivot_wider(names_from = featureSelector, values_from = value)

rownames(heatwide) <- heatwide$row_id
mat <- as.matrix(heatwide[ , -1, drop = FALSE ])
mat <- apply(mat, 2, function(col) as.numeric(as.character(col)))
rownames(mat) <- heatwide$row_id

# ---- 5) Column annotation (SelectorType) ----
col_anno <- df2 %>%
  distinct(featureSelector, SelectorType) %>%
  filter(featureSelector %in% colnames(mat)) %>%
  arrange(match(featureSelector, colnames(mat)))

annotation_col <- as.data.frame(col_anno$SelectorType)
rownames(annotation_col) <- col_anno$featureSelector
colnames(annotation_col) <- "SelectorType"

# ---- 6) Row annotation (Cohort) ----
row_anno_df <- agg_df %>%
  distinct(row_id, Cohort) %>%
  filter(row_id %in% rownames(mat)) %>%
  arrange(match(row_id, rownames(mat)))

annotation_row <- as.data.frame(row_anno_df$Cohort)
rownames(annotation_row) <- row_anno_df$row_id
colnames(annotation_row) <- "Cohort"

# ---- 7) Display numbers ----
text_mat <- matrix("", nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
mask <- !is.na(mat)
text_mat[mask] <- sprintf("%.3f", mat[mask])

# ---- 8) Plot heatmap with both annotations ----
pheatmap1 <- pheatmap(
  mat,
  cluster_rows = FALSE,
  cluster_cols = TRUE,
  annotation_col = annotation_col,   # column annotation
  annotation_row = annotation_row,   # row annotation
  color = viridis::viridis(100),
  display_numbers = text_mat,
  number_color = "black",
  border_color = "grey30",
  filename = "BENCHMARKING/FIGURES/Heatmap-Selector-vs-Model.png",  # <--- save directly
  cellwidth = 35,
  cellheight = 25,
  fontsize = 14,
  fontsize_number = 12,
  fontsize_row = 14,          # row label font size
  fontsize_col = 14,          # column label font size 
  angle_col = 45,
  cutree_rows = 4,
  cutree_cols = 3,
  legend_breaks = c(0.65, 0.80, 1), # legend customisation
  legend_labels = c("Low", "Medium", "High"), # legend customisation#
  width=21
)

pheatmap(
  mat,
  cluster_rows = FALSE,
  cluster_cols = TRUE,
  annotation_col = annotation_col,   # column annotation
  annotation_row = annotation_row,   # row annotation
  color = viridis::viridis(100),
  display_numbers = text_mat,
  number_color = "black",
  border_color = "grey30",
  filename = "BENCHMARKING/FIGURES/Heatmap-Selector-vs-Model.pdf",  # <--- save directly
  cellwidth = 35,
  cellheight = 25,
  fontsize = 14,
  fontsize_number = 12,
  fontsize_row = 14,          # row label font size
  fontsize_col = 14,          # column label font size 
  angle_col = 45,
  cutree_rows = 4,
  cutree_cols = 3,
  legend_breaks = c(0.65, 0.80, 1), # legend customisation
  legend_labels = c("Low", "Medium", "High"), 
  width=21
)







################################################################################
##################### Panel Size vs Selector Heatmap ###########################
################################################################################ 

# ---- Parameters ----
selected_scoring_feature   <- "MeanF1"
selected_secondary_feature <- "numFeatures"
agg_fun_name               <- "max"
agg_fun <- match.fun(agg_fun_name)

# ---- 2) Aggregate metric keeping Cohort ----
agg_df <- df2 %>%
  group_by(Cohort, !!sym(selected_secondary_feature), featureSelector) %>%
  summarise(value = agg_fun(.data[[selected_scoring_feature]], na.rm = TRUE), .groups = "drop")

# ---- 3) If numFeatures, coerce numeric and create labels ----
if (selected_secondary_feature == "numFeatures") {
  agg_df <- agg_df %>%
    mutate(numFeatures_num = as.numeric(as.character(!!sym(selected_secondary_feature)))) %>%
    arrange(Cohort, numFeatures_num) %>%
    mutate(!!selected_secondary_feature := paste0("K: ", numFeatures_num)) %>%
    select(-numFeatures_num)
}

# ---- 4) Pivot: rows = Cohort+numFeatures, columns = featureSelector ----
agg_df <- agg_df %>%
  mutate(row_id = paste(Cohort, !!sym(selected_secondary_feature), sep = "_"))

heatwide <- agg_df %>%
  select(row_id, featureSelector, value) %>%
  pivot_wider(names_from = featureSelector, values_from = value)

rownames(heatwide) <- heatwide$row_id
mat <- as.matrix(heatwide[ , -1, drop = FALSE ])
mat <- apply(mat, 2, function(col) as.numeric(as.character(col)))
rownames(mat) <- heatwide$row_id

# ---- 5) Column annotation (SelectorType) ----
col_anno <- df2 %>%
  distinct(featureSelector, SelectorType) %>%
  filter(featureSelector %in% colnames(mat)) %>%
  arrange(match(featureSelector, colnames(mat)))

annotation_col <- as.data.frame(col_anno$SelectorType)
rownames(annotation_col) <- col_anno$featureSelector
colnames(annotation_col) <- "SelectorType"

# ---- 6) Row annotation (Cohort) ----
row_anno_df <- agg_df %>%
  distinct(row_id, Cohort) %>%
  filter(row_id %in% rownames(mat)) %>%
  arrange(match(row_id, rownames(mat)))

annotation_row <- as.data.frame(row_anno_df$Cohort)
rownames(annotation_row) <- row_anno_df$row_id
colnames(annotation_row) <- "Cohort"

# ---- 7) Display numbers ----
text_mat <- matrix("", nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
mask <- !is.na(mat)
text_mat[mask] <- sprintf("%.3f", mat[mask])

# ---- 8) Plot heatmap with both annotations ----
pheatmap(
  mat,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  annotation_col = annotation_col,   # column annotation
  annotation_row = annotation_row,   # row annotation
  color = viridis::viridis(100),
  display_numbers = text_mat,
  number_color = "black",
  border_color = "grey30",
  filename = "BENCHMARKING/FIGURES/Heatmap-Selector-vs-PanelSize.png",  # <--- save directly
  cellwidth = 35,
  cellheight = 25,
  fontsize = 14,
  fontsize_number = 12,
  fontsize_row = 14,          # row label font size
  fontsize_col = 14,          # column label font size 
  angle_col = 45,
  cutree_rows = 2,
  cutree_cols = 3,
  legend_breaks = c(0.65, 0.80, 1), # legend customisation
  legend_labels = c("Low", "Medium", "High"), # legend customisation#
  width=21
)

pheatmap(
  mat,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  annotation_col = annotation_col,   # column annotation
  annotation_row = annotation_row,   # row annotation
  color = viridis::viridis(100),
  display_numbers = text_mat,
  number_color = "black",
  border_color = "grey30",
  filename = "BENCHMARKING/FIGURES/Heatmap-Selector-vs-PanelSize.pdf",  # <--- save directly
  cellwidth = 35,
  cellheight = 25,
  fontsize = 14,
  fontsize_number = 12,
  fontsize_row = 14,          # row label font size
  fontsize_col = 14,          # column label font size 
  angle_col = 45,
  cutree_rows = 2,
  cutree_cols = 3,
  legend_breaks = c(0.65, 0.80, 1), # legend customisation
  legend_labels = c("Low", "Medium", "High"), 
  width=21
)






################################################################################
######################### Model vs Selector - OmicsLevel Heatmap ###############
################################################################################

agg_fun_name               <- "mean"
agg_fun <- match.fun(agg_fun_name)

df3 <- df2 %>%
  group_by(OmicsLevel, numFeatures, featureSelector,modelName) %>%
  summarise(MeanF1 = agg_fun(.data[[selected_scoring_feature]], na.rm = TRUE), .groups = "drop")

df3 <- df3 %>%
  mutate(
    SelectorType = case_when(
      grepl("rank|weight", featureSelector, ignore.case = TRUE) ~ "Ensemble",
      TRUE ~ "Single"
    )
  ) 

# Ensure OmicsLevel follows the logical order
df3$OmicsLevel <- factor(df3$OmicsLevel, 
                        levels = c("SingleOmics", "DualOmics", "TripleOmics", "Unknown"))

# ---- 2) Aggregate metric keeping Cohort ---- 
# ---- 8) Plot heatmap with both annotations ---- 
# ---- Parameters ----
selected_scoring_feature   <- "MeanF1"
selected_secondary_feature <- "modelName"
agg_fun_name               <- "max"
agg_fun <- match.fun(agg_fun_name)


agg_df <- df3 %>%
  group_by(OmicsLevel, !!sym(selected_secondary_feature), featureSelector) %>%
  summarise(value = agg_fun(.data[[selected_scoring_feature]], na.rm = TRUE), .groups = "drop")

# ---- 3) If numFeatures, coerce numeric and create labels ----
if (selected_secondary_feature == "numFeatures") {
  agg_df <- agg_df %>%
    mutate(numFeatures_num = as.numeric(as.character(!!sym(selected_secondary_feature)))) %>%
    arrange(OmicsLevel, numFeatures_num) %>%
    mutate(!!selected_secondary_feature := paste0("K: ", numFeatures_num)) %>%
    select(-numFeatures_num)
}

# ---- 4) Pivot: rows = Cohort+numFeatures, columns = featureSelector ----
agg_df <- agg_df %>%
  mutate(row_id = paste(OmicsLevel, !!sym(selected_secondary_feature), sep = "_"))

heatwide <- agg_df %>%
  select(row_id, featureSelector, value) %>%
  pivot_wider(names_from = featureSelector, values_from = value)

rownames(heatwide) <- heatwide$row_id
mat <- as.matrix(heatwide[ , -1, drop = FALSE ])
mat <- apply(mat, 2, function(col) as.numeric(as.character(col)))
rownames(mat) <- heatwide$row_id

# ---- 5) Column annotation (SelectorType) ----
col_anno <- df3 %>%
  distinct(featureSelector, SelectorType) %>%
  filter(featureSelector %in% colnames(mat)) %>%
  arrange(match(featureSelector, colnames(mat)))

annotation_col <- as.data.frame(col_anno$SelectorType)
rownames(annotation_col) <- col_anno$featureSelector
colnames(annotation_col) <- "SelectorType"

# ---- 6) Row annotation (Cohort) ----
row_anno_df <- agg_df %>%
  distinct(row_id, OmicsLevel) %>%
  filter(row_id %in% rownames(mat)) %>%
  arrange(match(row_id, rownames(mat)))

annotation_row <- as.data.frame(row_anno_df$OmicsLevel)
rownames(annotation_row) <- row_anno_df$row_id
colnames(annotation_row) <- "OmicsLevel"

# ---- 7) Display numbers ----
text_mat <- matrix("", nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
mask <- !is.na(mat)
text_mat[mask] <- sprintf("%.3f", mat[mask])


# choose one of these:
mat_norm <- normalize_matrix(mat, margin = "col")    # per row normalization
# mat_norm <- normalize_cols(mat)   # per column normalization
# rebuild display numbers from mat_norm
text_mat_norm <- matrix("", nrow = nrow(mat_norm), ncol = ncol(mat_norm),
                        dimnames = dimnames(mat_norm))
mask <- !is.na(mat_norm)
text_mat_norm[mask] <- sprintf("%.3f", mat_norm[mask])

pheatmap2 <- pheatmap(
  mat,
  scale = "none",
  cluster_rows = F,
  cluster_cols = TRUE,
  annotation_col = annotation_col,   # column annotation
  annotation_row = annotation_row,   # row annotation
  color = viridis::viridis(100),
  display_numbers = text_mat,
  number_color = "black",
  border_color = "grey30",
  filename = "BENCHMARKING/FIGURES/Heatmap-Selector-vs-Model-vs-OmicsLevel.pdf",  # <--- save directly
  fontsize_number = 8,
  angle_col = 45,
  cutree_rows = 3,
  cutree_cols = 1,
  width = 14,   # make width bigger
  height = 8,   # keep height smaller
  units = "in", dpi = 300
)
pheatmap(
  mat,
  scale = "none",
  cluster_rows = F,
  cluster_cols = TRUE,
  annotation_col = annotation_col,   # column annotation
  annotation_row = annotation_row,   # row annotation
  color = viridis::viridis(100),
  display_numbers = text_mat,
  number_color = "black",
  border_color = "grey30",
  filename = "BENCHMARKING/FIGURES/Heatmap-Selector-vs-Model-vs-OmicsLevel.png",  # <--- save directly
  fontsize_number = 8,
  angle_col = 45,
  cutree_rows = 3,
  cutree_cols = 1,
  width = 14,   # make width bigger
  height = 8,   # keep height smaller
  units = "in", dpi = 300
)



################################################################################
######################### PanelSize vs Selector - OmicsLevel Heatmap ###############
################################################################################


# ---- 2) Aggregate metric keeping Cohort ---- 
# ---- 8) Plot heatmap with both annotations ---- 
# ---- Parameters ----
selected_scoring_feature   <- "MeanF1"
selected_secondary_feature <- "numFeatures"
agg_fun_name               <- "max"
agg_fun <- match.fun(agg_fun_name)


agg_df <- df3 %>%
  group_by(OmicsLevel, !!sym(selected_secondary_feature), featureSelector) %>%
  summarise(value = agg_fun(.data[[selected_scoring_feature]], na.rm = TRUE), .groups = "drop")

# ---- 3) If numFeatures, coerce numeric and create labels ----
if (selected_secondary_feature == "numFeatures") {
  agg_df <- agg_df %>%
    mutate(numFeatures_num = as.numeric(as.character(!!sym(selected_secondary_feature)))) %>%
    arrange(OmicsLevel, numFeatures_num) %>%
    mutate(!!selected_secondary_feature := paste0("K: ", numFeatures_num)) %>%
    select(-numFeatures_num)
}

# ---- 4) Pivot: rows = Cohort+numFeatures, columns = featureSelector ----
agg_df <- agg_df %>%
  mutate(row_id = paste(OmicsLevel, !!sym(selected_secondary_feature), sep = "_"))

heatwide <- agg_df %>%
  select(row_id, featureSelector, value) %>%
  pivot_wider(names_from = featureSelector, values_from = value)

rownames(heatwide) <- heatwide$row_id
mat <- as.matrix(heatwide[ , -1, drop = FALSE ])
mat <- apply(mat, 2, function(col) as.numeric(as.character(col)))
rownames(mat) <- heatwide$row_id

# ---- 5) Column annotation (SelectorType) ----
col_anno <- df3 %>%
  distinct(featureSelector, SelectorType) %>%
  filter(featureSelector %in% colnames(mat)) %>%
  arrange(match(featureSelector, colnames(mat)))

annotation_col <- as.data.frame(col_anno$SelectorType)
rownames(annotation_col) <- col_anno$featureSelector
colnames(annotation_col) <- "SelectorType"

# ---- 6) Row annotation (Cohort) ----
row_anno_df <- agg_df %>%
  distinct(row_id, OmicsLevel) %>%
  filter(row_id %in% rownames(mat)) %>%
  arrange(match(row_id, rownames(mat)))

annotation_row <- as.data.frame(row_anno_df$OmicsLevel)
rownames(annotation_row) <- row_anno_df$row_id
colnames(annotation_row) <- "OmicsLevel"

# ---- 7) Display numbers ----
text_mat <- matrix("", nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
mask <- !is.na(mat)
text_mat[mask] <- sprintf("%.3f", mat[mask])


# choose one of these:
mat_norm <- normalize_matrix(mat, margin = "col")    # per row normalization
# mat_norm <- normalize_cols(mat)   # per column normalization
# rebuild display numbers from mat_norm
text_mat_norm <- matrix("", nrow = nrow(mat_norm), ncol = ncol(mat_norm),
                        dimnames = dimnames(mat_norm))
mask <- !is.na(mat_norm)
text_mat_norm[mask] <- sprintf("%.3f", mat_norm[mask])

pheatmap(
  mat,
  scale = "none",
  cluster_rows = F,
  cluster_cols = TRUE,
  annotation_col = annotation_col,
  annotation_row = annotation_row,
  color = viridis::viridis(100),
  display_numbers = text_mat,
  number_color = "black",
  border_color = "grey30",
  filename = "BENCHMARKING/FIGURES/Heatmap-Selector-vs-PanelSize-vs-OmicsLevel.png",
  fontsize_number = 8,
  angle_col = 45,
  cutree_rows = 3,
  cutree_cols = 2,
  width = 14,   # make width bigger
  height = 8,   # keep height smaller
  units = "in", dpi = 300
)

 
pheatmap(
  mat,
  scale = "none",
  cluster_rows = F,
  cluster_cols = TRUE,
  annotation_col = annotation_col,
  annotation_row = annotation_row,
  color = viridis::viridis(100),
  display_numbers = text_mat,
  number_color = "black",
  border_color = "grey30",
  filename = "BENCHMARKING/FIGURES/Heatmap-Selector-vs-PanelSize-vs-OmicsLevel.pdf",
  fontsize_number = 8,
  angle_col = 45,
  cutree_rows = 3,
  cutree_cols = 2,
  width = 14,   # make width bigger
  height = 8,   # keep height smaller
  units = "in", dpi = 300
)

 







# ---- 2) Aggregate metric keeping Cohort ---- 
# ---- 8) Plot heatmap with both annotations ---- 
# ---- Parameters ----
selected_scoring_feature   <- "MeanF1"
selected_secondary_feature <- "numFeatures"
agg_fun_name               <- "max"
agg_fun <- match.fun(agg_fun_name)


agg_df <- df3 %>%
  group_by(OmicsLevel, !!sym(selected_secondary_feature), modelName) %>%
  summarise(value = agg_fun(.data[[selected_scoring_feature]], na.rm = TRUE), .groups = "drop")

# ---- 3) If numFeatures, coerce numeric and create labels ----
if (selected_secondary_feature == "numFeatures") {
  agg_df <- agg_df %>%
    mutate(numFeatures_num = as.numeric(as.character(!!sym(selected_secondary_feature)))) %>%
    arrange(OmicsLevel, numFeatures_num) %>%
    mutate(!!selected_secondary_feature := paste0("K: ", numFeatures_num)) %>%
    select(-numFeatures_num)
}

# ---- 4) Pivot: rows = Cohort+numFeatures, columns = featureSelector ----
agg_df <- agg_df %>%
  mutate(row_id = paste(OmicsLevel, !!sym(selected_secondary_feature), sep = "_"))

heatwide <- agg_df %>%
  select(row_id, modelName, value) %>%
  pivot_wider(names_from = modelName, values_from = value)

rownames(heatwide) <- heatwide$row_id
mat <- as.matrix(heatwide[ , -1, drop = FALSE ])
mat <- apply(mat, 2, function(col) as.numeric(as.character(col)))
rownames(mat) <- heatwide$row_id

# ---- 6) Row annotation (Cohort) ----
row_anno_df <- agg_df %>%
  distinct(row_id, OmicsLevel) %>%
  filter(row_id %in% rownames(mat)) %>%
  arrange(match(row_id, rownames(mat)))

annotation_row <- as.data.frame(row_anno_df$OmicsLevel)
rownames(annotation_row) <- row_anno_df$row_id
colnames(annotation_row) <- "OmicsLevel"

# ---- 7) Display numbers ----
text_mat <- matrix("", nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
mask <- !is.na(mat)
text_mat[mask] <- sprintf("%.3f", mat[mask])

pheatmap(
  t(mat),
  scale = "none",
  cluster_rows = T,
  cluster_cols = F,
  annotation_col = annotation_row,
  color = viridis::viridis(100),
  display_numbers = t(text_mat),
  number_color = "black",
  border_color = "grey30",
  filename = "BENCHMARKING/FIGURES/Heatmap-Model-vs-PanelSize-vs-OmicsLevel.png",
  fontsize_number = 8,
  angle_col = 45,
  cutree_rows = 2,
  cutree_cols = 3,
  width = 14,   # make width bigger
  height = 5,   # keep height smaller
  units = "in", dpi = 300
)


pheatmap(
  t(mat),
  scale = "none",
  cluster_rows = T,
  cluster_cols = F,
  annotation_col = annotation_row,
  color = viridis::viridis(100),
  display_numbers = t(text_mat),
  number_color = "black",
  border_color = "grey30",
  filename = "BENCHMARKING/FIGURES/Heatmap-Model-vs-PanelSize-vs-OmicsLevel.pdf",
  fontsize_number = 8,
  angle_col = 45,
  cutree_rows = 2,
  cutree_cols = 3,
  width = 14,   # make width bigger
  height = 5,   # keep height smaller
  units = "in", dpi = 300
)




################################################################################
############## Validation Results TPs by Database and OmicsLevel ###############
################################################################################

library(ggplot2)
library(dplyr)
library(rlang)

# ---------------- params ----------------
selected_values <- c(10, 30, 50, 100)   # method_cutoff values to plot
selected_top_N_val <- 1000              # groundtruth_cutoff
selected_validation_scorer <- "TP"      # column to plot on y-axis

# -------------- load data --------------
df_plot_all <- read.csv("BENCHMARKING/Biomarker_Validation_Results.csv")
df_plot_all$Cohort[df_plot_all$Cohort == "MayoRNASeq"] <- "PSP"

# Apply mapping
df_plot_all <- df_plot_all %>%
  mutate(featureSelector = recode(featureSelector, !!!label_map)) 


# Explicit column names (no guessing)
nb_col   <- "method_cutoff"
topN_col <- "groundtruth_cutoff"

# Validate columns exist
missing_cols <- setdiff(
  c(nb_col, topN_col, "featureSelector", "validationsource", "OmicsLevel", selected_validation_scorer),
  names(df_plot_all)
)
if (length(missing_cols)) {
  stop("Missing required columns in CSV: ", paste(missing_cols, collapse = ", "))
}

# Coerce cutoffs to numeric if needed
df_plot_all[[nb_col]]   <- suppressWarnings(as.numeric(df_plot_all[[nb_col]]))
df_plot_all[[topN_col]] <- suppressWarnings(as.numeric(df_plot_all[[topN_col]]))

# -------------- constant factor ordering --------------
methods_all <- unique(df_plot_all$featureSelector)
order_methods <- tryCatch(methods_all[order(as.numeric(methods_all))], error = function(e) sort(methods_all))
sources_all <- unique(df_plot_all$validationsource)

omics_levels <- c("SingleOmics","DualOmics","TripleOmics","Unknown")
omics_labels <- c("Single-Omics","Dual-Omics","Triple-Omics","Unknown")

# -------------- plot function --------------
make_val_plot <- function(df_plot, nb_value) {
  df_plot$featureSelector  <- factor(df_plot$featureSelector,  levels = order_methods)
  df_plot$validationsource <- factor(df_plot$validationsource, levels = sources_all)
  df_plot$OmicsLevel       <- factor(df_plot$OmicsLevel, levels = omics_levels, labels = omics_labels)
  
  ggplot(
    df_plot,
    aes(x = featureSelector, y = !!sym(selected_validation_scorer), fill = Cohort)
  ) +
    geom_bar(stat = "summary", fun = "mean", position = position_dodge()) +
    scale_fill_brewer(palette = "Set2") +  # change to scale_fill_manual(values = gg_default_cols) if using your palette
    theme_bw(base_size = 12) +
    theme(
      axis.text.y = element_text(hjust = 1),
      legend.position = "right",
      legend.title = element_text(face = "bold"),
      strip.text.y = element_text(size = 10, face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold")   # ✅ Center main title
    ) +
    labs(
      title = paste0(
        "Validation ", selected_validation_scorer,
        " | ", nb_col, " = ", nb_value,
        " | ", topN_col, " = ", selected_top_N_val
      ),
      x = "",
      y = selected_validation_scorer,
      fill = "Cohort"
    ) +
    facet_grid(
      cols = vars(validationsource),
      rows = vars(OmicsLevel),
      scales = "free_x"
    ) +
    coord_flip()
}

# -------------- loop, filter, save --------------
out_dir <- "BENCHMARKING/FIGURES"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

height <- 18
width  <- 15

for (nb in selected_values) {
  df_plot <- df_plot_all %>%
    filter(.data[[nb_col]] == nb, .data[[topN_col]] == selected_top_N_val)
  
  if (nrow(df_plot) == 0) {
    message("No rows for ", nb_col, " = ", nb, " and ", topN_col, " = ", selected_top_N_val)
    next
  }
  
  p <- make_val_plot(df_plot, nb)
  
  tag <- paste0("method", nb, "_gt", selected_top_N_val)
  ggsave(file.path(out_dir, paste0("validation_barlots_", tag, ".png")), p,
         width = width, height = height, dpi = 300)
  ggsave(file.path(out_dir, paste0("validation_barlots_", tag, ".pdf")), p,
         width = width, height = height, dpi = 300)
  
  print(p)
}






################################################################################
############## Upset Plots Showing Intersection of Biomarker Sets ##############
################################################################################
# read your data
df <- read.csv("BENCHMARKING/Selected_Biomarker_Panels.csv", stringsAsFactors = FALSE)
df$Cohort[df$Cohort == "MayoRNASeq"] <- "PSP"


df <- df %>%
  rename(
    `RF-FI`  = randomforest_feature_importance,
    `XGB-FI` = xgb_feature_importance,
    `RF-PFI` = rf_permutation_feature_importance,
    `XGB-PFI`= xgb_permutation_feature_importance,
    `MOGONET:Ranker` = MOGONET.Ranker, 
    `MORE:Ranker` = MORE.Ranker, 
  )

df <- df %>%
  rename_with(function(nm) {
    idx <- match(nm, names(label_map))
    out <- nm
    out[!is.na(idx)] <- unname(label_map[idx[!is.na(idx)]])
    make.unique(out)
  })



# Filter to TripleOmics
df_triple <- df %>% filter(OmicsLevel == "TripleOmics")

ranker_cols <- setdiff(
  names(df_triple),
  c("Feature", "Cohort", "OmicsLevel", "OmicsTypes")
)

cohorts <- unique(df_triple$Cohort)

# Create an empty list to hold the plots
upset_plots <- list()

for (cohort_name in cohorts) {
  message("Processing cohort: ", cohort_name)
  
  df_sub <- df_triple %>% filter(Cohort == cohort_name)
  
  top_features_long <- lapply(ranker_cols, function(col) {
    df_sub %>%
      arrange(.data[[col]]) %>%
      slice_head(n = 30) %>%
      select(Feature) %>%
      mutate(Ranker = col, Present = 1)
  }) %>% bind_rows()
  
  upset_matrix <- top_features_long %>%
    pivot_wider(
      id_cols = Feature,
      names_from = Ranker,
      values_from = Present,
      values_fill = 0
    )
  
  upset_binary_df <- upset_matrix %>%
    select(-Feature) %>%
    as.data.frame()
  
  if (cohort_name == "ROSMAP") {
    cohort_fill <- "#3cb44b"
  } else if (cohort_name == "BRCA") {
    cohort_fill <- "#4363d8"
  } else if (cohort_name == "PSP") {
    cohort_fill <- "#FE6D8C"
  } else {
    cohort_fill <- "grey70"  # fallback colour
  }

  p <- upset(
    upset_binary_df,
    intersect = colnames(upset_binary_df),
    base_annotations = list('Intersection size' = intersection_size(fill=cohort_fill)),
    set_sizes = FALSE,
    n_intersections = 15
  )
  
  # Store the plot in a named list
  upset_plots[[cohort_name]] <- p
  
  # (Optional) save files right here
  ggsave(paste0("BENCHMARKING/FIGURES/my_upset_plot_", cohort_name, ".png"), plot = p, width = 5, height = 6, dpi = 300)
  ggsave(paste0("BENCHMARKING/FIGURES/my_upset_plot_", cohort_name, ".pdf"), plot = p, width = 5, height = 6)
}

# Now you can access them like:
upset_plots$ROSMAP
upset_plots$BRCA
upset_plots$PSP


################################################################################
############## PCA Plots Showing Similarities Between Rankers ##################
################################################################################
# Cohorts (adjust names if your filenames differ)
cohorts <- c("ROSMAP", "BRCA", "PSP")

# Folder where Python saved CSVs (adjust if needed)
outdir <- "../BENCHMARKING/FIGURES"

# Containers to hold results
pca_plots <- list()
pca_objects <- list()
binary_matrices <- list()

# Helper to safely read CSV (preserve column names)
safe_read_csv <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

cohorts <- c("ROSMAP","BRCA","PSP")

pca_objects <- list()

for (cohort in cohorts) {
  # 1. Load CSV produced by Python
  df <- read.csv(paste0("BENCHMARKING/", cohort, "_PCA_data.csv"),
                 check.names = FALSE, stringsAsFactors = FALSE)
  
  df <- df %>%
    rename_with(function(nm) {
      idx <- match(nm, names(label_map))
      out <- nm
      out[!is.na(idx)] <- unname(label_map[idx[!is.na(idx)]])
      make.unique(out)
    })
  
  # 2. Keep only numeric columns
  df_num <- df[sapply(df, is.numeric)]
  
  # 3. Transpose so rows = algorithms (observations), cols = features (variables)
  mat <- t(as.matrix(df_num))
  
  # 3b. Drop rows (algorithms) that are entirely 0
  mat <- mat[rowSums(mat != 0) > 0, , drop = FALSE]
  
  # 3c. Drop columns (features) that are entirely 0
  mat <- mat[, colSums(mat != 0) > 0, drop = FALSE]
  
  # optional: give rownames automatically if missing
  if (is.null(rownames(mat)) || all(rownames(mat) == "")) {
    rownames(mat) <- paste0("alg_", seq_len(nrow(mat)))
  }
  
  # 4. Run PCA on the binary matrix
  pca <- prcomp(mat, center = TRUE, scale. = TRUE)
  
  # 5. Extract PC1 & PC2 scores
  scores <- as.data.frame(pca$x[, 1:2])
  scores$Algorithm <- rownames(scores)
  
  # 5b. Tag each algorithm as "Ensemble" if name has "rank" or "weight"
  scores$Type <- ifelse(
    grepl("rank|weight", scores$Algorithm, ignore.case = TRUE),
    "Ensemble",
    "Single"
  )
  
  # percent variance
  var_pct <- (pca$sdev^2)/sum(pca$sdev^2)
  pct1 <- round(100 * var_pct[1], 1)
  pct2 <- round(100 * var_pct[2], 1)
  
  # 6. Plot with color by Type
  p <- ggplot(scores, aes(PC1, PC2, label = Algorithm, color = Type)) +
    geom_point(size = 3) +
    ggrepel::geom_text_repel(size = 3) +
    theme_bw() +
    labs(title = "",
         x = paste0("PC1 (", pct1, "%)"),
         y = paste0("PC2 (", pct2, "%)"),
         color = "Ranker Type") +
    scale_color_manual(values = c("Single" = "#4363d8",
                                  "Ensemble" = '#f58231'))

  pca_plots[[cohort]] <- p
  pca_objects[[cohort]] <- pca
  
  # 7. Save
  ggsave(paste0("BENCHMARKING/FIGURES/PCA_", cohort, ".png"),
         p, width = 5, height = 4, dpi = 300)
  ggsave(paste0("BENCHMARKING/FIGURES/PCA_", cohort, ".pdf"),
         p, width = 5, height = 4, dpi = 300)
}


# After the loop you have:
# - pca_plots: named list of ggplot objects (one per cohort)
# - pca_objects: prcomp objects for each cohort
# - binary_matrices: the numeric matrices used for PCA (rows = algorithms, cols = features)
#
# Example: display a plot
if ("ROSMAP" %in% names(pca_plots)) print(pca_plots[["ROSMAP"]])
 


################################################################################
############################# Figure 5 Plot ####################################
################################################################################
library(patchwork)
library(grid)    # needed for grobs

# Wrap all upset plots
wrapped_upset_plots <- lapply(upset_plots, function(uplot) {
  wrap_elements(uplot)  # this turns the ComplexUpset output into a single patchwork element
})

# Now combine using patchwork
Figure5 <- (
  wrapped_upset_plots$ROSMAP + 
    wrapped_upset_plots$BRCA  + 
    wrapped_upset_plots$PSP 
) +
  plot_layout(
    design = "
AAA
BBB
CCC
",
    widths = c(1, 1, 1),
    heights = c(2, 2.7, 2.2)
  ) +
  plot_annotation(
    tag_levels = 'a', 
    tag_prefix = '(', tag_suffix = ')',
    theme = theme(
      plot.margin = unit(c(1.5, 1.5, 1.5, 1.5), "lines")
    )
  )

ggsave("BENCHMARKING/FIGURES/Figure5.png", Figure5, width = 14, height = 20, dpi = 300)
ggsave("BENCHMARKING/FIGURES/Figure5.pdf", Figure5, width = 14, height = 20, dpi = 300)



cohorts <- c("ROSMAP","BRCA","PSP")

heatmaps <- list()  # store each pheatmap object

# Helper: build named colors for whatever levels are present,
# preferring your distinct palette for any missing levels
mk_anno_colors <- function(levels, defaults, palette) {
  # Keep defaults that are present
  got <- defaults[names(defaults) %in% levels]
  
  # What levels still need colors?
  missing <- setdiff(levels, names(got))
  if (!length(missing)) return(got)
  
  # Don't reuse colors already taken by defaults
  remaining <- setdiff(palette, unname(got))
  
  # If palette runs short, top up with well-separated HCL colors
  if (length(remaining) < length(missing)) {
    extra_n <- length(missing) - length(remaining)
    remaining <- c(remaining, qualitative_hcl(extra_n, palette = "Dark 3"))
  }
  
  extra <- remaining[seq_len(length(missing))]
  names(extra) <- missing
  c(got, extra)
}

make_heatmap <- function(csv_path, cohort, out_dir = ".") {
  # 1) Load
  df <- read.csv(csv_path, stringsAsFactors = FALSE, check.names = FALSE)
  
  df <- df %>%
    rename_with(function(nm) {
      idx <- match(nm, names(label_map))
      out <- nm
      out[!is.na(idx)] <- unname(label_map[idx[!is.na(idx)]])
      make.unique(out)
    })
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
  
  # 4) Annotation color defaults
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
    FeatureType = mk_anno_colors(unique(annotation_row$FeatureType),
                                 feature_type_defaults, gg_default_cols),
    OmicsLevels = mk_anno_colors(unique(annotation_row$OmicsLevels),
                                 omics_level_defaults, gg_default_cols)
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
    # 🌤️ Light heatmap colors: pale yellow → light blue
    color = colorRampPalette(c("#ffffe0", "#b3cde3", "#6497b1", "#005b96"))(100),
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
    color = colorRampPalette(c("#ffffe0", "#b3cde3", "#6497b1", "#005b96"))(100),
    filename = pdf_path,
    width = 12, height = 8
  )
  
  
  message("Saved: ", png_path, " and ", pdf_path)
}

# --- Run for each cohort (MayoRNASeq -> PSP) ---
make_heatmap("results/plot_data_ROSMAP.csv", "ROSMAP", out_dir = "BENCHMARKING/FIGURES")
make_heatmap("results/plot_data_PSP.csv",    "PSP",    out_dir = "BENCHMARKING/FIGURES")     # <- changed here
make_heatmap("results/plot_data_BRCA.csv",   "BRCA",   out_dir = "BENCHMARKING/FIGURES")





















library(ggplot2)
library(dplyr)
library(rlang)

# ---------------- hard-coded params ----------------
nb_fixed   <- 30     # method_cutoff
topN_fixed <- 1000   # groundtruth_cutoff
y_metric   <- "TP"   # column to plot on y-axis

# -------------- load & prepare data --------------
df_plot_all <- read.csv("BENCHMARKING/Biomarker_Validation_Results.csv")

# Rename cohort: MayoRNASeq -> PSP
df_plot_all$Cohort[df_plot_all$Cohort == "MayoRNASeq"] <- "PSP"

# Apply mapping
df_plot_all <- df_plot_all %>%
  mutate(featureSelector = recode(featureSelector, !!!label_map)) 

# Explicit column names
nb_col   <- "method_cutoff"
topN_col <- "groundtruth_cutoff"

# Validate columns exist
missing_cols <- setdiff(
  c(nb_col, topN_col, "featureSelector", "validationsource", "OmicsLevel", y_metric),
  names(df_plot_all)
)
if (length(missing_cols)) {
  stop("Missing required columns in CSV: ", paste(missing_cols, collapse = ", "))
}

# Coerce cutoffs to numeric if needed
df_plot_all[[nb_col]]   <- suppressWarnings(as.numeric(df_plot_all[[nb_col]]))
df_plot_all[[topN_col]] <- suppressWarnings(as.numeric(df_plot_all[[topN_col]]))

# Filter to the fixed cutoffs
df_plot <- df_plot_all %>%
  filter(.data[[nb_col]] == nb_fixed,
         .data[[topN_col]] == topN_fixed)

if (nrow(df_plot) == 0) {
  stop("No rows for ", nb_col, " = ", nb_fixed, " and ", topN_col, " = ", topN_fixed)
}

# ----------- factor ordering (consistent, readable) -----------
methods_all  <- unique(df_plot$featureSelector)
order_methods <- tryCatch(methods_all[order(as.numeric(methods_all))],
                          error = function(e) sort(methods_all))
df_plot$featureSelector  <- factor(df_plot$featureSelector,  levels = order_methods)

sources_all <- unique(df_plot$validationsource)
df_plot$validationsource <- factor(df_plot$validationsource, levels = sources_all)

df_plot$OmicsLevel <- factor(df_plot$OmicsLevel,
                             levels = c("SingleOmics","DualOmics","TripleOmics","Unknown"),
                             labels = c("Single-Omics","Dual-Omics","Triple-Omics","Unknown"))

# ---------------- build plot ----------------
validation_barlots <- ggplot(
  df_plot,
  aes(x = featureSelector, y = !!sym(y_metric), fill = Cohort)
) +
  geom_bar(stat = "summary", fun = "mean", position = position_dodge()) +
  scale_fill_brewer(palette = "Set2") +
  theme_bw(base_size = 12) +
  theme(
    axis.text.y  = element_text(angle = -45, hjust = 1, vjust = 1),  # ✅ rotated y-ticks
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    strip.text.y = element_text(size = 10, face = "bold"),
    strip.text.x = element_text(size = 10, face = "bold"),
    plot.title   = element_text(hjust = 0.5, face = "bold")         # centered title
  ) +
  labs(
    title = paste0("Validation ", y_metric,
                   " | ", nb_col, " = ", nb_fixed,
                   " | ", topN_col, " = ", topN_fixed),
    x = "",
    y = y_metric,
    fill = "Cohort"
  ) +
  facet_grid(
    cols = vars(validationsource),
    rows = vars(OmicsLevel),
    scales = "free_x"
  ) +
  coord_flip()
ggsave(file.path(out_dir, "validation_barlots_method30_gt1000h.png"), validation_barlots, width = 12, height = 18, dpi = 300) 
ggsave(file.path(out_dir, "validation_barlots_method30_gt1000h.pdf"), validation_barlots, width = 12, height = 18, dpi = 300)

