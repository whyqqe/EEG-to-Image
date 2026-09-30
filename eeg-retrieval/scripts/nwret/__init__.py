"""NW-Retrieval: zero-shot EEG-to-image retrieval.

Pipeline
--------
  EEGTokenizer   montage interpolation + time windows -> tokens on a 2D grid
  ViTEEGEncoder  pretrained ViT (CLIP visual tower / DINOv2-L / ViT-B16-21k)
                 with its patch_embed interface replaced by the EEG tokenizer
  LayerFusion    subject-aware multi-granularity routing over intermediate layers
  RetrievalModel alignment to frozen image features in CLIP joint space

See docs/DESIGN_semantic_structure_branches.md for the reasoning.
"""
