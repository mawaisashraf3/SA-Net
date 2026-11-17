# Implemented methodology and mathematical conventions

Read the [critical manuscript audit](research-audit.md) for equation/figure-level decisions. The code is an independent interpretation with explicitly documented alternatives.

## Inputs and coordinate system

An RGB tensor has shape B×V×3×H×W, ImageNet mean/std normalization. ResNet-34 produces hierarchical features; lateral 1×1 projections and top-down bilinear FPN aggregation return stride-4 features. The compact alternative uses three convolutional stages and exists for CPU verification, with no pretrained weights. `pretrained: true` explicitly requests torchvision's ImageNet weights and a network download; it is false by default.

Inputs are assumed to be rectified; lens distortion and sensor timestamp alignment must be handled during dataset preparation. Camera points use right/down/forward axes and camera-z depth d. Ego points use forward/left/up axes. For a processed pixel q=(u,v,1),

\[
X_e=R_{e\leftarrow c}(dK^{-1}q)+t_{e\leftarrow c}.
\]

Pixel centers follow the center-aligned resize convention; processed intrinsics are `A K`. Flipping changes image coordinates through A without reflecting the physical ego world. BEV rows correspond to x, columns to y; all ranges are half open. Lane height is ego z. Lateral residual is measured from the BEV column center. There is no assumed flat road.

## Sparse Fusion Module: manuscript Eqs. 1–5

The auxiliary segmentation head predicts a lane probability; the depth head predicts a distribution over uniform camera-z bin centers. Default selection threshold is .3, depth K=3 of 40 bins at 2–80m, at most 4,096 image positions, and a minimum 16 high-score positions per view. These are engineering defaults, not manuscript parameters.

Selected features are scattered into active BEV cells with weights `lane_probability × original_depth_probability`. Features sharing a cell form a weighted mean:

\[
 F_C(b)=\frac{\sum_{(u,v,k)\mapsto b}p_{lane}(u,v)p_k(u,v)f(u,v)}{\sum_{(u,v,k)\mapsto b}p_{lane}(u,v)p_k(u,v)+\epsilon}.
\]

Top-K probabilities retain their original softmax mass. Integer selection/voxel coordinates are discrete. Feature values and weights remain differentiable, but there is no gradient through selection indices. The default first two epochs lift every image position; later epochs use selected positions. Auxiliary segmentation and observed depth provide direct supervision.

The portable LiDAR encoder forms sparse xyz voxels, embeds raw xyz/intensity plus mean-centered and voxel-center residuals through a point MLP, max-pools within voxels, then averages voxel features in BEV. This is an explicit replacement for the manuscript's unspecified deep sparse voxel backbone. Filtering removes nonfinite/out-of-range points; the training audit rejects nonfinite input files earlier.

Camera and LiDAR coordinates form a union; missing modality features are zero-filled and concatenated. A trainable 3×3 **submanifold** convolution looks up only active neighbors using sorted linear keys and `searchsorted`; the implementation is tested against dense convolution values and gradients. LayerNorm/SiLU refine values. This avoids a mandatory spconv/CUDA dependency but does not establish the manuscript's speed advantage. AMPM and ERA use a dense tensor after fusion. Disabling SFM in BEV ablations selects all pixels and all depth bins, retaining calibration and fusion. In image mode, SFM is bypassed entirely.

## AMPM: Eqs. 6–10

Pool over active support; empty support gives zero descriptors:

\[
z=\sigma(\beta_0)\,GAP_A(F)+(1-\sigma(\beta_0))\,GMP_A(F),\quad
\alpha=softmax(W_rz+b_r).
\]

Each branch uses an input-dependent softmax mixture of M=4 depthwise 3×3 base kernels, then pointwise channel mixing. Dilation branches default to 1,2,3. This factorization is an efficiency-oriented choice filling an unspecified detail. Routing weights are latent and unsupervised; they must not be interpreted as weather labels. Weighted branches pass through a two-layer 1×1 FFN and residual addition. There is no hard expert dispatch or claim that routing skips branch execution.

## ERA: Eqs. 11–17

Directional mean pooling creates height and width descriptors. Convolutions and sigmoid produce spatial gates; a bottleneck MLP on global average features produces channel gates. The spatial and channel tensors are concatenated, projected with 1×1 convolution, and added to the input. ERA refines features; no edge-image output or gradient supervision is invented from the schematic.

## Heads, objectives and reconstruction: Eqs. 18–24

A shared convolutional head predicts confidence logits, four-dimensional embeddings, lateral residual and height. Image mode upsamples maps to the processed image plane and supervises confidence/instances only, plus auxiliary segmentation. BEV mode uses all relevant terms:

\[
L=2L_c+0.5L_e+4L_o+L_h+0.5L_s+0.5L_d.
\]

Confidence/segmentation use BCE-with-logits, with optional capped positive weighting (default cap20, an introduced imbalance correction). Embedding loss uses within-instance hinge variance margin .5, between-center separation margin 1.5, and center regularization .001. At most 256 evenly spaced embeddings per instance bound memory. Centers and losses are weighted equally across lanes/frames.

Offset/height use MSE averaged over valid positive lane cells, correcting the manuscript's all-cell normalization. Height has a separate validity mask. Every 3D point here includes measured/annotated height; sparse camera-z supervision comes from real LiDAR and/or real annotated 3D keypoints after projection. Such ground truth is used only in target construction, never supplied as sensor features. Unavailable depth is ignored. No height/depth is derived from 2D images to create supervision.

BEV labels interpolate lateral/height along longitudinal cell centers. Multiple lanes hitting one cell are ignored. This head assumes each lane is locally representable as y(x),z(x); nonmonotonic loops and branching need a graph representation. Image targets rasterize instance polylines, ignoring overlaps. Width is defined in original-image pixels and scaled on resize; it is separate from the 30px evaluation width.

Inference thresholds confidence and performs bounded deterministic embedding mean-shift grouping, followed by one strongest cell per sampled row. BEV points are `(x_center, y_center + clipped_offset, height)`. Image points are inverse-transformed into original resolution. It can emit no lanes; it never fills absent predictions with ground truth. Linear interpolation samples image lanes at TuSimple's requested rows.

## Evaluation and uncertainty

* Binary pixel metrics include TN, accuracy and IoU. They are **diagnostics**, not the lane benchmark score.
* 2D instance diagnostic: rasterize 30px lines at original resolution, maximize cardinality of threshold-valid one-to-one matches and then overlap quality, at IoU .5/.75. OpenCV rasterization differs from the official SCNN evaluator, so these are not official CULane results.
* TuSimple protocol: angle-dependent 20px tolerance, .85 lane match threshold, last-frame point accuracy, special >4-lane FN/accuracy handling, >200ms rejection and excessive lane count rejection. This faithfully retains the official protocol's possibly nonunique GT matching semantics. `run_time` measures model+decoding/transfers excluding I/O; the timing boundary is disclosed.
* 3D diagnostic: symmetric mean Euclidean nearest-neighbor Chamfer after .5m arc-length resampling, threshold .5m. This differs from ONCE's projection/matching/directional distance implementation and is never labeled official ONCE F1/CD.
* Counts aggregate globally. Group percentile bootstrap resamples recording groups; fewer than two groups yields no interval. No frame-level pseudo-independence is asserted.

Original benchmark comparisons remain in the [manuscript tables](manuscript/tables.md). Full official evaluations require their released tools, input data and properly matched baselines.
