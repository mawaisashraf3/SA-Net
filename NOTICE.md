# Attribution and rights

The new implementation is licensed under MIT. The user-supplied manuscript, original figures, reported result tables, and underlying dataset imagery are **not** relicensed by the code license; their rights remain with their respective authors/owners. Figures and tables were included at the user's request with explicit source labels. No original manuscript author code or pretrained SA-Net weights were provided.

Research foundation: “SA-Net: Sparse Multimodal Fusion with Adaptive Scene Perception for Robust Lane Detection,” user-supplied `SA.docx`; publication venue/year/DOI and full author names were not supplied in a reliable citation field. Do not cite this independent implementation as the original author implementation or invent publication metadata. The source checksum and all 43 supplied reference entries appear under `docs/manuscript/`.

Research context includes SparseFusion, VoxelNet, ResNet, BANet and BEV-LaneDet, as cited in the manuscript. The implementation follows the manuscript's equations with documented corrections and independent design choices. Related methods and datasets should be cited in any research using them.

TuSimple protocol semantics follow the Apache-2.0 `TuSimple/tusimple-benchmark` reference evaluator. This repository contains a separately written NumPy implementation of that protocol and acknowledges TuSimple's contribution. Torch/torchvision, NumPy, SciPy, Pillow, OpenCV, PyYAML, Matplotlib and developer tools retain their own licenses.

The small experiment uses locally acquired real images/annotations from a pinned third-party TuSimple mirror; provenance includes its license assertion, without claiming verification of original image redistribution terms. Downloaded data are excluded from Git. Original manuscript performance values are attributed claims; newly executed results are separately labeled, including failed lane-instance detection.
