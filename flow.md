## Stage 0: Prototype Coordinate Extraction

```mermaid
    graph TB
        subgraph Input["📥 INPUT: Configuration & Data Files"]
            A1["YAML Config"]
            A2["Split CSV<br/>train/val/test"]
            A3["Labels CSV<br/>image → class"]
            A4["WSI Directory<br/>.tif files"]
            A5["Coordinates Dir<br/>.npy files"]
        end

        subgraph Load["Load & Parse"]
            B1["Read Config<br/>paths, class_order, patch_size"]
            B2["Read Split CSV<br/>extract train column"]
            B3["Read Labels CSV<br/>build image→label dict"]
        end

        subgraph Organize["Organize by Class"]
            C1["Match files to classes<br/>using label values"]
            C2["Group files per class<br/>benign, tumor, etc"]
        end

        subgraph Sample["Sample WSIs & Coordinates"]
            D1["For each class:<br/>sample up to num_per_class WSIs"]
            D2["For each WSI:<br/>load .npy coordinate file"]
            D3["Randomly sample<br/>up to samples_per_wsi coords"]
            D4["Track source WSI name<br/>for each coordinate"]
        end

        subgraph Store["Store Coordinates<br/>in Slide2Vec Format"]
            E1["Create structured array<br/>x, y, tile_level, tile_size, wsi_name"]
            E2["Save per class<br/>benign.npy, tumor.npy"]
        end

        subgraph Output["📊 OUTPUT: Prototype Coordinates"]
            F1["prototype_coordinates/<br/>benign.npy"]
            F2["prototype_coordinates/<br/>tumor.npy"]
            F3["⚡ Patches extracted<br/>on-the-fly later"]
        end

        A1 --> B1
        A2 --> B2
        A3 --> B3
        
        B1 & B2 & B3 --> Load
        Load --> Organize
        
        A4 & A5 --> D2
        C2 --> D1
        D1 --> D2
        D2 --> D3
        D3 --> D4
        
        C1 --> C2
        Organize --> Sample
        
        D4 --> E1
        E1 --> E2
        
        Sample --> Store
        Store --> Output
        
        E2 --> F1
        E2 --> F2
        F1 & F2 --> F3

        style Input fill:#B3E5FC,stroke:#01579B,stroke-width:3px
        style Load fill:#FFE0B2,stroke:#E65100,stroke-width:2px
        style Organize fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
        style Sample fill:#E8F5E9,stroke:#1B5E20,stroke-width:2px
        style Store fill:#FCE4EC,stroke:#880E4F,stroke-width:2px
        style Output fill:#F1F8E9,stroke:#33691E,stroke-width:3px
```

## Stage 1: MedCLIP Feature Extraction

```mermaid
    graph TB
        subgraph Input["📥 INPUT"]
            A1["prototype_coordinates/<br/>benign.npy, tumor.npy"]
            A2["WSI Directory<br/>.tif files"]
            A3["Config"]
        end

        subgraph Process["Process Coordinates"]
            B1["Load structured arrays<br/>with wsi_name field"]
            B2["For each coordinate:<br/>x, y, wsi_name"]
            B3["Construct WSI path<br/>using wsi_name"]
            B4["OpenSlide.read_region<br/>224×224 patch"]
            B5["MedCLIP vision model<br/>forward pass"]
            B6["Generate 512-D<br/>feature vector"]
        end

        subgraph Aggregate["Aggregate Features"]
            C1["Stack embeddings<br/>per class"]
            C2["Build metadata<br/>class_names, wsi_sources"]
        end

        subgraph Output["📊 OUTPUT"]
            D1["medclip_exemplars.pkl<br/>raw embeddings"]
        end

        A1 & A2 & A3 --> B1
        B1 --> B2
        B2 --> B3
        B3 --> B4
        B4 --> B5
        B5 --> B6
        B6 --> C1
        C1 --> C2
        C2 --> D1

        style Input fill:#B3E5FC,stroke:#01579B,stroke-width:3px
        style Process fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
        style Aggregate fill:#E8F5E9,stroke:#1B5E20,stroke-width:2px
        style Output fill:#F1F8E9,stroke:#33691E,stroke-width:3px
```

## Stage 2: K-Means Clustering → Prototypes

```mermaid
    graph TB
        subgraph Input["📥 INPUT"]
            A1["medclip_exemplars.pkl<br/>all embeddings"]
            A2["Config<br/>k_list, class_order"]
        end

        subgraph Process["K-Means Clustering"]
            B1["Load embeddings<br/>group by class"]
            B2["For each class:<br/>k-means (k=3)"]
            B3["Extract cluster centers<br/>3 benign + 3 tumor"]
            B4["Combine into prototype<br/>tensor: 6 × 512"]
        end

        subgraph Metadata["Create Metadata"]
            C1["k_list: [3, 3]"]
            C2["class_order: [benign, tumor]"]
            C3["cumsum_k: [0, 3, 6]"]
        end

        subgraph Output["📊 OUTPUT"]
            D1["label_fea_pro.pkl<br/>prototypes + metadata"]
        end

        A1 & A2 --> B1
        B1 --> B2
        B2 --> B3
        B3 --> B4
        B4 --> C1
        C1 & C2 & C3 --> D1

        style Input fill:#B3E5FC,stroke:#01579B,stroke-width:3px
        style Process fill:#E8F5E9,stroke:#1B5E20,stroke-width:2px
        style Metadata fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
        style Output fill:#F1F8E9,stroke:#33691E,stroke-width:3px
```

## Stage 3: Training with Weak Supervision

```mermaid
    graph TB
        subgraph Input["📥 INPUT"]
            A1["label_fea_pro.pkl<br/>prototypes"]
            A2["WSI Directory<br/>.tif files"]
            A3["Coordinates<br/>.npy files"]
            A4["Labels CSV<br/>image → class"]
            A5["Split CSV<br/>train/val"]
        end

        subgraph TrainLoop["Training Loop (per epoch)"]
            B1["Load TileDataset<br/>training split"]
            B2["For each batch:<br/>random coordinate"]
            B3["OpenSlide region read<br/>224×224 patch"]
            B4["Albumentations:<br/>Normalize + Augment"]
        end

        subgraph Forward["Forward Pass"]
            C1["SegFormer backbone<br/>multi-scale features"]
            C2["Prototype matching<br/>cosine similarity"]
            C3["Classification logits<br/>4 scales"]
            C4["CAM generation<br/>at each scale"]
            C5["Extract FG/BG<br/>features from CAMs"]
        end

        subgraph Loss["Loss Computation"]
            D1["Classification Loss<br/>0.0×L1 + 0.1×L2 + 1.0×L3 + 1.0×L4"]
            D2["Contrastive Loss<br/>0.1×FG + 0.1×BG InfoNCE"]
            D3["CAM Regularization<br/>smooth + sparsity"]
            D4["Total Loss<br/>= L_cls + L_contra + L_cam"]
        end

        subgraph Update["Weight Update"]
            E1["Backward Pass"]
            E2["Compute Gradients"]
            E3["Optimizer Step<br/>AdamW"]
        end

        subgraph Validation["Validation Loop"]
            F1["Load val split<br/>deterministic coords"]
            F2["Forward pass<br/>no augmentation"]
            F3["Generate masks<br/>from CAMs"]
            F4["Compare with GT<br/>compute metrics"]
            F5["Save best checkpoint<br/>based on IoU"]
        end

        subgraph Output["📊 OUTPUT"]
            G1["checkpoints/best.pth<br/>trained weights"]
            G2["TensorBoard logs<br/>losses, metrics"]
        end

        A1 & A2 & A3 & A4 & A5 --> B1
        B1 --> B2
        B2 --> B3
        B3 --> B4
        B4 --> C1
        C1 --> C2
        C2 --> C3
        C3 --> C4
        C4 --> C5
        C5 --> D1
        D1 & D2 & D3 --> D4
        D4 --> E1
        E1 --> E2
        E2 --> E3
        E3 -->|loop| B2
        B1 -.->|every N batches| F1
        F1 --> F2
        F2 --> F3
        F3 --> F4
        F4 --> F5
        F5 --> G1
        B2 -->|log losses| G2

        style Input fill:#B3E5FC,stroke:#01579B,stroke-width:3px
        style TrainLoop fill:#FFE0B2,stroke:#E65100,stroke-width:2px
        style Forward fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
        style Loss fill:#FCE4EC,stroke:#880E4F,stroke-width:2px
        style Update fill:#F9C6B8,stroke:#BF360C,stroke-width:2px
        style Validation fill:#E8F5E9,stroke:#1B5E20,stroke-width:2px
        style Output fill:#F1F8E9,stroke:#33691E,stroke-width:3px
```

## Stage 4: Validation & Evaluation

```mermaid
    graph TB
        subgraph Input["📥 INPUT"]
            A1["Trained Model<br/>best.pth"]
            A2["Test Split<br/>coordinates"]
            A3["WSI Directory<br/>.tif files"]
            A4["Ground Truth Masks"]
            A5["Config"]
        end

        subgraph Eval["Evaluation Process"]
            B1["Load model<br/>set to eval mode"]
            B2["For each test WSI:<br/>deterministic coord"]
            B3["Extract patch<br/>from WSI"]
            B4["Forward pass<br/>no augmentation"]
            B5["Generate segmentation<br/>mask from CAMs"]
            B6["Load corresponding<br/>GT mask"]
            B7["Crop GT to patch<br/>region"]
        end

        subgraph Metrics["Compute Metrics"]
            C1["IoU: TP/(TP+FP+FN)"]
            C2["Dice: 2×TP/(2×TP+FP+FN)"]
            C3["Precision, Recall, F1"]
            C4["Aggregate:<br/>macro + weighted avg"]
        end

        subgraph Report["Generate Report"]
            D1["metrics.json"]
            D2["confusion_matrix.csv"]
            D3["prediction_samples/<br/>input, GT, pred"]
            D4["evaluation_report.txt"]
        end

        A1 & A2 & A3 & A4 & A5 --> B1
        B1 --> B2
        B2 --> B3
        B3 --> B4
        B4 --> B5
        B5 --> B6
        B6 --> B7
        B7 --> C1
        B7 --> C2
        B7 --> C3
        C1 & C2 & C3 --> C4
        C4 --> D1
        C4 --> D2
        B5 --> D3
        C4 --> D4

        style Input fill:#B3E5FC,stroke:#01579B,stroke-width:3px
        style Eval fill:#FFE0B2,stroke:#E65100,stroke-width:2px
        style Metrics fill:#E8F5E9,stroke:#1B5E20,stroke-width:2px
        style Report fill:#F1F8E9,stroke:#33691E,stroke-width:3px
```

## Stage 5: Inference on New WSIs

```mermaid
    graph TB
        subgraph Input["📥 INPUT"]
            A1["Trained Model<br/>best.pth"]
            A2["New WSI Directory<br/>.tif files"]
            A3["New Coordinates<br/>.npy files"]
            A4["Config"]
        end

        subgraph Extract["Batch Extraction"]
            B1["Load all coordinates<br/>for new WSI"]
            B2["Batch load patches<br/>via OpenSlide"]
            B3["Preprocess:<br/>Normalize only"]
        end

        subgraph Predict["Batch Prediction"]
            C1["Batch forward pass<br/>on GPU"]
            C2["Generate logits<br/>& CAMs"]
            C3["Convert CAMs to<br/>segmentation masks"]
        end

        subgraph Assemble["Assemble Full WSI"]
            D1["Handle patch overlaps<br/>average/voting"]
            D2["Apply post-processing:<br/>morphological ops"]
            D3["Connected component<br/>filtering"]
            D4["Upsample to<br/>full resolution"]
        end

        subgraph Output["📊 OUTPUT"]
            E1["predictions/<br/>segmentation_mask.png"]
            E2["confidence_maps.npz<br/>per-class probabilities"]
            E3["visualization.png<br/>RGB overlay"]
            E4["metadata.json<br/>timing, version"]
        end

        A1 & A2 & A3 & A4 --> B1
        B1 --> B2
        B2 --> B3
        B3 --> C1
        C1 --> C2
        C2 --> C3
        C3 --> D1
        D1 --> D2
        D2 --> D3
        D3 --> D4
        D4 --> E1
        D4 --> E2
        D4 --> E3
        C1 --> E4

        style Input fill:#B3E5FC,stroke:#01579B,stroke-width:3px
        style Extract fill:#FFE0B2,stroke:#E65100,stroke-width:2px
        style Predict fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
        style Assemble fill:#FCE4EC,stroke:#880E4F,stroke-width:2px
        style Output fill:#F1F8E9,stroke:#33691E,stroke-width:3px
```

## Complete End-to-End Pipeline

```mermaid
    graph TB
        subgraph Stage0["Stage 0️⃣: Prototype Extraction"]
            A1["extract_prototypes_from_gt.py"]
            A2["Output: prototype_coordinates/<br/>benign.npy, tumor.npy"]
        end

        subgraph Stage1["Stage 1️⃣: MedCLIP Features"]
            B1["excract_medclip_proces.py"]
            B2["Output: medclip_exemplars.pkl"]
        end

        subgraph Stage2["Stage 2️⃣: K-Means Clustering"]
            C1["k_mean_cos_per_class.py"]
            C2["Output: label_fea_pro.pkl<br/>6 prototypes"]
        end

        subgraph Stage3["Stage 3️⃣: Training"]
            D1["train_stage_1.py"]
            D2["Weak supervision training<br/>8 epochs"]
            D3["Output: checkpoints/best.pth<br/>TensorBoard logs"]
        end

        subgraph Stage4["Stage 4️⃣: Validation"]
            E1["utils/validate.py"]
            E2["Evaluate on test set<br/>with GT masks"]
            E3["Output: metrics.json<br/>confusion_matrix.csv"]
        end

        subgraph Stage5["Stage 5️⃣: Inference"]
            F1["inference.py"]
            F2["Segment new WSIs<br/>batch processing"]
            F3["Output: predictions/<br/>masks + confidence maps"]
        end

        subgraph Final["📊 FINAL RESULTS"]
            G1["Trained Model<br/>weights"]
            G2["Validation Metrics"]
            G3["WSI Predictions<br/>& visualizations"]
        end

        Input["Input Data:<br/>Split CSV, Labels CSV,<br/>WSI Dir, Coordinates Dir"]
        
        Input --> Stage0
        Stage0 --> A1
        A1 --> A2
        
        A2 --> Stage1
        Stage1 --> B1
        B1 --> B2
        
        B2 --> Stage2
        Stage2 --> C1
        C1 --> C2
        
        C2 --> Stage3
        Input --> Stage3
        Stage3 --> D1
        D1 --> D2
        D2 --> D3
        
        D3 --> Stage4
        Stage4 --> E1
        E1 --> E2
        E2 --> E3
        
        D3 --> Stage5
        Stage5 --> F1
        F1 --> F2
        F2 --> F3
        
        A2 --> Final
        E3 --> Final
        F3 --> Final
        D3 --> G1
        E3 --> G2
        F3 --> G3

        style Input fill:#B3E5FC,stroke:#01579B,stroke-width:3px
        style Stage0 fill:#FFE0B2,stroke:#E65100,stroke-width:2px
        style Stage1 fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
        style Stage2 fill:#E8F5E9,stroke:#1B5E20,stroke-width:2px
        style Stage3 fill:#FCE4EC,stroke:#880E4F,stroke-width:2px
        style Stage4 fill:#F9C6B8,stroke:#BF360C,stroke-width:2px
        style Stage5 fill:#E0F2F1,stroke:#004D40,stroke-width:2px
        style Final fill:#F1F8E9,stroke:#33691E,stroke-width:3px
```
