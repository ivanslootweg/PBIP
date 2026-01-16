import torch

def pair_features(fg_features, bg_features, text_features, labels):
    batch_indices, class_indices = torch.where(labels == 1)
    
    paired_fg_features = [] 
    paired_bg_features = [] 
    paired_fg_text = [] 
    paired_bg_text = [] 
 
    for i in range(len(batch_indices)):
        curr_class = class_indices[i]
        
        curr_fg = fg_features[i]  # [D]
        curr_bg = bg_features[i]  # [D]
        
        curr_fg_text = text_features[curr_class]  # [D]
        
        bg_text_indices = [j for j in range(text_features.shape[0]) if j != curr_class]
        curr_bg_text = text_features[bg_text_indices]  # [3, D]
        
        paired_fg_features.append(curr_fg)
        paired_bg_features.append(curr_bg)
        paired_fg_text.append(curr_fg_text)
        paired_bg_text.append(curr_bg_text)
    
    paired_fg_features = torch.stack(paired_fg_features)  # [N, D]
    paired_bg_features = torch.stack(paired_bg_features)  # [N, D]
    paired_fg_text = torch.stack(paired_fg_text)         # [N, D]
    paired_bg_text = torch.stack(paired_bg_text)         # [N, 3, D]
    
    return {
        'fg_features': paired_fg_features,  
        'bg_features': paired_bg_features,  
        'fg_text': paired_fg_text,         
        'bg_text': paired_bg_text       
    }


def merge_to_parent_predictions(predictions, k_list, nk=1, method='max'):
    """Merge subclass predictions to parent predictions, accounting for Nk representatives per subclass.
    
    With Nk=1 (legacy): Directly merges K subclass predictions to parent classes
    With Nk>1: First merges K*Nk features to K subclass predictions, then to parent predictions
    
    Args:
        predictions: [batch_size, K*Nk] logits
        k_list: List of K (subclasses per parent class)
        nk: Number of representative features per subclass (default 1)
        method: Merge method ('mean' or 'max')
    
    Returns:
        [batch_size, num_parent_classes] merged predictions
    """
    parent_preds = []
    start_idx = 0
    
    for k in k_list:
        # For this parent class: select K*Nk features (K subclasses × Nk each)
        class_preds = predictions[:, start_idx:start_idx + k*nk]
        
        # First, merge Nk features per subclass to get K subclass predictions
        subclass_preds = []
        for subclass_idx in range(k):
            subclass_features = class_preds[:, subclass_idx*nk:(subclass_idx+1)*nk]
            if nk > 1:
                # Merge Nk representatives of this subclass
                if method == 'max':
                    subclass_prob = torch.softmax(subclass_features, dim=1)
                    subclass_pred = (subclass_prob * subclass_features).sum(dim=1)
                else:  # mean
                    subclass_pred = torch.mean(subclass_features, dim=1)
            else:
                subclass_pred = subclass_features.squeeze(1)
            subclass_preds.append(subclass_pred)
        
        # Stack to get [batch_size, K]
        subclass_preds = torch.stack(subclass_preds, dim=1)
        
        # Then merge K subclass predictions to parent
        if k > 1:
            if method == 'max':
                class_probs = torch.softmax(subclass_preds, dim=1)
                parent_pred = (class_probs * subclass_preds).sum(dim=1)
            else:  # mean
                parent_pred = torch.mean(subclass_preds, dim=1)
        else:
            parent_pred = subclass_preds.squeeze(1)
        
        parent_preds.append(parent_pred)
        start_idx += k*nk
    
    parent_preds = torch.stack(parent_preds, dim=1)
    return parent_preds


def merge_subclass_cams_to_parent(cams, k_list, nk=1, method='max'):
    """Merge subclass CAMs to parent CAMs, accounting for Nk representatives per subclass.
    
    Args:
        cams: [batch_size, K*Nk, H, W] CAM tensor
        k_list: List of K (subclasses per parent class)
        nk: Number of representative features per subclass (default 1)
        method: Merge method ('mean' or 'max')
    
    Returns:
        [batch_size, num_parent_classes, H, W] merged CAMs
    """
    batch_size, _, H, W = cams.shape
    num_parent_classes = len(k_list)

    parent_cams = torch.zeros(batch_size, num_parent_classes, H, W, 
                            device=cams.device, dtype=cams.dtype)
    
    start_idx = 0
    for parent_idx, k in enumerate(k_list):
        # For this parent class: select K*Nk CAMs
        class_cams = cams[:, start_idx:start_idx + k*nk, :, :]
        
        # First, merge Nk CAMs per subclass to get K subclass CAMs
        subclass_cams_list = []
        for subclass_idx in range(k):
            subclass_cams_nk = class_cams[:, subclass_idx*nk:(subclass_idx+1)*nk, :, :]
            if nk > 1:
                # Merge Nk representatives of this subclass
                if method == 'max':
                    B, nk_dim, H_cam, W_cam = subclass_cams_nk.shape
                    cams_flat = subclass_cams_nk.view(B, nk_dim, H_cam*W_cam)
                    cams_probs = torch.softmax(cams_flat, dim=1)
                    subclass_cam_flat = (cams_probs * cams_flat).sum(dim=1)
                    subclass_cam = subclass_cam_flat.view(B, H_cam, W_cam)
                else:  # mean
                    subclass_cam = torch.mean(subclass_cams_nk, dim=1)
            else:
                subclass_cam = subclass_cams_nk.squeeze(1)
            subclass_cams_list.append(subclass_cam)
        
        # Stack to get [batch_size, K, H, W]
        subclass_cams_stacked = torch.stack(subclass_cams_list, dim=1)
        
        # Then merge K subclass CAMs to parent
        if k > 1:
            if method == 'max':
                B, k_dim, H_cam, W_cam = subclass_cams_stacked.shape
                cams_flat = subclass_cams_stacked.view(B, k_dim, H_cam*W_cam)
                cams_probs = torch.softmax(cams_flat, dim=1)
                parent_cam_flat = (cams_probs * cams_flat).sum(dim=1)
                parent_cam = parent_cam_flat.view(B, H_cam, W_cam)
            else:  # mean
                parent_cam = torch.mean(subclass_cams_stacked, dim=1)
        else:
            parent_cam = subclass_cams_stacked.squeeze(1)
        
        parent_cams[:, parent_idx, :, :] = parent_cam
        start_idx += k*nk
    
    return parent_cams


def expand_parent_to_subclass_labels(parent_labels, k_list, nk=1):
    """Expand parent class labels to subclass labels, accounting for Nk representatives per subclass.
    
    With Nk=1 (legacy): Expands to K subclasses
    With Nk>1: Expands to K*Nk features (each subclass has Nk representatives)
    
    Args:
        parent_labels: [batch_size, num_parent_classes] one-hot parent labels
        k_list: List of K (subclasses per parent class)
        nk: Number of representative features per subclass (default 1 for backward compatibility)
    
    Returns:
        subclass_labels: [batch_size, sum(k_list)*nk] expanded labels
    """
    batch_size = parent_labels.size(0)
    total_subclasses = sum(k_list)
    total_features = total_subclasses * nk  # Account for Nk representatives
    
    subclass_labels = torch.zeros(batch_size, total_features, 
                                device=parent_labels.device, 
                                dtype=parent_labels.dtype)
    start_idx = 0
    for parent_idx, k in enumerate(k_list):
        parent_label = parent_labels[:, parent_idx:parent_idx+1]  # [batch_size, 1]
        # Each of the K subclasses has Nk representatives with the same parent label
        subclass_labels[:, start_idx:start_idx+k*nk] = parent_label.repeat(1, k*nk)
        start_idx += k*nk
    
    return subclass_labels 