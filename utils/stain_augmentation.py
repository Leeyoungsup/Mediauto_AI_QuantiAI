"""
Advanced HnE Stain Augmentation
병리 이미지의 스캐너/병원별 염색 차이를 위한 고급 증강 기법

Based on:
- Macenko stain normalization
- Reinhard color normalization
- Random stain perturbation
"""

import numpy as np
import cv2


class StainAugmentor:
    """
    HnE stain augmentation for pathology images
    """
    
    @staticmethod
    def normalize_stain_macenko(image, target_concentrations=None):
        """
        Macenko stain normalization
        Reference: "A method for normalizing histology slides for quantitative analysis"
        """
        # Convert to optical density
        image = image.astype(np.float32)
        image = np.maximum(image, 1)  # Avoid log(0)
        od = -np.log(image / 255.0)
        
        # Remove background (pixels with very low OD)
        od_hat = od[~np.any(od < 0.15, axis=2)]
        
        if len(od_hat) == 0:
            return image.astype(np.uint8)
        
        # Compute eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
            eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
            
            # Project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
            t_hat = od_hat.dot(eigvecs[:, :2])
            
            # Find the min and max vectors
            phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])
            min_phi = np.percentile(phi, 1)
            max_phi = np.percentile(phi, 99)
            
            v1 = eigvecs[:, :2].dot(np.array([np.cos(min_phi), np.sin(min_phi)]))
            v2 = eigvecs[:, :2].dot(np.array([np.cos(max_phi), np.sin(max_phi)]))
            
            # Make sure v1 is H&E (typically H is more abundant)
            if v1[0] > v2[0]:
                he = np.array([v1, v2])
            else:
                he = np.array([v2, v1])
            
            # Normalize
            he = he / np.linalg.norm(he, axis=1)[:, None]
            
            # Random perturbation for augmentation
            if target_concentrations is None:
                # Apply random perturbation to stain matrix
                alpha = np.random.uniform(0.9, 1.1, size=2)
                beta = np.random.uniform(-0.05, 0.05, size=2)
                he = he * alpha[:, None] + beta[:, None]
            
            # Calculate concentrations
            h, w, c = image.shape
            od_flat = od.reshape(-1, 3)
            concentrations = np.linalg.lstsq(he.T, od_flat.T, rcond=None)[0].T
            
            # Reconstruct with augmented stain matrix
            od_reconstructed = concentrations.dot(he)
            image_reconstructed = np.exp(-od_reconstructed) * 255
            image_reconstructed = np.clip(image_reconstructed, 0, 255).reshape(h, w, c)
            
            return image_reconstructed.astype(np.uint8)
        
        except:
            # If normalization fails, return original
            return image.astype(np.uint8)
    
    @staticmethod
    def augment_stain_random(image, alpha_range=(0.8, 1.2), beta_range=(-0.1, 0.1)):
        """
        Random stain augmentation
        스캐너별 염색 차이를 시뮬레이션
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Augment L (brightness)
        alpha_l = np.random.uniform(alpha_range[0], alpha_range[1])
        beta_l = np.random.uniform(beta_range[0] * 10, beta_range[1] * 10)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * alpha_l + beta_l, 0, 255)
        
        # Augment A and B (color channels)
        for i in [1, 2]:
            alpha = np.random.uniform(alpha_range[0], alpha_range[1])
            beta = np.random.uniform(beta_range[0] * 10, beta_range[1] * 10)
            lab[:, :, i] = np.clip(lab[:, :, i] * alpha + beta, 0, 255)
        
        # Convert back to RGB
        image_aug = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return image_aug
    
    @staticmethod
    def augment_hed(image, h_range=(0.9, 1.1), e_range=(0.9, 1.1), d_range=(0.9, 1.1)):
        """
        HED (Hematoxylin-Eosin-DAB) color deconvolution augmentation
        """
        # HED color deconvolution matrix
        hed_from_rgb = np.array([
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
            [0.27, 0.57, 0.78]
        ])
        
        # Convert to optical density
        image = image.astype(np.float32)
        image = np.maximum(image, 1)
        od = -np.log(image / 255.0)
        
        # Apply color deconvolution
        h, w, c = image.shape
        od_flat = od.reshape(-1, 3)
        hed = od_flat.dot(np.linalg.inv(hed_from_rgb).T)
        
        # Augment each stain component
        h_factor = np.random.uniform(h_range[0], h_range[1])
        e_factor = np.random.uniform(e_range[0], e_range[1])
        d_factor = np.random.uniform(d_range[0], d_range[1])
        
        hed[:, 0] *= h_factor  # Hematoxylin
        hed[:, 1] *= e_factor  # Eosin
        hed[:, 2] *= d_factor  # DAB (not present in H&E, but included for completeness)
        
        # Convert back to RGB
        od_reconstructed = hed.dot(hed_from_rgb)
        image_reconstructed = np.exp(-od_reconstructed) * 255
        image_reconstructed = np.clip(image_reconstructed, 0, 255).reshape(h, w, c)
        
        return image_reconstructed.astype(np.uint8)


# Usage example:
# augmentor = StainAugmentor()
# augmented_image = augmentor.augment_stain_random(image)
# or
# augmented_image = augmentor.augment_hed(image)
