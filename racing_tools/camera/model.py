import csv
import numpy as np
import logging
import os

class CameraModel:
    def __init__(self, model_name="Standard Pinhole", rmse=0.0, matrix=None, dist_coeffs=None):
        self.model_name = model_name
        self.rmse = rmse
        if matrix is None:
            self.matrix = np.eye(3)
        else:
            self.matrix = np.array(matrix)
            
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros(5)
        else:
            self.dist_coeffs = np.array(dist_coeffs)

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        model_name = "Unknown"
        rmse = 0.0
        fx = fy = cx = cy = 0.0
        dist_list = []

        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            # define a simple dict to hold values for easier parsing if needed, 
            # but linear scan is fine for small file
            rows = list(reader)
            
            # Simple parser based on known format
            # Row 0: Parameter, Value (Header)
            
            data_map = {row[0]: row[1] for row in rows if len(row) >= 2}
            
            model_name = data_map.get('Model', 'Unknown')
            rmse = float(data_map.get('RMSE', 0.0))
            
            fx = float(data_map.get('fx', 0.0))
            fy = float(data_map.get('fy', 0.0))
            cx = float(data_map.get('cx', 0.0))
            cy = float(data_map.get('cy', 0.0))
            
            # Collect distortion coefficients dist_0, dist_1, ...
            # We don't know how many there are upfront (5, 8, 4...)
            idx = 0
            while True:
                key = f'dist_{idx}'
                if key in data_map:
                    dist_list.append(float(data_map[key]))
                    idx += 1
                else:
                    break
        
        matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        dist_coeffs = np.array(dist_list)
        
        return cls(model_name, rmse, matrix, dist_coeffs)

    def save(self, filepath):
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Parameter', 'Value'])
                writer.writerow(['Model', self.model_name])
                writer.writerow(['RMSE', self.rmse])
                
                # Intrinsics
                writer.writerow(['fx', self.matrix[0,0]])
                writer.writerow(['fy', self.matrix[1,1]])
                writer.writerow(['cx', self.matrix[0,2]])
                writer.writerow(['cy', self.matrix[1,2]])
                
                # Distortion coeffs
                dist_flat = self.dist_coeffs.flatten()
                for i, d in enumerate(dist_flat):
                    writer.writerow([f'dist_{i}', d])
            
            logging.info(f"Saved {self.model_name} intrinsics to {filepath}")

        except IOError as e:
            logging.error(f"Error saving {filepath}: {e}")
            raise

    def __repr__(self):
        return (f"CameraModel(name='{self.model_name}', rmse={self.rmse:.4f}, "
                f"matrix={self.matrix.tolist()}, dist={self.dist_coeffs.tolist()})")
