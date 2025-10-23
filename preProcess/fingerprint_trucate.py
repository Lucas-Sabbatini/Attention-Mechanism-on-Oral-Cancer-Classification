from pathlib import Path
import numpy as np

class WavenumberTruncator:
    def __init__(self, filepath = "wavenumbers_cancboca.dat"):
        self.filepath = Path(filepath)
        self._cache_file_info()
    
    def _cache_file_info(self):
        """Cache file size and line positions for efficient binary search"""
        with open(self.filepath, 'rb') as f:
            self.line_positions = [0]  # First line starts at position 0
            while f.readline():
                self.line_positions.append(f.tell())

            self.line_positions.pop()
            self.total_lines = len(self.line_positions)
    
    def _get_line_value(self, line_number):
        """Get the float value at a specific line number (0-indexed)"""
        with open(self.filepath, 'rb') as f:
            f.seek(self.line_positions[line_number])
            line = f.readline().decode('utf-8').strip()
            return float(line)
    
    def binary_search_closest(self, target):
        """
        Binary search for the closest wavenumber value.
        Returns: (line_number, value, exact_match)
        - line_number: 0-indexed line number in the file
        - value: the closest wavenumber found
        - exact_match: True if exact match, False if closest
        """
        left, right = 0, self.total_lines - 1
        closest_line = 0
        closest_value = self._get_line_value(0)
        closest_diff = abs(target - closest_value)
        
        while left <= right:
            mid = (left + right) // 2
            mid_value = self._get_line_value(mid)
            
            # Update closest if this is closer
            diff = abs(target - mid_value)
            if diff < closest_diff:
                closest_diff = diff
                closest_line = mid
                closest_value = mid_value
            
            # Check for exact match
            if abs(mid_value - target) < 1e-9:  # floating point comparison
                return mid, mid_value, True
            
            # Since file is in DESCENDING order
            if mid_value > target:
                left = mid + 1
            else:
                right = mid - 1
        
        return closest_line, closest_value, False
    
    def trucate_range(self, lower_bound :int, upper_bound :int, X:np.ndarray):
        """
        Truncate the dataset to only include wavenumbers within [lower_bound, upper_bound].
        Remember all the metrics are in reciprocal centimeter (DESC order). 
        So the lower_bound is actually larger than upper_bound.
        """
        upper_line, _, _ = self.binary_search_closest(upper_bound)  
        lower_line, _, _ = self.binary_search_closest(lower_bound)  

        return X[:, lower_line:upper_line + 1]
        
        
# Usage example
if __name__ == "__main__":
    # Initialize with your file
    searcher = WavenumberTruncator("wavenumbers_cancboca.dat")
    
    # Search for a specific wavenumber
    target = 900.0
    line_num, value, exact = searcher.binary_search_closest(target)
    
    print(f"Searching for: {target}")
    print(f"Found at line: {line_num} (Starts at first line = 0)")
    print(f"Value: {value}")
    print(f"Exact match: {exact}")

    # Example dataset
    dataset = np.loadtxt("dataset_cancboca.dat")
    X = dataset[:,:-1]
    y = dataset[:,-1].astype(int)

    # Truncate dataset to wavenumber range [1800, 900]
    X_truncated = searcher.trucate_range(1800, 900, X)
    print(f"Original shape: {X.shape}, Truncated shape: {X_truncated.shape}")