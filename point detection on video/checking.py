import pandas as pd
from scipy.spatial.distance import euclidean

def compare_landmarks(image_landmarks, excel_landmarks):
    """
    Compare image landmarks with Excel landmarks and return matching status.
    """
    # Assuming both lists have the same length and order of landmarks
    for i, (img_lmk, exc_lmk) in enumerate(zip(image_landmarks, excel_landmarks)):
        # Calculate Euclidean distance between corresponding landmarks
        dist = euclidean(img_lmk, exc_lmk)
        
        # Define a threshold distance for similarity
        threshold = 10  # Adjust this value based on your requirements
        
        # Check if the distance is within the threshold
        if dist <= threshold:
            return f"Match at Landmark {i + 1}"
    
    # If none of the landmarks matched within the threshold
    return "Doesn't Match"

# Example usage
excel_file_path = 'path/to/your/excel/file.xlsx'
df = pd.read_excel(excel_file_path)