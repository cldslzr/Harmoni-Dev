from .models import (
    INSIGHTFACE_MODEL,
)

def get_face_embedding_model(name: str):
    """
    Returns the appropriate model configuration based on the given name.
    
    Args:
        name (str): Model identifier, e.g., "INSIGHTFACE"

    Returns:
        dict: Dictionary containing model and related components
    """
    if name == "INSIGHTFACE":
        return {
            "name": "INSIGHTFACE",
            "model": INSIGHTFACE_MODEL
        }
    else:
        raise ValueError(f"Unknown embedding model name: {name}")