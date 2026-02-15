"""
Model loader for production inference
Production version - v7
"""

from keras.models import load_model
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Production model loader"""
    
    @staticmethod
    def load_production_model(model_path: str) -> object:
        """
        Load Keras model for production inference
        
        Args:
            model_path: Path to model .h5 file
            
        Returns:
            Loaded Keras model
        """
        try:
            model = load_model(model_path, compile=False)
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise
    
    @staticmethod
    def get_model_info(model: object) -> dict:
        """
        Extract model information
        
        Args:
            model: Loaded Keras model
            
        Returns:
            Dictionary with model metadata
        """
        try:
            return {
                "total_parameters": model.count_params(),
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape),
                "layers": len(model.layers)
            }
        except Exception as e:
            logger.warning(f"Failed to extract model info: {str(e)}")
            return {}
