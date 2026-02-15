"""
Production Malicious URL Detection System - v7
Main entry point for inference
"""

import argparse
import json
import sys
import logging
from pathlib import Path

from src.inference import ProductionInferenceEngine
from src.utils import setup_logging


def load_config() -> dict:
    """Load production configuration"""
    return {
        "model_path": "models/production/model_v7.h5",
        "preprocessor_path": "models/production/preprocessor.pkl",
        "log_level": "INFO",
        "log_file": None
    }


def predict_url(url: str, include_metadata: bool = False) -> dict:
    """
    Predict URL classification
    
    Args:
        url: URL to classify
        include_metadata: Include detailed metadata
        
    Returns:
        Prediction result dictionary
    """
    config = load_config()
    
    # Initialize inference engine
    engine = ProductionInferenceEngine(
        model_path=config["model_path"],
        preprocessor_path=config["preprocessor_path"]
    )
    
    # Run prediction
    result = engine.predict(url, include_metadata=include_metadata)
    
    return result


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Malicious URL Detection System - v7 Production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --predict "https://google.com"
  python main.py --predict "http://phishing-site.tk" --metadata
  python main.py --batch urls.txt --output results.json
        """
    )
    
    parser.add_argument(
        '--predict',
        type=str,
        help='Single URL to classify'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='File containing URLs (one per line)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path'
    )
    
    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Include detailed metadata in output'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not args.predict and not args.batch:
        parser.print_help()
        sys.exit(1)
    
    try:
        results = []
        
        # Single prediction
        if args.predict:
            logger.info(f"Processing single URL: {args.predict}")
            result = predict_url(args.predict, include_metadata=args.metadata)
            results.append(result)
        
        # Batch prediction
        elif args.batch:
            logger.info(f"Processing batch file: {args.batch}")
            
            with open(args.batch, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            config = load_config()
            engine = ProductionInferenceEngine(
                model_path=config["model_path"],
                preprocessor_path=config["preprocessor_path"]
            )
            
            for url in urls:
                result = engine.predict(url, include_metadata=args.metadata)
                results.append(result)
            
            logger.info(f"Processed {len(urls)} URLs")
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        else:
            # Print to stdout
            output = results[0] if len(results) == 1 else results
            print(json.dumps(output, indent=2))
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        print(json.dumps({
            "error": str(e),
            "status": "failed"
        }), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
