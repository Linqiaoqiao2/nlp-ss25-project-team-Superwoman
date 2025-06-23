import os
from util.logger_config import setup_logger
from pathlib import Path


logger = setup_logger("FileUtil", log_file="logs/RAGPipeline.log")


class FileUtil:

    @staticmethod
    def get_first_file_path_dir(directory_path):
        # List all files in the directory
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.lower().endswith(".gguf")]

        # Sort files alphabetically (or by any other criteria)
        files.sort()

        if not files:
            raise FileNotFoundError("No files found in the directory.")

        # Get the first file
        first_file = files[0]
        logger.info(f"Model Choosen: {first_file}")

        model_file_path = os.path.join(directory_path, first_file)
        return model_file_path

    @staticmethod
    def get_model_filePath():
        current_dir = os.path.dirname(__file__)
        dir = os.path.join(current_dir,"../llms")
        logger.info(f"llm directory Choosen: {dir}")
        return FileUtil.get_first_file_path_dir(dir)
    

    @staticmethod
    def get_all_cleaned_data_filenames():
        current_dir = Path(__file__).parent
        data_dir = (current_dir / "../data").resolve()
        print(f"Data directory path: {data_dir.resolve()}")

        # Find all subdirectories starting with 'cleaned_'
        cleaned_dirs = [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("cleaned_")]
        if not cleaned_dirs:
            raise FileNotFoundError(f"No folder starting with 'cleaned_' found in {data_dir}")
        
        cleaned_dir = cleaned_dirs[0]

        allowed_suffixes = {".txt", ".md", ".pdf"}
        document_paths = [
            f.as_posix()
            for f in cleaned_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in allowed_suffixes
        ]
        return document_paths