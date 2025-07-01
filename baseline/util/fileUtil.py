import os
from pathlib import Path
from util.logger_config import setup_logger

logger = setup_logger("FileUtil", log_file="logs/RAGPipeline.log")

class FileUtil:
    @staticmethod
    def get_first_file_path_dir(directory_path):
        """
        Returns the first .gguf file found in the specified directory.
        """
        files = [
            f for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f)) and f.lower().endswith(".gguf")
        ]
        files.sort()

        if not files:
            raise FileNotFoundError("No .gguf files found in the specified directory.")

        first_file = files[0]
        logger.info(f"LLM model chosen: {first_file}")

        return os.path.join(directory_path, first_file)

    @staticmethod
    def get_model_filePath():
        """
        Returns the path to the first .gguf model in the ../llms directory.
        """
        current_dir = Path(__file__).parent
        model_dir = (current_dir / "../llms").resolve()
        logger.info(f"LLM directory: {model_dir}")
        return FileUtil.get_first_file_path_dir(model_dir)

    @staticmethod
    def get_all_cleaned_data_filenames():
        """
        Returns a list of all file paths in the first 'cleaned_' directory inside ../data.
        Accepts .txt, .md, .pdf, and .json files.
        """
        current_dir = Path(__file__).parent
        data_dir = (current_dir / "../data").resolve()
        print(f"Data directory path: {data_dir}")

        # Find all 'cleaned_' directories
        cleaned_dirs = [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("cleaned_")]
        if not cleaned_dirs:
            raise FileNotFoundError(f"No folder starting with 'cleaned_' found in {data_dir}")

        cleaned_dir = cleaned_dirs[0]

        allowed_suffixes = {".txt", ".md", ".pdf", ".json"}
        document_paths = [
            f.as_posix()
            for f in cleaned_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in allowed_suffixes
        ]
        return document_paths

    @staticmethod
    def get_url_from_path(path: str) -> str:
        """
        Maps specific cleaned_json filenames to their corresponding URLs for citation.
        This is optional if you prefer not to rely solely on the 'url' inside JSON files.
        """
        mapping = {
            "Career_Prospects": "https://www.uni-bamberg.de/en/ma-ai/career-prospects/",
            "Application_and_Admission": "https://www.uni-bamberg.de/en/ma-ai/application-and-admission/",
            "Content_and_Structure": "https://www.uni-bamberg.de/en/ma-ai/content-and-structure/",
            "Qualification_Objectives": "https://www.uni-bamberg.de/en/ma-ai/qualification-objectives/",
            "Reasons_for_Study": "https://www.uni-bamberg.de/en/ma-ai/reasons-for-study/",
            "Profile": "https://www.uni-bamberg.de/en/ma-ai/profile/",
            "Part_Time_Study": "https://www.uni-bamberg.de/en/ma-ai/part-time-study/",
            "Contact_Persons": "https://www.uni-bamberg.de/en/ma-ai/contact-persons/",
            "FAQ": "https://www.uni-bamberg.de/en/ma-ai/faq/",
            "Regulations_and_documents": "https://www.uni-bamberg.de/en/ma-ai/regulations-and-documents/",
            "Structure_and_Curriculum": "https://www.uni-bamberg.de/en/ma-ai/structure-and-curriculum/",
        }
        for key in mapping:
            if key.lower() in path.lower():
                return mapping[key]
        return "N/A"
