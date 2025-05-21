import os


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
        print(f"First file: {first_file}")
        model_file_path = os.path.join(directory_path, first_file)
        print(f"model path file: {model_file_path}")

        return model_file_path

    @staticmethod
    def get_model_filePath():
        current_dir = os.path.dirname(__file__)
        dir = os.path.join(current_dir,"../llms")
        return FileUtil.get_first_file_path_dir(dir)