import argparse
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, login

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str, required=True, help="HuggingFace repo id, e.g. username/vit-lora-cifar100")
    parser.add_argument('--weights_dir', type=str, default='weights')
    args = parser.parse_args()

    login(token=os.getenv("HF_TOKEN"))
    api = HfApi()
    try:
        create_repo(args.repo_id, exist_ok=True)
    except Exception as e:
        print(f"Repo may already exist: {e}")

    api.upload_folder(
        folder_path=args.weights_dir,
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Weights uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == '__main__':
    main()
