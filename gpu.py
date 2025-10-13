import torch
def main():


    print("CUDA available:", torch.cuda.is_available())
    print("PyTorch built with CUDA:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

if __name__ == "__main__":
    main()