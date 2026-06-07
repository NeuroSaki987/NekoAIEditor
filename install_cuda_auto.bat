@echo off
python -m pip install --upgrade pip

set IDX1=%NEKOAI_PYTORCH_CUDA_INDEX%
set IDX2=https://download.pytorch.org/whl/cu132
set IDX3=https://download.pytorch.org/whl/cu130
set IDX4=https://download.pytorch.org/whl/cu126
set IDX5=https://download.pytorch.org/whl/cu128

if not "%IDX1%"=="" (
  echo Trying PyTorch CUDA index: %IDX1%
  python -m pip install torch torchvision torchaudio --index-url %IDX1% && goto torch_ok
)

echo Trying PyTorch CUDA index: %IDX2%
python -m pip install torch torchvision torchaudio --index-url %IDX2% && goto torch_ok

echo Trying PyTorch CUDA index: %IDX3%
python -m pip install torch torchvision torchaudio --index-url %IDX3% && goto torch_ok

echo Trying PyTorch CUDA index: %IDX4%
python -m pip install torch torchvision torchaudio --index-url %IDX4% && goto torch_ok

echo Trying PyTorch CUDA index: %IDX5%
python -m pip install torch torchvision torchaudio --index-url %IDX5% && goto torch_ok

echo All CUDA wheel attempts failed. Install from the PyTorch selector, then run: python -m pip install -r requirements.txt
exit /b 1

:torch_ok
python -m pip install -r requirements.txt
python -c "import torch; print('Torch:', torch.__version__); print('Torch CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); [print(i, torch.cuda.get_device_name(i), 'CC', torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"
