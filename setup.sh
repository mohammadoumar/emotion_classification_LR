rm -rf LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
ls

pip install -e .[torch,bitsandbytes]

pip uninstall -y pydantic
pip install pydantic==2.9.0

pip uninstall -y gradio
pip install gradio==4.43.0

pip uninstall -y bitsandbytes
pip install --upgrade bitsandbytes

pip install tqdm
pip install ipywidgets
pip install scikit-learn

pip install --upgrade transformers