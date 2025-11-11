## Setup

Setup poetry:
```bash
sudo apt update && yes | sudo apt install pipx && pipx ensurepath
yes | pipx install poetry
. ~/.bashrc
poetry config virtualenvs.in-project true
```

Install repo:
```bash
poetry install
```

Activate env:
```bash
. .venv/bin/activate
```

Download the model data from
https://mitprod-my.sharepoint.com/:f:/g/personal/nepfaff_mit_edu/Es7wAEHXwrFOi1KXUW8pPPsB3ToO68rylBRaQs73IMFOPw?e=yu3WMq

This is the Steerable Scene Generation dataset.
