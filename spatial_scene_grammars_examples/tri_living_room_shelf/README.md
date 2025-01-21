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

From the `tri_table` folder, create the model data using the following commands:

```bash
cp ~/efs/nicholas/scene_gen_data/anzu.zip . && yes | unzip anzu.zip
cp ~/efs/nicholas/scene_gen_data/gazebo.zip . && yes | unzip gazebo.zip
```


