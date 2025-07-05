# Tutorial 2: Configuring a New Experiment

This tutorial demonstrates how to modify the existing configuration to run a new
experiment. We will change the model's encoder to see how it affects performance.

**Goal**: Learn to use Hydra's command-line overrides to launch custom experiments.
**Prerequisite**: [Tutorial 1: Running Your First Training](01_basic_training.md)

---

## Step 1: Understand the Configuration Structure

Our project uses Hydra, which means all settings are stored in YAML files inside the
`configs/` directory. The structure of this directory mirrors the project's code
structure.

For example, available model encoders are defined in `configs/model/encoder/`.
Take a moment to explore the files in that directory. You might see `resnet34.yaml`,
`efficientnet_b4.yaml`, etc.

## Step 2: Choose a New Component

Let's assume the default configuration (`configs/base.yaml`) uses the `resnet34`
encoder. For our new experiment, we want to try the `efficientnet_b4` encoder instead.

## Step 3: Launch Training with an Override

You do **not** need to edit any YAML files directly. Hydra's power comes from its
command-line overrides. We will launch the training from the terminal and tell Hydra
to swap the encoder.

1. Activate the `crackseg` environment: `conda activate crackseg`
2. Run the main training script with a specific override for `model/encoder`:

```bash
python src/main.py --config-name base model/encoder=efficientnet_b4
```

### Deconstructing the Command

- `python src/main.py`: We run the main script directly, bypassing the GUI for this
    CLI-focused tutorial.
- `--config-name base`: This tells Hydra to start with `configs/base.yaml` as the
    foundation.
- `model/encoder=efficientnet_b4`: This is the override. It instructs Hydra to
    ignore the default value in `model/encoder` and instead use the configuration
    found in `configs/model/encoder/efficientnet_b4.yaml`.

## Step 4: Observe the Output

The training will start in your terminal. Pay attention to the initial output from
Hydra, which will show the composed configuration. You should see that the `encoder`
is now set to `efficientnet_b4`.

The results of this run will be saved to a new, unique directory in `outputs/`,
named according to the date and time of the run. This keeps your experiment
results neatly organized and separate from one another.

---

**Congratulations!** You have successfully launched a custom experiment. You can use
this override technique for any configuration parameter in the `configs/` directory,
allowing for rapid iteration and testing of new ideas.
