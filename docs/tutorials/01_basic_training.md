# Tutorial 1: Running Your First Training

This tutorial guides you through the process of launching a basic training run using the
default project settings. It's the best way to verify that your installation is working
correctly.

**Goal**: Start a training run and see the model learn.
**Time**: ~5-10 minutes.

---

## Step 1: Activate the Environment

Open your terminal, navigate to the project's root directory, and activate the `crackseg`
Conda environment. All subsequent commands must be run from this activated environment.

```bash
conda activate crackseg
```

## Step 2: Launch the GUI

The easiest way to start training is through the Streamlit-based Graphical User Interface.
Launch it with the following command:

```bash
streamlit run scripts/gui/app.py
```

Your web browser should open with the application loaded.

> **[Screenshot: The main welcome page of the CrackSeg GUI.]**

## Step 3: Start the Training Process

For this first run, we will use all the default settings.

1. Navigate to the **Train** page using the sidebar.
2. You will see the default configuration loaded, typically using a U-Net model.
3. Click the **Start Training** button.

The training process will begin in the background.

## Step 4: Monitor the Training

On the **Train** page, you can monitor the progress in real-time:

- **Live Metrics**: Watch the loss decrease and other metrics (like IoU) increase.
- **Log Output**: See detailed log messages from the training engine.
- **GPU Usage**: Keep an eye on the GPU resource monitor.

> **[Screenshot: The 'Train' page during a run, showing live charts and log output.]**

## Step 5: View the Results

Once the training is complete (or after a few epochs have run), you can view the
initial results.

1. Navigate to the **Results** page from the sidebar.
2. Here you will find a gallery of predictions made by the model on the validation set.
    You should see the model starting to produce rough outlines of the cracks.

> **[Screenshot: The 'Results' page showing a grid of images, masks, and model predictions.]**

---

**Congratulations!** You have successfully launched your first training run. You can now
explore other tutorials to learn how to customize experiments and extend the project.
