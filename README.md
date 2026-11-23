---
pretty_name: CS541 Fall 2026 CPR Challenge
license: other
license_name: cs541-coursework-only
license_link: LICENSE.md
task_categories:
- video-classification
viewer: false
---

# CS541 Fall 2026 class challenge: CPR error detection

## The task

Each video shows one person performing CPR chest compressions on a training manikin, filmed by one camera. Your job is to say **which of 13 technique errors are present** in each video. A video can contain none, one, or several errors.

The twist is that **every training video contains at most one error**, while many test videos contain **two to four errors at once**. Some test videos also come from **a camera angle that never appears in training**. The question is how well your model generalizes to error combinations, and viewpoints, it has never seen.

## What's in this folder

```
/projectnb/cs541/materials/cpr_challenge/
├── README.md               this file
├── train.csv               1,008 training videos and their labels
├── test_ids.csv            4,320 test video ids (no labels)
├── sample_submission.csv   the exact format Kaggle expects
├── videos/
│   ├── train/<video_id>.mp4
│   └── test/<video_id>.mp4
├── flow_pose_subset/       100 videos as 960 × 540 frames, with reference optical flow and
│                           keypoints to evaluate your models against in the report
├── downloads/              the same files as 3 zips, to copy to your laptop or Colab
└── starter/
    ├── cpr_data.py         finds and reads the data: the SCC copy, your own copy, or Hugging Face
    ├── starter.py          loads a video, writes a valid submission
    └── train_job.sh        an example GPU batch job for the SCC (submit with qsub)
```

**On the SCC, read the data in place. Do not copy it into your own SCC folder;** the course space has a shared quota. To work on your laptop or Colab instead, download the zips (below).

## Getting the data

There are three ways. `starter/cpr_data.py` works with all of them, so your code stays the same.

**1. On the SCC (recommended):** nothing to download. `cpr_data.py` reads the files in place.

**2. Your own copy, for a laptop or Colab:** `downloads/` holds the same files as three zips.

| Zip | Contents | Size |
|---|---|---|
| `cpr_train.zip` | this README, the CSVs, `starter/`, the training videos | 0.6 GB |
| `cpr_test.zip` | the test videos (you need them to predict) | 2.6 GB |
| `flow_pose_subset.zip` | the report subset | 5 GB |

- **Download** them in [SCC OnDemand](https://scc-ondemand.bu.edu): open Files, use **Go To…** to enter `/projectnb/cs541/materials/cpr_challenge/downloads`, select a zip and click **Download**. From a terminal, `scp <your BU login>@scc1.bu.edu:/projectnb/cs541/materials/cpr_challenge/downloads/cpr_train.zip .` works too.
- **Unzip them all into one folder**, e.g. `unzip cpr_train.zip -d cpr_challenge`, then tell `cpr_data.py` where it is: `export CPR_DATA=~/cpr_challenge` in the shell, or `os.environ["CPR_DATA"] = ...` in Python before importing `cpr_data`.
- **Colab:** upload the zips to your Google Drive (about 8 GB in total; free Drive has 15 GB), then unzip them onto Colab's own disk at the start of every session. Don't unzip into Drive: it is very slow with thousands of files.

  ```python
  from google.colab import drive; drive.mount("/content/drive")
  !unzip -q /content/drive/MyDrive/cpr_train.zip -d /content/cpr_challenge   # add cpr_test.zip when you predict
  import os, sys
  os.environ["CPR_DATA"] = "/content/cpr_challenge"
  sys.path.insert(0, "/content/cpr_challenge/starter")
  from cpr_data import data_file, video_file, load_video, subset_dir
  ```
- **Your copy is for this course only.** Don't share it or post it anywhere (no public Drive links, no GitHub), and delete it when the course ends.

**3. Hugging Face (not open yet; staff will announce it on Piazza if necessary):** the same files at [`dghadiya/cs541-cpr-challenge`](https://huggingface.co/datasets/dghadiya/cs541-cpr-challenge), with no account or login needed.
  - Get the helper first: `hf download dghadiya/cs541-cpr-challenge starter/cpr_data.py --repo-type dataset --local-dir .` (needs `pip install huggingface_hub`, which Colab already has).
  - `download_all("train")` fetches every training video at once, which is faster than one at a time; fetch the test videos with `download_all("test")` when you're ready to predict. On Colab they disappear when the session ends; fetching the training videos again (~0.6 GB) is quick.
  - **Don't read from Hugging Face inside your training loop.** Work from the local files.

**Using `starter/cpr_data.py`:**
- `data_file("train.csv")` and `video_file(video_id, "train")` return a local path: your own copy if `CPR_DATA` is set, otherwise the SCC copy, otherwise a copy downloaded once from Hugging Face and cached.
- `subset_dir()` returns the report subset's folder. On Hugging Face the subset is a single file, `flow_pose_subset.zip` (~5 GB); `subset_dir()` downloads and unzips it once.

## Videos

- MP4 (H.264), **854 × 480 pixels, 25 frames per second**, about 6–55 s long (~16 s on average). All videos together are about 3 GB; the training videos alone are about 0.6 GB.
- Most video models take smaller inputs: resize as you load, e.g. `load_video(path, size=(224, 224))`.
- The original videos have a date/time overlay in the top-left corner; it has been **blacked out**.
- **Training videos** are named after their source recording: `<recording>__<class>__<round>__<camera>`. **Test videos** have random ids, and their start and end have been randomly trimmed by a few frames.
- **Sampling tip:** at 25 fps, one chest compression takes about 12–15 frames. If you subsample frames more sparsely than every ~6 frames, the compression motion becomes invisible to your model.

## Labels

`train.csv` has one row per video. `video_id`, `take`, `view` (camera `ch0`–`ch2`), `class_id` and `class_name` describe it, followed by **13 binary columns, one per error:**

| Column | Error | Column | Error |
|---|---|---|---|
| `e01_overlap_hands` | Overlap Hands | `e08_standing` | Standing |
| `e02_clenching_hands` | Clenching Hands | `e09_wrong_position` | Wrong Position |
| `e03_single_hand` | Single Hand | `e10_insufficient_pressing` | Insufficient Pressing |
| `e04_bending_arms` | Bending Arms | `e11_slow_frequency` | Slow Frequency |
| `e05_tilting_arms` | Tilting Arms | `e12_excessive_pressing` | Excessive Pressing |
| `e06_jump_pressing` | Jump Pressing | `e13_random_position_pressing` | Random Position Pressing |
| `e07_squatting` | Squatting | | |

- **"Correct" technique** (`class_id` 0) means all 13 columns are 0.
- Watch a few training videos per class to see what each error looks like.
- **Each recorded performance (`take`) was filmed by several cameras at once.** All of its videos share the same labels, so keep every view of a take on the same side of any train/validation split you make. Otherwise your validation score will be optimistic.

## Submitting

- **Kaggle competition:** <https://www.kaggle.com/t/6c646931df9a41a1bc34e7336ae1f51c> (an invitation link: please don't share it outside the class)
- Upload a CSV with **exactly the columns and rows of `sample_submission.csv`**: one row per test video and 13 columns of **0 or 1**. `starter.py` includes a format check.
- You can submit up to **2 times per day**. Before the deadline, select up to **2** submissions as final.
- **Your Kaggle team name must be your name as it appears on the BU student portal**, so we can match your score to you.
- **Metric: macro-F1.** F1 is computed separately for each of the 13 errors and then averaged, so every error counts equally, rare or common.
  - Predicting no errors scores 0.00.
  - Flagging every error on every video scores **0.25**.
- **Public vs private leaderboard:** the public leaderboard scores about 30% of the test set. Final rankings use the **other 70%**, which you never see during the competition, so don't overfit the public score.
- **Deadline:** Tuesday, December 8, 2026, 11:59 PM Boston time. Kaggle shows times in UTC, where this is 04:59 on December 9. **Individual work:** one student per team.

## Parts, points and report (100 points)

| What | Points | Due (11:59 PM Boston time) |
|---|---|---|
| Initial submission: a macro-F1 above 0.25 on the public leaderboard | 10 | Tuesday, November 24 |
| Parts 1–4 below | 75 | Tuesday, December 8 |
| Report | 15 | Tuesday, December 8 |

**Hand in on Gradescope by December 8:** your report as a PDF, and your code (a zip without data or model weights, or a Colab/GitHub link staff can view).

The top 3 on the private leaderboard will present their methods in class on Thursday, December 10.

1. **Part 1, RGB video model (40 points):** train a model on the RGB videos only and submit its predictions to the leaderboard. This is the required baseline.
2. **Part 2, optical flow (10 points):** run a state-of-the-art optical flow model, inference only, on the 100 videos in `flow_pose_subset/`, and compare its flow with the reference flow (format: `flow_pose_subset/README.md`). Report these three numbers:
   - **EPE (endpoint error):** the distance in pixels between your flow vector and the reference vector, averaged over all pixels and frame pairs. Clip your flow to ±20 px first, because the reference saturates there.
   - **Outlier rate:** the percentage of pixels whose endpoint error is above 3 px.
   - **Photometric error:** use the flow to warp frame k+1 back onto frame k, then take the mean absolute pixel difference from frame k. Compute it for your flow **and** for the reference flow. The reference (TV-L1) is not ground truth, so a large EPE alone doesn't tell you which of the two flows is wrong; this number needs no reference.
3. **Part 3, pose (10 points):** run a state-of-the-art **pose estimation** model, inference only, on the same subset and compare it with the provided reference keypoints. Compare on the 17 COCO keypoints, which are the reference's first 17 points (divide the reference coordinates by 4 to match the 960 × 540 frames). The reference keypoints were themselves produced by an algorithm (AlphaPose), so discuss what that means for your comparison.
4. **Part 4, ablation (15 points):** add optical flow and/or pose to your RGB model. Report when the extra inputs help, and when they don't.
   - **Mind the compute.** Running a flow model on every frame of all 5,328 videos at full size takes a day or more of GPU time. Run flow and pose only on the frames your model actually uses, e.g. 32–64 frames per video resized to about 224 px; that brings it down to around an hour. Start early: GPUs get busy near the deadline.

**Suggested models** (you are welcome to pick a different model of your choice):
- Optical flow: [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT).
- Pose: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (its pose models output the 17 COCO keypoints).

## Working on the SCC

```bash
# ONCE, before anything else: keep conda files and downloaded models out of your 10 GB home. Very Important!
module load miniconda
setup_scc_condarc.sh          # pick "cs541", then "Yes"
echo 'export HF_HOME=/projectnb/cs541/students/$USER/.cache/huggingface' >> ~/.bashrc
echo 'export TORCH_HOME=/projectnb/cs541/students/$USER/.cache/torch' >> ~/.bashrc

# software: PyTorch, OpenCV, transformers, etc.
module load miniconda academic-ml/fall-2026
conda activate fall-2026-pyt

# a GPU batch job: copy the example, change its last line to run your own script, submit it.
# It already asks for the right GPU (gpu_c=7.5: this PyTorch can't use older GPUs).
cp /projectnb/cs541/materials/cpr_challenge/starter/train_job.sh /projectnb/cs541/students/$USER/
cd /projectnb/cs541/students/$USER && qsub train_job.sh
qstat -u $USER                # qw = waiting, r = running; output goes to cpr_train.o<job id>
```

- **Interactive work:** use [SCC OnDemand](https://scc-ondemand.bu.edu). Choose Project `cs541`, put `module load miniconda academic-ml/fall-2026 && conda activate fall-2026-pyt` in the "Pre-Launch Command" box, and set the GPU compute capability to **7.5**.
- **End your session when you're done** (the red "Delete" button under My Interactive Sessions). Closing the browser tab does not stop it, and it keeps holding a GPU until its hours run out.
- **Your own files** go in `/projectnb/cs541/students/<your BU login>/` (only you and the course staff can open it), not in your home directory, which is small.
- **GPU time is shared by the whole class.** Request only what you need, and debug on a few videos before launching long jobs.
- **SCC questions go to the course staff first** (Piazza). Staff will escalate to Research Computing if needed.

## Rules

- Train only on the provided training videos and labels: **no external data.** You can train on several modalities, e.g. RGB frames + optical flow + pose. Pretrained models are allowed (ImageNet, Kinetics, off-the-shelf pose/depth/flow models, etc.).
- **Individual work: no collaboration and no code sharing.**
- **The data comes from the public CPR-Coach dataset.** Do not use the original dataset's composite-error videos, its test lists or its annotations. Do not try to identify which original recording a test video came from. Doing either is an academic-integrity violation.

## Credit

The data is derived from **CPR-Coach** (S. Wang et al., *CPR-Coach: Recognizing Composite Error Actions based on Single-class Training*, [arXiv:2309.11718](https://arxiv.org/abs/2309.11718)). It is for coursework only; do not redistribute it, including any copy you download.
