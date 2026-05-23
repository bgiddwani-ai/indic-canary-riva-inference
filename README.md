# Indic Canary RIVA Offline Inference

End-to-end guide for deploying the [Indic Canary](https://indicwhisper.objectstore.e2enetworks.net/indic-canary/) ASR model on NVIDIA RIVA for **offline (non-streaming) inference** across 22 Indian languages.

The pipeline has three stages:

1. **NeMo → RIVA** — convert the `.nemo` checkpoint to a `.riva` artifact using the `nemo2riva` tool.
2. **RIVA build & deploy** — package the `.riva` into an `.rmir`, deploy it as a Triton model repository, and start the RIVA server.
3. **Client inference** — transcribe audio files using the `nvidia-riva-client` Python SDK.

---

## Prerequisites

- NVIDIA GPU with recent drivers (tested with a single GPU, `device=0`).
- Docker with NVIDIA Container Toolkit installed.
- Access to NGC containers:
  - `nvcr.io/nvidia/nemo:25.02`
  - `nvcr.io/nvidia/riva/riva-speech:2.19.0`
- ~20 GB free disk for the model, intermediate artifacts, and Triton repository.

---

## 1. Environment & Model Download

Set up the working and data directories on the **host**:

```bash
export WORKDIR="/userhome/home/bgiddwani/bodhan"
export DATADIR="/userhome/home/bgiddwani/data"

mkdir -p $WORKDIR
mkdir -p $DATADIR/rmir
mkdir -p $DATADIR/models
```

Download the Indic Canary NeMo source and checkpoint:

```bash
cd $WORKDIR

# NeMo source tree used during conversion
wget NeMo repo
tar -xvf canary-nemo.tar.gz

# Model checkpoint
wget NeMo ckpt
```

After this step `$WORKDIR` should contain at least:

```
NeMo/                  # extracted source tree
indic-canary.nemo      # model checkpoint
```

---

## 2. NeMo → RIVA Conversion

Launch the NeMo container with `$WORKDIR` and `$DATADIR` mounted:

```bash
docker run -it --rm \
  --gpus '"device=0"' \
  -v $WORKDIR:/home/bodhan \
  -v $DATADIR:/data \
  nvcr.io/nvidia/nemo:25.02
```

Inside the container, install `nemo2riva` and reinstall the bundled NeMo source so the converter sees the Canary code paths:

```bash
pip install nvidia-pyindex
pip install --extra-index-url https://pypi.nvidia.com nvidia-eff

git clone https://github.com/nvidia-riva/nemo2riva.git
cd nemo2riva
pip install --no-deps .

cd ../NeMo/
bash reinstall.sh
```

Run the conversion. `--key` is the encryption key embedded in the `.riva` artifact (here we use `bodhan`; pick any string but remember it — the build step needs the same key):

```bash
export PYTHONPATH=/home/bodhan/NeMo:$PYTHONPATH

nemo2riva \
  --key=bodhan \
  --out /home/bodhan/riva/indic-canary.riva \
  --format nemo \
  /home/bodhan/indic-canary.nemo
```

The result is `/home/bodhan/riva/indic-canary.riva` on the host (`$WORKDIR/riva/indic-canary.riva`). You can exit this container.

---

## 3. RIVA Build & Deploy

Launch the RIVA Speech container with the same mounts:

```bash
docker run -it --rm \
  --gpus '"device=0"' \
  -v $WORKDIR:/home/bodhan \
  -v $DATADIR:/data \
  -p 50051:50051 \
  nvcr.io/nvidia/riva/riva-speech:2.19.0
```

> **Important:** mount `$DATADIR/models` to `/data/models` inside the container. `riva-deploy` writes the Triton model repository there, and `start-riva` reads from `/data/models` by default.

### 3.1 Patch the in-container environment

The default RIVA 2.19.0 image needs a few package adjustments to host the Canary decoder cleanly:

```bash
pip install -U --no-cache-dir --ignore-installed \
  setuptools==75.3.4 pybind11 wheel pip

apt-get remove -y python3-blinker

pip uninstall -y nemo_toolkit
cd /home/bodhan/NeMo
bash reinstall.sh

pip install -U --no-cache-dir --ignore-installed 'protobuf>=5.26,<6'
pip install "transformers==4.48.3"
```

### 3.2 Build the RMIR

`riva-build` packages the `.riva` into an `.rmir` ready for Triton deployment. The flags below configure an **offline Canary 1B multi-language ASR** model with a 30 s chunk size and no streaming padding:

```bash
riva-build speech_recognition \
  "/home/bodhan/riva/rmir/indic-canary-multi.rmir":"bodhan" \
  "/home/bodhan/riva/indic-canary.riva":"bodhan" \
  --offline \
  --name=canary-1b-multi-asr-offline \
  --return_separate_utterances=True \
  --chunk_size 30 \
  --left_padding_size 0 \
  --right_padding_size 0 \
  --decoder_type nemo \
  --nemo_decoder.nemo_decoder_type canary \
  --feature_extractor_type torch \
  --torch_feature_type nemo \
  --featurizer.norm_per_feature true \
  --max_batch_size 8 \
  --featurizer.use_utterance_norm_params=False \
  --featurizer.precalc_norm_params False \
  --featurizer.max_batch_size=128 \
  --featurizer.max_execution_batch_size=128 \
  --language_code=as,bn,brx,doi,en,gu,hi,kn,ks,kok,mai,ml,mni,mr,ne,or,pa,sa,sat,sd,ta,te,ur
```

The `:bodhan` suffixes on the input/output paths are encryption keys — they must match the key used in `nemo2riva`.

### 3.3 Deploy to the Triton repository

```bash
riva-deploy /home/bodhan/riva/rmir/indic-canary-multi.rmir:bodhan /data/models -f
```

`-f` overwrites any existing model directory of the same name. After this completes you should see a directory like:

```
/data/models/riva-nemo-canary-1b-multi-asr-offline-am-streaming-offline/
```

### 3.4 Replace the offline ASR model script

Replace the auto-generated `nemo_offline_asr_model.py` with the patched version provided alongside this README (it adjusts the decode loop for Canary multi-language inference):

```bash
rm -rf /data/models/riva-nemo-canary-1b-multi-asr-offline-am-streaming-offline/1/nemo_offline_asr_model.py
cp nemo_offline_asr_model.py /data/models/riva-nemo-canary-1b-multi-asr-offline-am-streaming-offline/1/
```

### 3.5 Start the RIVA server

`start-riva` automatically loads the repository at `/data/models`. NLP and TTS services are disabled since only ASR is needed here:

```bash
start-riva \
  --asr_service=true \
  --nlp_service=false \
  --tts_service=false \
  --riva-uri=0.0.0.0:50051
```

The server is ready when it logs that all models are `READY` and the gRPC endpoint is listening on `0.0.0.0:50051`.

---

## 4. Client Inference

Open a **separate container** (the NeMo image works fine) on the same host, joining the host network so it can reach `localhost:50051`:

```bash
docker run -it --rm \
  --gpus '"device=0"' \
  -v $WORKDIR:/home/bodhan \
  -v $DATADIR:/data \
  --net=host \
  nvcr.io/nvidia/nemo:25.02
```

Install the RIVA Python client and grab the official client scripts:

```bash
pip install nvidia-riva-client
git clone -b release/2.19.0 https://github.com/nvidia-riva/python-clients.git
```

Transcribe a file (Hindi example):

```bash
python3 python-clients/scripts/asr/transcribe_file_offline.py \
  --server 0.0.0.0:50051 \
  --language hi \
  --input-file /path/to/hi-IN_sample.wav
```

Swap `--language` for any of the supported codes below to transcribe other languages. Audio should be 16 kHz mono WAV for best results; resample first if needed.

---

## Supported Languages

The deployed model accepts the following BCP-47–style codes (passed via `--language` on the client or `--language_code` at build time):

| Code  | Language        | Code  | Language        | Code  | Language        |
|-------|-----------------|-------|-----------------|-------|-----------------|
| `as`  | Assamese        | `bn`  | Bengali         | `brx` | Bodo            |
| `doi` | Dogri           | `en`  | English         | `gu`  | Gujarati        |
| `hi`  | Hindi           | `kn`  | Kannada         | `ks`  | Kashmiri        |
| `kok` | Konkani         | `mai` | Maithili        | `ml`  | Malayalam       |
| `mni` | Manipuri        | `mr`  | Marathi         | `ne`  | Nepali          |
| `or`  | Odia            | `pa`  | Punjabi         | `sa`  | Sanskrit        |
| `sat` | Santali         | `sd`  | Sindhi          | `ta`  | Tamil           |
| `te`  | Telugu          | `ur`  | Urdu            |       |                 |

---

## Notes & Troubleshooting

- **Encryption key consistency.** The same key (`bodhan` above) is used in `nemo2riva`, in both arguments to `riva-build`, and in `riva-deploy`. Changing it in any one place will cause the next stage to fail with a decryption error.
- **Mount paths.** `$DATADIR/models` must be mounted to `/data/models` inside the RIVA container — `start-riva` does not accept an alternative repository path on the command line in this image.
- **Container reuse.** Steps 2 and 3 are intentionally split across `nemo:25.02` and `riva-speech:2.19.0` because the two toolchains have incompatible Python environments. Don't try to combine them.
- **Offline only.** This build disables streaming (`--offline`, zero padding). For streaming inference, rebuild the RMIR with non-zero `--left_padding_size` / `--right_padding_size` and remove `--offline`.
- **Chunk size.** `--chunk_size 30` means audio up to 30 s per utterance. Longer files are split by the client script automatically; for very long files, prefer the offline transcription script over the streaming one.
- **Patched decoder script.** Forgetting step 3.4 (copying `nemo_offline_asr_model.py`) is the most common cause of incorrect or empty transcriptions — the auto-generated script doesn't handle Canary's multi-language prompt format.
