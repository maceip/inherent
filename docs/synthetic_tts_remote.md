# Remote synthetic TTS (person + event)

Generate `hasPersonContext` and `hasEventContext` training audio on a Linux machine
with a real GPU. Default engine is [Supertonic 3](https://huggingface.co/Supertone/supertonic-3)
(fast, `pip install supertonic`). OpenF5 remains available via
`synthetic_tts_engine: openf5-tts` and `INHERENT_OPENF5_MODEL`.

## Quick start (remote)

```bash
git clone https://github.com/maceip/inherent.git
cd inherent
chmod +x scripts/run_synthetic_remote.sh
./scripts/run_synthetic_remote.sh
tail -f artifacts/synthetic_remote.log
```

Config: `configs/synthetic_person_event_remote.yaml` (32k clips, Supertonic, mic augment).

## Resume from the Mac partial run

The laptop already generated ~3.6k OpenF5 clips under `data/synthetic_audio/`. Copy
them to the remote box so synthesis skips finished `(head, voice, transcript)` keys:

```bash
rsync -avz --progress \
  data/synthetic_person_event_manifest.csv.partial \
  data/synthetic_audio/hasPersonContext \
  data/synthetic_audio/hasEventContext \
  USER@REMOTE:~/inherent/data/
```

Then run `./scripts/run_synthetic_remote.sh` on the remote host. Remaining clips use
Supertonic unless you set `synthetic_tts_engine: openf5-tts` in the config.

## Slice vs full

| Config | Clips | Engine |
|--------|-------|--------|
| `synthetic_person_event_slice.yaml` | 2,000 | OpenF5 (local) |
| `synthetic_person_event_remote.yaml` | 32,000 | Supertonic (remote) |

Finalize a smaller eval/train manifest:

```bash
PYTHONPATH=src python -m inherent.scripts.finalize_synthetic_slice \
  --input data/synthetic_person_event_manifest.csv.partial \
  --output data/synthetic_person_event_slice_manifest.csv \
  --config configs/synthetic_person_event_slice.yaml
```

## Dramabox

[ResembleAI/Dramabox](https://huggingface.co/ResembleAI/Dramabox) needs ~24 GB VRAM and a
separate install; add as a third `synthetic_tts_engine` after benchmark completes.
