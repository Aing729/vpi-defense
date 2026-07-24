# Label Studio deployment

The `docker-compose.yml` in this directory is portable — the same stack runs
locally and on all major container hosts. Pick whichever fits your budget
and privacy needs.

## Recommendation summary

| Host                | Persistent storage       | Cold start | Free tier reality |
|---------------------|--------------------------|------------|-------------------|
| **HF Spaces (Docker)**  | 2 GB included, 24/7 uptime | none  | best for this project |
| Local Docker        | your disk                | none       | requires Docker on the machine |
| Render (paid)       | $0.25 / GB / month       | none       | ~$10/month total |
| Render (free)       | **none** — data resets   | 30-60 s    | annotation DB is lost |
| Fly.io (with volume)| paid volume required     | none       | small monthly cost |

**HuggingFace Spaces is the easiest fit** for this project because Docker
Spaces include 2 GB of persistent storage and stay online 24/7 on the free
plan. Setup takes about 10 minutes.

## Common prerequisites

1. Set credentials before the first run. Never commit these.

   ```bash
   cp .env.example .env      # if you keep a template
   # edit .env:
   #   LABEL_STUDIO_USERNAME=you@example.com
   #   LABEL_STUDIO_PASSWORD=<random string>
   ```

2. The stack expects images under `data/` (project root). The compose file
   mounts `../../data` read-only into the container at `/label-studio/images`.
   Point Label Studio at that path when you create a project.

3. Once up, import the labeling config from `src/vpi_defense/labeling/config.xml`
   under the project's **Settings → Labeling Interface → Code** tab.

## Local Docker

```bash
cd deploy/label-studio
docker compose up -d
open http://localhost:8080
```

Log in with the credentials you set. Import the labeling config from
`src/vpi_defense/labeling/config.xml`.

## HuggingFace Spaces (Docker)

1. Create a new Space, choose **Docker → Blank template**, set visibility
   to Private.
2. Push a repository containing a `Dockerfile`:

   ```dockerfile
   FROM heartexlabs/label-studio:latest
   ENV LABEL_STUDIO_HOST=""
   ENV LABEL_STUDIO_BASE_DATA_DIR=/data
   EXPOSE 8080
   ```

3. In the Space settings, add secrets `LABEL_STUDIO_USERNAME` and
   `LABEL_STUDIO_PASSWORD`.
4. Add a `Persistent Storage` (2 GB free) mounted at `/data`.
5. Restart the Space. Label Studio is live at
   `https://<user>-<space>.hf.space`.

For image hosting, use the same trick as local: upload your `data/`
directory to the Space's persistent volume, or serve images via public
Google Drive share URLs.

## Fly.io

```bash
fly launch --image heartexlabs/label-studio:latest
fly volumes create ls_data --size 3   # $3-ish / month
fly deploy
```

Set the same env vars via `fly secrets set`.

## Render (paid tier only)

Free tier does NOT support persistent disks — annotation data will be
wiped on every redeploy. Only use Render for this when you're on a paid
plan and have a persistent disk attached to `/label-studio/data`.

## After deployment: importing the labeling config

1. Create a new project in Label Studio's UI.
2. Go to **Settings → Labeling Interface → Code**.
3. Paste the contents of `src/vpi_defense/labeling/config.xml`.
4. Save. The project is now ready to accept tasks with our schema.

## Exporting and converting annotations

Once you've labeled some tasks, export them:

1. In the Data Manager, click **Export → JSON**. Save the file.
2. Convert to our schema:

   ```bash
   python -m vpi_defense.labeling \
       --in  data/labels/export-2026-07-16.json \
       --out data/labels/regions.jsonl
   ```

The output JSONL contains `RegionAnnotation` records ready for the
downstream analysis (M6) and integration into a `DatasetBundle`.
