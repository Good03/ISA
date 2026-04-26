# Installation Manual — MNM Terminal

**Authors:** Bc. Mykyta Prykhodko, Bc. Nikita Koliasnikov

---

## 1. Prerequisites

You need a working Docker installation. The application is delivered as a
Compose project, so you also need the `docker compose` plugin (bundled with
Docker Desktop on Windows / macOS).

### 1.1 Install Docker

| Platform | Install link | Notes |
|---|---|---|
| Windows 10/11 | <https://docs.docker.com/desktop/install/windows-install/> | Enable WSL 2 backend (recommended). |
| macOS (Apple Silicon or Intel) | <https://docs.docker.com/desktop/install/mac-install/> | Choose the matching CPU build. |
| Ubuntu / Debian | <https://docs.docker.com/engine/install/ubuntu/> | Install the `docker-compose-plugin` package. |
| Fedora / RHEL | <https://docs.docker.com/engine/install/fedora/> | — |

---

## 2. Dataset

The historical price CSV is inside the `data/` folder, but **not** included in the image.

---

## 3. Build &amp; start the container

From the project root (the folder containing `docker-compose.yml`):

```bash
docker compose up --build
```
---

## 4. Open the app

Once the log shows `Starting server → http://localhost:8080`

> **<http://localhost:8080>**

If port 8080 is already in use, change the `ports` mapping in `docker-compose.yml`

---

## 5. Stop and clean up

| Goal                                   | Command                       |
|----------------------------------------|-------------------------------|
| Stop the running container             | `docker compose down`         |
| Stop **and** delete trained models     | `docker compose down -v`      |
| Re-build after editing source code     | `docker compose up --build`   |
| Free disk space (image + dangling)     | `docker system prune`         |


---

### If you want to choose which tickers to pretrain

Edit `project/train.py`:

```python
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]    # ← edit
```

Rebuild the image and run `docker compose down -v && docker compose up --build`.
