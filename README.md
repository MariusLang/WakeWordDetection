# WakeWordDetection

## rsync to sync all files in git repo

```bash
git ls-files | rsync -av --files-from=- ./ marius@192.168.178.62:/home/marius/Downloads/WakeWordDetection
```

```bash
git ls-files | rsync -av --files-from=- ./ pi@192.168.178.194:/home/pi/WakeWordDetection
```
