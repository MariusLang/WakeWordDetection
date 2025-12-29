import os
import json
import numpy as np
from datetime import datetime
import hashlib
import shutil


def hash_mel(mel: np.ndarray):
    mel_bytes = mel.tobytes()
    return hashlib.sha1(mel_bytes).hexdigest()[:12]

def get_dataset_manager_for_dir(root_dir, dir_name, json_name, label):
    set_manager = MelSetManager(root_dir, dir_name, json_name)

    full_dir = os.path.join(root_dir, dir_name)

    for filename in os.listdir(full_dir):
        if not filename.endswith(".npy"):
            continue

        set_manager.save_meta_for_entry(filename, label)

    set_manager._save_json()
    print(f"Created JSON for {len(set_manager.meta)} files")

    return set_manager

class MelSetManager:

    def __init__(self, root_dir, save_dir, json_name):
        self.root_dir = root_dir
        self.json_path = os.path.join(root_dir, json_name)
        self.save_dir = os.path.join(root_dir, save_dir)

        os.makedirs(self.save_dir, exist_ok=True)

        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                try:
                    self.meta = json.load(f)
                    print(f"[MelSaver] Loaded existing JSON: {len(self.meta)} entries")
                except:
                    print("[MelSaver] JSON exists but could not be parsed. Starting empty.")
                    self.meta = []
        else:
            self.meta = []
            print("[MelSaver] New JSON will be created.")

        self.existing_hashes = {os.path.basename(item["npy_path"]).replace(".npy", "")
                                for item in self.meta}

    def save(self, mel, label):
        mel_hash = "mel_" + hash_mel(mel)
        filename = mel_hash + ".npy"
        full_path = os.path.join(self.save_dir, filename)

        if mel_hash in self.existing_hashes:
            print(f"[MelSaver] Already exists: {filename}, skip saving")
        else:
            np.save(full_path,mel)
            self.existing_hashes.add(mel_hash)
            print(f"[MelSaver] Saved: {filename}")

        entry = self.save_meta_for_entry(filename, label)
        #self._save_json()
        return entry

    def save_meta_for_entry(self, file_name, label):
        entry = {
            "npy_path": os.path.join(self.save_dir, file_name),
            "label": int(label),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.meta.append(entry)
        return entry

    def _save_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.meta, f, indent=4)
    def get_meta(self):
        return self.meta

    def merge(self, other_saver):
        print(f"[MelSaver] Merging dataset from: {other_saver.root_dir}")

        copied = 0
        skipped = 0

        for entry in other_saver.meta:
            src_rel = entry["npy_path"]
            src = os.path.join(other_saver.root_dir, src_rel)

            name = os.path.basename(src)
            mel_hash = name.replace(".npy", "")
            dst = os.path.join(self.save_dir, name)

            if mel_hash in self.existing_hashes:
                skipped += 1
                continue

            shutil.copy2(src, dst)
            self.existing_hashes.add(mel_hash)

            new_entry = entry.copy()
            new_entry["npy_path"] = os.path.join(self.save_dir, name)
            self.meta.append(new_entry)
            copied += 1

        self._save_json()

        print(f"[MelSaver] Merge complete. Copied: {copied}, skipped (duplicates): {skipped}")

    def JSON_union(self, other_saver, output_json):
        merged = {}

        for entry in self.meta:
            key = os.path.basename(entry["npy_path"])
            merged[key] = entry

        for entry in other_saver.meta:
            key = os.path.basename(entry["npy_path"])
            if key not in merged:
                merged[key] = entry

        merged_list = list(merged.values())

        output_path = os.path.join(self.root_dir, output_json)

        with open(output_path, "w") as f:
            json.dump(merged_list, f, indent=4)

        print(
            f"[MelSaver] JSON union saved: {output_json} "
            f"(total: {len(merged_list)}, "
            f"self: {len(self.meta)}, other: {len(other_saver.meta)})"
        )

        return merged_list

    def split_dataset(self, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1 ,seed=42 ,prefix=""):
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        rng = np.random.default_rng(seed)

        indices = np.arange(len(self.meta))
        rng.shuffle(indices)

        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_dev = int(n_total * dev_ratio)

        train_idx = indices[:n_train]
        dev_idx = indices[n_train:n_train + n_dev]
        test_idx = indices[n_train + n_dev:]

        def subset(idxs):
            return [self.meta[i] for i in idxs]

        splits = {
            "train": subset(train_idx),
            "dev": subset(dev_idx),
            "test": subset(test_idx),
        }

        for name, data in splits.items():
            file_name = f"{prefix}{name}.json"
            path = os.path.join(self.root_dir, file_name)

            with open(path, "w") as f:
                json.dump(data, f, indent=4)

            print(f"[MelSaver] {name}: {len(data)} samples → {file_name}")

        return splits