"""Klasifikasi status akademik mahasiswa menggunakan LLM (Groq API).
File ini dipakai saat EVALUASI (Modul 2A) dan saat DEPLOY di Streamlit (Modul 2B)
agar perilaku model yang dievaluasi sama persis dengan yang di-deploy."""
import json
import re
import time

import groq

# nama fitur yang ditampilkan ke LLM
NAMA_FITUR = {
    "ipk": "IPK",
    "kehadiran": "Kehadiran (%)",
    "jam_belajar": "Jam belajar mandiri per minggu",
    "tugas_terkumpul": "Tugas terkumpul (%)",
    "nilai_uts": "Nilai UTS",
    "aktif_organisasi": "Aktif organisasi",
}


def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_data(data):
    """dict data mahasiswa -> teks satu baris untuk LLM."""
    return "; ".join(f"{nama}: {data[kolom]}" for kolom, nama in NAMA_FITUR.items())


def build_messages(cfg, data):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot", []):                      # contoh (few-shot), boleh kosong
        msgs.append({"role": "user", "content": format_data(ex["data"])})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": format_data(data)})
    return msgs


def parse_label(output, labels):
    """Rapikan jawaban model: 'waspada.' -> 'Waspada'. Di luar daftar label -> 'invalid'."""
    jawab = re.sub(r"[^a-z]", "", output.lower())
    peta = {label.lower(): label for label in labels}
    return peta.get(jawab, "invalid")


def classify(client, cfg, data, max_retry=3):
    for percobaan in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                messages=build_messages(cfg, data),
                **cfg.get("params", {}),                     # parameter tambahan model (opsional), mis. reasoning
            )
            return parse_label(resp.choices[0].message.content, cfg["labels"])
        except groq.RateLimitError:
            time.sleep(15 * (percobaan + 1))                # kena batas kuota: tunggu lalu coba lagi
        except groq.APIConnectionError:
            time.sleep(10 * (percobaan + 1))                # koneksi internet terputus: tunggu lalu coba lagi
    return "invalid"
