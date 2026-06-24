import streamlit as st
import sqlite3
import json
import os
import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt

# --- KONFIGURATION ---
DB_FILE = "stimm_rpg.db"
UPLOAD_DIR = "."  # Cloud-kompatibel im Hauptordner speichern

PITCH_FLOOR = 75.0
PITCH_CEILING = 500.0
MAX_FORMANT_HZ = 5500.0
F1_CLOSE_MAX = 450.0
F1_OPEN_MIN = 750.0
MIN_FRAMES_PER_BIN = 8

# --- DATENBANK FUNKTIONEN ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS idole (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            file_path TEXT,
            preview_path TEXT,
            pitch_median REAL,
            pitch_std REAL,
            f2_global REAL,
            f2_bins TEXT,
            jitter REAL,
            shimmer REAL,
            hnr REAL
        )
    ''')
    conn.commit()
    conn.close()

def speichere_idol_in_db(name, file_path, preview_path, features):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO idole 
            (name, file_path, preview_path, pitch_median, pitch_std, f2_global, f2_bins, jitter, shimmer, hnr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            name, file_path, preview_path,
            features["pitch_median"], features["pitch_std"], features["f2_global"],
            json.dumps(features["f2_bins"]), features["jitter"], features["shimmer"], features["hnr"]
        ))
        conn.commit()
    except Exception as e:
        st.error(f"Datenbankfehler: {e}")
    finally:
        conn.close()

def lade_alle_idole():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT name, file_path, preview_path, pitch_median, pitch_std, f2_global, f2_bins, jitter, shimmer, hnr FROM idole')
    rows = cursor.fetchall()
    conn.close()
    
    idole = {}
    for r in rows:
        idole[r[0]] = {
            "file_path": r[1], "preview_path": r[2], "pitch_median": r[3], "pitch_std": r[4], "f2_global": r[5],
            "f2_bins": json.loads(r[6]), "jitter": r[7], "shimmer": r[8], "hnr": r[9]
        }
    return idole

# --- AKUSTISCHE ANALYSE (Praat) ---
def analysiere_stimme(sound):
    ergebnis = {"valid": False}
    try:
        pitch_obj = sound.to_pitch(time_step=None, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        intensity_obj = sound.to_intensity()
        formant_obj = sound.to_formant_burg(time_step=None, max_number_of_formants=5, maximum_formant=MAX_FORMANT_HZ)
        
        zeiten = pitch_obj.xs()
        alle_intensitaeten = [intensity_obj.get_value(t) for t in zeiten if intensity_obj.get_value(t) is not None and not np.isnan(intensity_obj.get_value(t))]
        if not alle_intensitaeten: return ergebnis
        schwellenwert = max(alle_intensitaeten) - 25.0

        f0_w, f1_w, f2_w = [], [], []
        for t in zeiten:
            f0 = pitch_obj.get_value_at_time(t)
            if f0 is None or np.isnan(f0) or f0 <= 0: continue
            if intensity_obj.get_value(t) < schwellenwert: continue
            f1, f2 = formant_obj.get_value_at_time(1, t), formant_obj.get_value_at_time(2, t)
            if f1 is None or f2 is None or np.isnan(f1) or np.isnan(f2): continue
            f0_w.append(f0)
            f1_w.append(f1)
            f2_w.append(f2)

        if not f0_w: return ergebnis
        f0_arr, f1_arr, f2_arr = np.array(f0_w), np.array(f1_w), np.array(f2_w)
        
        bins = {"geschlossen": [], "mittel": [], "offen": []}
        for f1, f2 in zip(f1_arr, f2_arr):
            if f1 < F1_CLOSE_MAX: bins["geschlossen"].append(f2)
            elif f1 > F1_OPEN_MIN: bins["offen"].append(f2)
            else: bins["mittel"].append(f2)

        bin_stats = {}
        for k, v in bins.items():
            if len(v) >= MIN_FRAMES_PER_BIN: bin_stats[k] = {"f2_mean": float(np.mean(v)), "n": len(v)}

        point_process = call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
        harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=PITCH_FLOOR)
        hnr_w = harmonicity.values.flatten()
        hnr = float(np.mean(hnr_w[hnr_w > -200])) if hnr_w.size > 0 else np.nan

        return {
            "valid": True, "pitch_median": float(np.median(f0_arr)), "pitch_std": float(np.std(f0_arr)),
            "f2_global": float(np.mean(f2_arr)), "f2_bins": bin_stats, "jitter": jitter, "shimmer": shimmer, "hnr": hnr
        }
    except Exception:
        return ergebnis

# --- SCORES BERECHNEN ---
def berechne_scores(live, idol):
    p_diff = abs(idol["pitch_median"] - live["pitch_median"])
    p_score = max(0, min(100, int(100 - (p_diff * 2.5))))
    
    g_bins = [b for b in idol["f2_bins"] if b in live["f2_bins"]]
    if g_bins:
        sc, gw = [], []
        for b in g_bins:
            if live["f2_bins"][b]["f2_mean"] >= idol["f2_bins"][b]["f2_mean"]: sc.append(100)
            else: sc.append(max(0, min(100, int(100 - (idol["f2_bins"][b]["f2_mean"] - live["f2_bins"][b]["f2_mean"]) * 0.6))))
            gw.append(min(idol["f2_bins"][b]["n"], live["f2_bins"][b]["n"]))
        r_score = int(np.average(sc, weights=gw))
    else:
        r_score = max(0, min(100, int(100 - abs(idol["f2_global"] - live["f2_global"]) * 0.6)))

    pros_score = 100 if live["pitch_std"] >= idol["pitch_std"] else max(0, min(100, int(100 - (idol["pitch_std"] - live["pitch_std"]) * 5.0)))
    
    sq_sc = []
    if not np.isnan(live["jitter"]) and not np.isnan(idol["jitter"]): 
        sq_sc.append(100 if live["jitter"] <= idol["jitter"] else max(0, int(100 - (live["jitter"] - idol["jitter"]) * 30)))
    if not np.isnan(live["shimmer"]) and not np.isnan(idol["shimmer"]): 
        sq_sc.append(100 if live["shimmer"] <= idol["shimmer"] else max(0, int(100 - (live["shimmer"] - idol["shimmer"]) * 15)))
    if not np.isnan(live["hnr"]) and not np.isnan(idol["hnr"]): 
        sq_sc.append(100 if live["hnr"] >= idol["hnr"] else max(0, int(100 - (idol["hnr"] - live["hnr"]) * 5)))
    sq_score = int(np.mean(sq_sc)) if sq_sc else 50

    gesamt = int((p_score * 0.30) + (r_score * 0.35) + (pros_score * 0.15) + (sq_score * 0.20))
    return gesamt, p_score, r_score, pros_score, sq_score

# --- APP SETUP ---
init_db()
st.set_page_config(page_title="Stimm-RPG v2 Cloud", layout="wide")
st.title("🎙️ Stimm-RPG v2: Logopädisches Web-Training")

# AUTOMATISCHER IMPORT FÜR DEINE SPEZIFISCHEN DATEIEN (Jetzt mit FLAC-Previews)
IDOL_PAARE = [
    {"id": "84", "name": "Idol 84 (Durchschnitt)", "analysis": "chain_84.wav", "preview": "84_preview.flac"},
    {"id": "1462", "name": "Idol 1462 (Durchschnitt)", "analysis": "chain_1462.wav", "preview": "1462_preview.flac"}
]

aktuelle_idole = lade_alle_idole()

for idol in IDOL_PAARE:
    if idol["name"] not in aktuelle_idole:
        if os.path.exists(idol["analysis"]):
            with st.spinner(f"Importiere {idol['name']}... Bitte kurz warten."):
                features = analysiere_stimme(parselmouth.Sound(idol["analysis"]))
                if features["valid"]:
                    p_path = idol["preview"] if os.path.exists(idol["preview"]) else None
                    speichere_idol_in_db(idol["name"], idol["analysis"], p_path, features)
            st.rerun()
# --- SIDEBAR ---
st.sidebar.header("📁 Referenz-Datenbank")
alle_idole = lade_alle_idole()

if alle_idole:
    ausgewaehltes_idol_name = st.sidebar.selectbox("Wähle dein Trainings-Idol:", list(alle_idole.keys()))
    idol_daten = alle_idole[ausgewaehltes_idol_name]
    
    # Hörprobe abspielen
    st.sidebar.write("🎧 **Hörprobe / Vorkost (Einzelspur):**")
    if idol_daten.get("preview_path") and os.path.exists(idol_daten["preview_path"]):
        st.sidebar.audio(idol_daten["preview_path"], format="audio/wav")
    elif os.path.exists(idol_daten["file_path"]):
        st.sidebar.audio(idol_daten["file_path"], format="audio/wav")
        st.sidebar.caption("Hinweis: Keine separate Vorschau-Datei gefunden, spiele Analyse-Datei.")
    else:
        st.sidebar.warning("Keine Audiodatei für die Hörprobe gefunden.")
else:
    st.sidebar.info("Noch kein Idol vorhanden. Bitte lade eins hoch!")
    idol_daten = None

# Neues Idol mit getrennter Analyse & Hörprobe hochladen
with st.sidebar.expander("➕ Neues Referenz-Idol hochladen"):
    neu_name = st.text_input("Name des neuen Idols")
    neu_analysis_file = st.file_uploader("1. Analyse-Datei (60 Spuren nacheinander!)", type=["wav"])
    neu_preview_file = st.file_uploader("2. Optionale Hörprobe (Einzelspur-Vorschau)", type=["wav"])
    
    if st.button("Beide Analysieren & Speichern") and neu_name and neu_analysis_file:
        a_pfad = os.path.join(UPLOAD_DIR, f"{neu_name}_analysis.wav")
        with open(a_pfad, "wb") as f: f.write(neu_analysis_file.getbuffer())
        
        p_pfad = None
        if neu_preview_file:
            p_pfad = os.path.join(UPLOAD_DIR, f"{neu_name}_preview.wav")
            with open(p_pfad, "wb") as f: f.write(neu_preview_file.getbuffer())
            
        with st.spinner("Analysiere mathematische Merkmale..."):
            features = analysiere_stimme(parselmouth.Sound(a_pfad))
            if features["valid"]:
                speichere_idol_in_db(neu_name, a_pfad, p_pfad, features)
                st.success(f"'{neu_name}' erfolgreich in der Cloud-Datenbank gespeichert!")
                st.rerun()
            else:
                st.error("Analyse-Datei fehlerhaft. Keine Stimme erkannt.")

# --- HAUPTSEITE ---
st.subheader("📜 Phonetisch ausbalancierter Trainingssatz:")
st.info('“Während Yvonne früh am Ufer über glühende Kohlen schritt und dabei laut über ihre Lieblingsbücher, große Öko-Häuser und ungewöhnliche Pflanzenarten nachdachte...”')

col1, col2 = st.columns([1, 2])

with col1:
    st.write("### 🎤 Deine Live-Aufnahme")
    live_audio_file = st.audio_input("Klicke zum Aufnehmen")

if live_audio_file and idol_daten:
    with open("temp_live.wav", "wb") as f: f.write(live_audio_file.getbuffer())
    
    with st.spinner("Berechne dein Match..."):
        live_features = analysiere_stimme(parselmouth.Sound("temp_live.wav"))
        
        if live_features["valid"]:
            gesamt, p_sc, r_sc, pr_sc, q_sc = berechne_scores(live_features, idol_daten)
            
            with col2:
                st.write(f"### 🏆 Übereinstimmung mit '{ausgewaehltes_idol_name}': {gesamt}%")
                st.progress(gesamt / 100)
                
                # Feedback-Grafik
                fig, axs = plt.subplots(4, 1, figsize=(6, 9))
                plt.style.use('dark_background')
                
                # Pitch
                axs[0].barh(['Deine Tonhöhe'], [live_features["pitch_median"]], color='#3498db', height=0.4)
                axs[0].axvline(idol_daten["pitch_median"], color='white', linestyle='--', label=f'Idol ({idol_daten["pitch_median"]:.0f} Hz)')
                axs[0].set_xlim(0, 300)
                axs[0].legend(loc='upper right', fontsize=8)
                
                # Resonanz
                g_bins = [b for b in idol_daten["f2_bins"] if b in live_features["f2_bins"]]
                if g_bins:
                    y_pos = np.arange(len(g_bins))
                    axs[1].barh(y_pos - 0.2, [idol_daten["f2_bins"][b]["f2_mean"] for b in g_bins], height=0.35, color='white', alpha=0.4, label='Idol')
                    axs[1].barh(y_pos + 0.2, [live_features["f2_bins"][b]["f2_mean"] for b in g_bins], height=0.35, color='#9b59b6', label='Du')
                    axs[1].set_yticks(y_pos)
                    axs[1].set_yticklabels([b.capitalize() for b in g_bins])
                axs[1].set_xlim(0, 2800)
                axs[1].legend(loc='upper right', fontsize=8)
                
                # Melodie
                axs[2].barh(['Stimm-Melodie'], [live_features["pitch_std"]], color='#e67e22', height=0.4)
                axs[2].axvline(idol_daten["pitch_std"], color='white', linestyle='--', label=f'Idol ({idol_daten["pitch_std"]:.1f} Hz)')
                axs[2].set_xlim(0, 80)
                axs[2].legend(loc='upper right', fontsize=8)
                
                # Qualität
                sq_labels = ['Jitter (%)', 'Shimmer (%)', 'HNR (dB)']
                y_pos_sq = np.arange(len(sq_labels))
                axs[3].barh(y_pos_sq - 0.2, [idol_daten["jitter"], idol_daten["shimmer"], idol_daten["hnr"]], height=0.35, color='white', alpha=0.4, label='Idol')
                axs[3].barh(y_pos_sq + 0.2, [live_features["jitter"], live_features["shimmer"], live_features["hnr"]], height=0.35, color='#2ecc71', label='Du')
                axs[3].set_yticks(y_pos_sq)
                axs[3].set_yticklabels(sq_labels)
                axs[3].set_xlim(0, max(35, live_features["hnr"]+5, idol_daten["hnr"]+5))
                axs[3].legend(loc='upper right', fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.error("Stimme zu leise oder unklar – bitte lauter sprechen!")
